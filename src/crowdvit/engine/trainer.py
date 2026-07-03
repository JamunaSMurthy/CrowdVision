"""Training loop: AdamW, cosine decay with linear warm-up, mixed precision,
gradient clipping, checkpointing, and the three-term CrowdViT loss (Section
"Implementation Details" of the paper).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crowdvit.config import Config
from crowdvit.engine.evaluator import evaluate
from crowdvit.losses.combined import CrowdViTLoss
from crowdvit.models.crowdvit import CrowdViT
from crowdvit.utils.checkpoint import load_checkpoint, save_checkpoint
from crowdvit.utils.distributed import barrier, is_main_process
from crowdvit.utils.logging import TensorBoardLogger, get_logger
from crowdvit.utils.lr_scheduler import build_cosine_warmup_scheduler


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: CrowdViT,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn = CrowdViTLoss(cfg.loss)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=tuple(cfg.optim.betas),
        )

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * cfg.optim.epochs
        warmup_steps = steps_per_epoch * cfg.optim.warmup_epochs
        self.scheduler = build_cosine_warmup_scheduler(self.optimizer, warmup_steps, total_steps)

        amp_enabled = cfg.optim.amp and device.type == "cuda"
        self.amp_enabled = amp_enabled
        self.scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)

        self.output_dir = Path(cfg.train.output_dir)
        self.logger = get_logger("crowdvit.train", self.output_dir / "train.log")
        self.tb = TensorBoardLogger(self.output_dir / "tensorboard", enabled=is_main_process())

        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = -1.0

        if cfg.train.resume:
            state = load_checkpoint(
                cfg.train.resume, self.model, self.optimizer, self.scheduler, map_location=device
            )
            self.start_epoch = state["epoch"] + 1
            self.best_metric = state.get("best_metric") or -1.0

    def train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
        num_steps = len(self.train_loader)

        progress = tqdm(
            self.train_loader,
            total=num_steps,
            desc=f"epoch {epoch}",
            disable=not is_main_process(),
            leave=False,
        )
        for step, (clips, targets) in enumerate(progress):
            clips = clips.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                output = self.model(clips)
                loss_out = self.loss_fn(output, targets)

            self.scaler.scale(loss_out.total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            progress.set_postfix(loss=f"{loss_out.total.item():.4f}")

            if step % self.cfg.train.log_interval == 0 and is_main_process():
                lr = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f"epoch {epoch} step {step}/{num_steps} "
                    f"loss {loss_out.total.item():.4f} ce {loss_out.ce.item():.4f} "
                    f"temp {loss_out.temp.item():.4f} view {loss_out.view.item():.4f} lr {lr:.2e}"
                )
                self.tb.log_scalar("train/loss", loss_out.total.item(), self.global_step)
                self.tb.log_scalar("train/ce", loss_out.ce.item(), self.global_step)
                self.tb.log_scalar("train/temp", loss_out.temp.item(), self.global_step)
                self.tb.log_scalar("train/view", loss_out.view.item(), self.global_step)
                self.tb.log_scalar("train/lr", lr, self.global_step)

    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.cfg.optim.epochs):
            self.train_one_epoch(epoch)

            if self.val_loader is not None and is_main_process():
                metrics = evaluate(
                    self.model,
                    self.val_loader,
                    self.device,
                    self.cfg.model.num_classes,
                    amp=self.amp_enabled,
                )
                self.logger.info(
                    f"epoch {epoch} val top1 {metrics['top1']:.2f} "
                    f"top5 {metrics['top5']:.2f} mAP {metrics['mAP']:.2f}"
                )
                for name, value in metrics.items():
                    self.tb.log_scalar(f"val/{name}", value, epoch)

                is_best = metrics["top1"] > self.best_metric
                if is_best:
                    self.best_metric = metrics["top1"]

                if epoch % self.cfg.train.checkpoint_interval == 0:
                    save_checkpoint(
                        self.output_dir / "checkpoint_last.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        self.best_metric,
                    )
                if is_best:
                    save_checkpoint(
                        self.output_dir / "checkpoint_best.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        self.best_metric,
                    )

            barrier()

        self.tb.close()
