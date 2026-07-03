"""Train CrowdViT. Usage:

  python scripts/train.py --config configs/datasets/kinetics400.yaml \
      --train.output_dir outputs/crowdvit_k400

  torchrun --nproc_per_node=N scripts/train.py --config ... --train.distributed true
"""

from __future__ import annotations

from crowdvit.cli.common import build_config_from_cli, build_dataloaders
from crowdvit.engine.trainer import Trainer
from crowdvit.models.crowdvit import CrowdViT
from crowdvit.utils.distributed import init_distributed, wrap_model_ddp
from crowdvit.utils.seed import set_seed


def main(argv: list[str] | None = None) -> None:
    cfg, _ = build_config_from_cli(argv)
    set_seed(cfg.train.seed)

    rank, world_size, device = init_distributed(cfg.train.device)
    cfg.train.distributed = world_size > 1

    train_loader, val_loader = build_dataloaders(cfg, distributed=cfg.train.distributed)

    model = CrowdViT(cfg.model).to(device)
    if cfg.train.distributed:
        model = wrap_model_ddp(model, device)

    trainer = Trainer(cfg, model, train_loader, val_loader, device)
    trainer.fit()


if __name__ == "__main__":
    main()
