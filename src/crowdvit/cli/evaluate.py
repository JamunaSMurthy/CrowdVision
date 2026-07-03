"""Evaluate a CrowdViT checkpoint. Usage:

  python scripts/evaluate.py --config configs/datasets/kinetics400.yaml \
      --checkpoint outputs/crowdvit_k400/checkpoint_best.pt
"""

from __future__ import annotations

import argparse

import torch

from crowdvit.cli.common import build_config_from_cli, build_test_dataloader
from crowdvit.engine.evaluator import evaluate
from crowdvit.models.crowdvit import CrowdViT
from crowdvit.utils.checkpoint import load_checkpoint


def main(argv: list[str] | None = None) -> None:
    cfg, remaining = build_config_from_cli(argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = parser.parse_known_args(remaining)

    device = torch.device(args.device)
    model = CrowdViT(cfg.model).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    test_loader = build_test_dataloader(cfg)
    metrics = evaluate(model, test_loader, device, cfg.model.num_classes, amp=cfg.optim.amp)

    print(f"Top-1: {metrics['top1']:.2f}  Top-5: {metrics['top5']:.2f}  mAP: {metrics['mAP']:.2f}")


if __name__ == "__main__":
    main()
