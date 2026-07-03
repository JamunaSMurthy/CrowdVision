"""Measure params / GFLOPs / FPS for a config, matching the columns of the
paper's accuracy-efficiency trade-off table. Reported paper numbers come
from the authors' own hardware (4xV100 for GPU FPS, Jetson Nano for edge
FPS) — this script measures your hardware, not the paper's exact figures.

  python scripts/benchmark_fps.py --config configs/base.yaml --device cuda
  python scripts/benchmark_fps.py --config configs/base.yaml --device cpu   # Jetson-class proxy
"""

from __future__ import annotations

import argparse
import time

import torch

from crowdvit.cli.common import build_config_from_cli
from crowdvit.models.crowdvit import CrowdViT
from crowdvit.utils.flops import count_gflops, count_parameters


def main(argv: list[str] | None = None) -> None:
    cfg, remaining = build_config_from_cli(argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args, _ = parser.parse_known_args(remaining)

    device = torch.device(args.device)
    model = CrowdViT(cfg.model).to(device).eval()
    clip = torch.randn(
        args.batch_size,
        cfg.model.num_frames,
        3,
        cfg.model.image_size,
        cfg.model.image_size,
        device=device,
    )

    params = count_parameters(model)
    gflops = count_gflops(model, clip)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(clip)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(args.iters):
            model(clip)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    fps = (args.iters * args.batch_size) / elapsed
    print(
        f"Params: {params / 1e6:.2f}M  GFLOPs: {gflops:.1f}  FPS: {fps:.1f}  "
        f"(device={args.device}, batch={args.batch_size})"
    )


if __name__ == "__main__":
    main()
