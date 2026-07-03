"""Export a CrowdViT checkpoint to a static-shape ONNX graph for edge
deployment (onnxruntime / TensorRT on Jetson-class devices).

  python scripts/export_onnx.py --config configs/base.yaml \
      --checkpoint outputs/crowdvit_k400/checkpoint_best.pt --out crowdvit.onnx
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from crowdvit.cli.common import build_config_from_cli
from crowdvit.models.crowdvit import CrowdViT
from crowdvit.utils.checkpoint import load_checkpoint


class _LogitsOnly(nn.Module):
    """ONNX export requires a plain tensor output, not the CrowdViTOutput
    dataclass returned by CrowdViT.forward."""

    def __init__(self, model: CrowdViT):
        super().__init__()
        self.model = model

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.model(clip).logits


def main(argv: list[str] | None = None) -> None:
    cfg, remaining = build_config_from_cli(argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args, _ = parser.parse_known_args(remaining)

    model = CrowdViT(cfg.model).eval()
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location="cpu")

    wrapped = _LogitsOnly(model)
    dummy = torch.randn(1, cfg.model.num_frames, 3, cfg.model.image_size, cfg.model.image_size)

    torch.onnx.export(
        wrapped,
        dummy,
        args.out,
        input_names=["clip"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,  # static-shape TorchScript-based export; avoids the onnxscript dependency
    )
    print(f"Exported ONNX model to {args.out}")


if __name__ == "__main__":
    main()
