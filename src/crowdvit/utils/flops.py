"""GFLOPs and parameter-count measurement for the accuracy-efficiency
trade-off table. Uses torch's built-in dispatch-based FlopCounterMode, which
counts every aten op (including the einsums inside Performer/KGA attention)
rather than only nn.Module-wrapped layers, falling back to a coarse
Linear/Conv hook-based counter on older torch versions where it's
unavailable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_gflops(model: nn.Module, example_input: torch.Tensor) -> float:
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(example_input)
        total_flops = counter.get_total_flops()
        return total_flops / 1e9
    except ImportError:
        return _count_gflops_fallback(model, example_input)


def _count_gflops_fallback(model: nn.Module, example_input: torch.Tensor) -> float:
    total_macs = 0

    def linear_hook(module, inputs, output):
        nonlocal total_macs
        total_macs += output.numel() * module.in_features

    def conv3d_hook(module, inputs, output):
        nonlocal total_macs
        kernel_volume = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
        in_channels_per_group = module.in_channels // module.groups
        total_macs += output.numel() * in_channels_per_group * kernel_volume

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.Conv3d):
            handles.append(module.register_forward_hook(conv3d_hook))

    model.eval()
    with torch.no_grad():
        model(example_input)
    for handle in handles:
        handle.remove()

    return (2 * total_macs) / 1e9
