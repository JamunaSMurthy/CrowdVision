"""Thin helpers around torch.distributed for multi-GPU training via
torchrun: process-group init from env vars, rank/world-size queries that
degrade gracefully to single-process values, and a DDP wrapping helper.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed(device: str) -> tuple[int, int, torch.device]:
    """Initializes torch.distributed from torchrun-provided env vars.
    Returns (rank, world_size, device).
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        backend = "nccl" if device == "cuda" and torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if device == "cuda":
            torch.cuda.set_device(local_rank)
            resolved_device = torch.device("cuda", local_rank)
        else:
            resolved_device = torch.device(device)
    else:
        resolved_device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )

    return rank, world_size, resolved_device


def is_main_process() -> bool:
    if not is_distributed_available():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if is_distributed_available() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed_available() else 1


def barrier() -> None:
    if is_distributed_available():
        dist.barrier()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not is_distributed_available():
        return value
    reduced = value.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= get_world_size()
    return reduced


def wrap_model_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if not is_distributed_available():
        return model
    device_ids = [device.index] if device.type == "cuda" else None
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
