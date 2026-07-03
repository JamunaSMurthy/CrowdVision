"""Clip index sampling and frame decoding.

Uses ``decord`` for fast random-access video decoding when installed, and
falls back to ``torchvision.io.read_video`` otherwise so the package works
without the optional ``video`` extra (at the cost of decoding speed).
"""

from __future__ import annotations

import random

import numpy as np


def sample_clip_indices(
    total_frames: int, num_frames: int, stride: int, mode: str, jitter_max_shift: int = 0
) -> np.ndarray:
    """mode: 'random' (train) or 'uniform' (eval, center/deterministic).

    In 'random' mode, in addition to a randomly chosen clip start, each
    sampled index is independently perturbed by up to ``jitter_max_shift``
    frames (clamped to a valid index) — this is the temporal-jittering
    augmentation described in the paper's implementation details.
    """
    span = (num_frames - 1) * stride + 1

    if total_frames <= span:
        # Clip is shorter than the requested span: sample with the largest
        # stride that fits, repeating the last valid index if still short.
        indices = np.arange(0, total_frames, max(1, stride))[:num_frames]
        if len(indices) < num_frames:
            pad = np.full(num_frames - len(indices), indices[-1] if len(indices) else 0)
            indices = np.concatenate([indices, pad])
        return indices.astype(np.int64)

    max_start = total_frames - span
    if mode == "random":
        start = random.randint(0, max_start)
    elif mode == "uniform":
        start = max_start // 2
    else:
        raise ValueError(f"Unknown clip sampling mode '{mode}'")

    indices = (start + np.arange(num_frames) * stride).astype(np.int64)

    if mode == "random" and jitter_max_shift > 0:
        shifts = np.random.randint(-jitter_max_shift, jitter_max_shift + 1, size=num_frames)
        indices = np.clip(indices + shifts, 0, total_frames - 1)

    return indices


def _decode_with_decord(path: str, indices: np.ndarray) -> np.ndarray:
    import decord

    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(path, num_threads=1)
    frames = reader.get_batch(indices.tolist()).asnumpy()  # (T, H, W, C) uint8, RGB
    return frames


def _decode_with_torchvision(path: str, indices: np.ndarray) -> np.ndarray:
    from torchvision.io import read_video

    video, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
    video = video.numpy()
    max_idx = video.shape[0] - 1
    clipped = np.clip(indices, 0, max_idx)
    return video[clipped]


def decode_video_clip(path: str, indices: np.ndarray) -> np.ndarray:
    try:
        return _decode_with_decord(path, indices)
    except ImportError:
        return _decode_with_torchvision(path, indices)


def get_total_frames(path: str) -> int:
    try:
        import decord

        return len(decord.VideoReader(path, num_threads=1))
    except ImportError:
        from torchvision.io import read_video

        video, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
        return video.shape[0]
