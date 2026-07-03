"""Clip-consistent video transforms: the same random crop / flip / color
jitter parameters are drawn once per clip and applied identically to every
frame, so augmentation does not itself introduce frame-to-frame jitter.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF

from crowdvit.config import AugmentationConfig


class ToTensorVideo:
    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """frames: (T, H, W, C) uint8 -> (T, C, H, W) float in [0, 1]."""
        return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0


class RandomResizedCropVideo:
    def __init__(self, size: int, scale: tuple[float, float]):
        self.size = size
        self.scale = scale

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        _, _, h, w = clip.shape
        area = h * w
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(3 / 4, 4 / 3)
            new_w = int(round((target_area * aspect_ratio) ** 0.5))
            new_h = int(round((target_area / aspect_ratio) ** 0.5))
            if 0 < new_w <= w and 0 < new_h <= h:
                x0 = random.randint(0, w - new_w)
                y0 = random.randint(0, h - new_h)
                cropped = clip[:, :, y0 : y0 + new_h, x0 : x0 + new_w]
                return TF.resize(cropped, [self.size, self.size], antialias=True)
        side = min(h, w)
        cropped = TF.center_crop(clip, [side, side])
        return TF.resize(cropped, [self.size, self.size], antialias=True)


class CenterCropResizeVideo:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        _, _, h, w = clip.shape
        side = min(h, w)
        cropped = TF.center_crop(clip, [side, side])
        return TF.resize(cropped, [self.size, self.size], antialias=True)


class RandomHorizontalFlipVideo:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            return torch.flip(clip, dims=[-1])
        return clip


class ColorJitterVideo:
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float):
        self.brightness, self.contrast, self.saturation, self.hue = (
            brightness,
            contrast,
            saturation,
            hue,
        )

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        b = random.uniform(max(0.0, 1 - self.brightness), 1 + self.brightness)
        c = random.uniform(max(0.0, 1 - self.contrast), 1 + self.contrast)
        s = random.uniform(max(0.0, 1 - self.saturation), 1 + self.saturation)
        h = random.uniform(-self.hue, self.hue)
        out = TF.adjust_brightness(clip, b)
        out = TF.adjust_contrast(out, c)
        out = TF.adjust_saturation(out, s)
        out = TF.adjust_hue(out, h)
        return out


class NormalizeVideo:
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self.mean) / self.std


class ComposeVideo:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


def build_train_transforms(cfg: AugmentationConfig, image_size: int) -> ComposeVideo:
    return ComposeVideo(
        [
            ToTensorVideo(),
            RandomResizedCropVideo(image_size, cfg.random_resized_crop_scale),
            RandomHorizontalFlipVideo(cfg.horizontal_flip_prob),
            ColorJitterVideo(
                cfg.color_jitter_brightness,
                cfg.color_jitter_contrast,
                cfg.color_jitter_saturation,
                cfg.color_jitter_hue,
            ),
            NormalizeVideo(cfg.mean, cfg.std),
        ]
    )


def build_eval_transforms(cfg: AugmentationConfig, image_size: int) -> ComposeVideo:
    return ComposeVideo(
        [
            ToTensorVideo(),
            CenterCropResizeVideo(image_size),
            NormalizeVideo(cfg.mean, cfg.std),
        ]
    )
