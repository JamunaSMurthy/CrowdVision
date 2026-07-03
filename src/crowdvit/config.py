"""Configuration system for CrowdViT.

YAML configs support a single-level inheritance key ``_base_`` (a path
relative to the including file) that is merged before the including file's
own keys are applied on top. This gives ablation configs (configs/ablation_*.yaml)
a way to start from ``configs/base.yaml`` and only override the fields that
change, which keeps every ablation table in the paper reproducible from a
one-line diff against the base config.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    base_key = raw.pop("_base_", None)
    if base_key is not None:
        base_path = (path.parent / base_key).resolve()
        base_raw = _load_raw_yaml(base_path)
        raw = _deep_merge(base_raw, raw)
    return raw


@dataclass
class ModelConfig:
    num_frames: int = 16
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    stem_channels: int = 64
    stem_descriptor_dim: int = 256

    hidden_dim: int = 768
    num_heads: int = 8
    kernel_dim: int = 256

    view_window_sizes: list[int] = field(default_factory=lambda: [8, 4, 2])
    view_mode: str = "adaptive"  # adaptive | fixed | pyramid

    attention_type: str = "performer"  # performer | linformer | full
    linformer_proj_dim: int = 128

    fusion_type: str = (
        "kga"  # kga | concat_only | performer_only | graph_only | full_cross_attention
    )
    affinity_type: str = "learnable"  # learnable | uniform | random | fixed | none

    use_global_fusion_encoder: bool = True
    view_encoder_depth: int = 4
    fusion_encoder_depth: int = 2
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.1
    proj_dropout: float = 0.1
    head_dropout: float = 0.3

    num_classes: int = 400

    @property
    def num_views(self) -> int:
        return len(self.view_window_sizes)

    @property
    def head_dim(self) -> int:
        assert self.hidden_dim % self.num_heads == 0
        return self.hidden_dim // self.num_heads


@dataclass
class LossConfig:
    lambda_temp: float = 0.1
    lambda_view: float = 0.1
    use_temporal_loss: bool = True
    use_view_loss: bool = True
    view_collapse_weight: float = 1.0
    view_collapse_target_std: float = 1.0
    label_smoothing: float = 0.0


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_epochs: int = 5
    epochs: int = 50
    batch_size: int = 16
    grad_clip: float = 1.0
    amp: bool = True


@dataclass
class AugmentationConfig:
    random_resized_crop_scale: tuple[float, float] = (0.66, 1.0)
    horizontal_flip_prob: float = 0.5
    temporal_jitter_max_shift: int = 2
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1
    mean: tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: tuple[float, float, float] = (0.225, 0.225, 0.225)


@dataclass
class DataConfig:
    name: str = "kinetics400"
    train_manifest: str = "data/manifests/kinetics400_train.csv"
    val_manifest: str = "data/manifests/kinetics400_val.csv"
    test_manifest: str = "data/manifests/kinetics400_test.csv"
    class_map: str = "data/manifests/kinetics400_classes.json"
    num_workers: int = 8
    frame_stride: int = 4
    clip_sampling: str = "random"  # random (train) | uniform (eval)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class TrainConfig:
    seed: int = 42
    output_dir: str = "outputs/crowdvit_base"
    log_interval: int = 20
    checkpoint_interval: int = 1
    resume: str | None = None
    device: str = "cuda"
    distributed: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = _load_raw_yaml(Path(path))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        cfg = cls()
        for section_name in ("model", "loss", "optim", "data", "train"):
            section_raw = raw.get(section_name, {})
            section_obj = getattr(cfg, section_name)
            _assign_dataclass(section_obj, section_raw)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def set_by_path(cfg: "Config", dotted_path: str, value: Any) -> None:
    """Set a (possibly nested) config field from a dotted path, e.g.
    ``set_by_path(cfg, "optim.lr", 5e-5)`` or
    ``set_by_path(cfg, "model.view_window_sizes", [8, 4, 2])``.
    Used by the CLI scripts to apply ``--section.field value`` overrides.
    """
    parts = dotted_path.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    last = parts[-1]
    if not hasattr(obj, last):
        raise ValueError(f"Unknown config field '{dotted_path}'")
    current = getattr(obj, last)
    if isinstance(current, tuple) and isinstance(value, list):
        value = tuple(value)
    setattr(obj, last, value)


def _assign_dataclass(obj: Any, raw: dict[str, Any]) -> None:
    for key, value in raw.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config field '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _assign_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
