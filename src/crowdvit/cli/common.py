"""Shared CLI plumbing: load a YAML config, apply ``--section.field value``
overrides on top of it, and build train/val/test dataloaders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from crowdvit.config import Config, set_by_path
from crowdvit.data.dataset_registry import build_dataset


def _parse_scalar(value_str: str):
    value = yaml.safe_load(value_str)
    if isinstance(value, str):
        # PyYAML's YAML-1.1 float resolver requires a decimal point (e.g.
        # "1.0e-4"), so exponent-only literals like "1e-4" come back as
        # plain strings. Coerce those back to numbers explicitly.
        for cast in (int, float):
            try:
                return cast(value_str)
            except ValueError:
                continue
    return value


def apply_overrides(cfg: Config, overrides: list[str]) -> None:
    it = iter(overrides)
    for token in it:
        if not token.startswith("--"):
            raise ValueError(
                f"Unexpected override token '{token}', expected '--section.field value'"
            )
        key = token[2:]
        try:
            value_str = next(it)
        except StopIteration as e:
            raise ValueError(f"Missing value for override '--{key}'") from e
        value = _parse_scalar(value_str)
        set_by_path(cfg, key, value)


def _split_overrides(argv: list[str]) -> tuple[list[str], list[str]]:
    """Config overrides use dotted keys ('--section.field value'); anything
    else (e.g. a script's own '--device', '--checkpoint') is passed through
    untouched for the calling script's own argparse to consume.
    """
    override_tokens: list[str] = []
    other_tokens: list[str] = []
    it = iter(argv)
    for token in it:
        if token.startswith("--") and "." in token[2:]:
            override_tokens.append(token)
            try:
                override_tokens.append(next(it))
            except StopIteration as e:
                raise ValueError(f"Missing value for override '{token}'") from e
        else:
            other_tokens.append(token)
    return override_tokens, other_tokens


def resolve_num_classes(cfg: Config) -> None:
    """Sync cfg.model.num_classes to the actual class map on disk, if one
    exists. Datasets with a fixed, universally standard taxonomy (Kinetics,
    UCF101) already have the right value baked into their config presets,
    so this is a no-op for them until a class map happens to exist at that
    path too. For datasets with no single standard class list (e.g.
    ShanghaiTech, XD-Violence, or a custom surveillance dataset), this is
    what actually determines num_classes: build the manifest and class map
    first with scripts/make_manifest.py, and the model head will always
    match it exactly rather than relying on a hand-set number that could
    silently drift out of sync with the data.
    """
    class_map_path = Path(cfg.data.class_map)
    if not class_map_path.is_file():
        return
    with open(class_map_path) as f:
        class_map = json.load(f)
    cfg.model.num_classes = len(class_map)


def build_config_from_cli(argv: list[str] | None = None) -> tuple[Config, list[str]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    known, remaining = parser.parse_known_args(argv)
    cfg = Config.from_yaml(known.config)

    override_tokens, other_tokens = _split_overrides(remaining)
    apply_overrides(cfg, override_tokens)
    resolve_num_classes(cfg)
    return cfg, other_tokens


def build_dataloaders(cfg: Config, distributed: bool = False) -> tuple[DataLoader, DataLoader]:
    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")

    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.optim.batch_size,
            sampler=DistributedSampler(train_ds, shuffle=True),
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.optim.batch_size,
            sampler=DistributedSampler(val_ds, shuffle=False),
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.optim.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.optim.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )
    return train_loader, val_loader


def build_test_dataloader(cfg: Config) -> DataLoader:
    test_ds = build_dataset(cfg, "test")
    return DataLoader(
        test_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
