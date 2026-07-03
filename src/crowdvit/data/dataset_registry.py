"""Maps a dataset name in DataConfig to its manifest files. Every supported
dataset uses the same underlying ManifestVideoDataset; this module just
validates the name and picks the right manifest path for the requested split.
"""

from __future__ import annotations

from crowdvit.config import Config
from crowdvit.data.manifest_dataset import ManifestVideoDataset

SUPPORTED_DATASETS = (
    "kinetics400",
    "kinetics600",
    "kinetics700",
    "ucf101",
    "shanghaitech",
    "xdviolence",
    "publicpark",
)

_MANIFEST_ATTR = {
    "train": "train_manifest",
    "val": "val_manifest",
    "test": "test_manifest",
}


def build_dataset(cfg: Config, split: str) -> ManifestVideoDataset:
    if cfg.data.name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{cfg.data.name}'. Supported: {SUPPORTED_DATASETS}. "
            "Add a new entry to SUPPORTED_DATASETS and a configs/datasets/<name>.yaml "
            "preset to register another dataset."
        )
    manifest_path = getattr(cfg.data, _MANIFEST_ATTR[split])
    return ManifestVideoDataset(manifest_path, cfg.model, cfg.data, split)
