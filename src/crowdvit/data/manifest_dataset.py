"""Generic clip-level video dataset backed by a two-column CSV manifest
(``video_path,label_index``), used for every dataset in
Section "Datasets and Evaluation Protocol" of the paper (Kinetics-400/600/700,
UCF101, ShanghaiTech, XD-Violence, Public Park).
"""

from __future__ import annotations

import csv
from pathlib import Path

from torch.utils.data import Dataset

from crowdvit.config import DataConfig, ModelConfig
from crowdvit.data.transforms import build_eval_transforms, build_train_transforms
from crowdvit.data.video_io import decode_video_clip, get_total_frames, sample_clip_indices


def read_manifest(path: str | Path) -> list[tuple[str, int]]:
    samples = []
    with open(path, "r", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            video_path, label = row[0], row[1]
            samples.append((video_path, int(label)))
    return samples


class ManifestVideoDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        model_cfg: ModelConfig,
        data_cfg: DataConfig,
        split: str,
    ):
        assert split in ("train", "val", "test")
        self.samples = read_manifest(manifest_path)
        self.num_frames = model_cfg.num_frames
        self.stride = data_cfg.frame_stride
        self.split = split
        self.jitter_max_shift = data_cfg.augmentation.temporal_jitter_max_shift

        if split == "train":
            self.transform = build_train_transforms(data_cfg.augmentation, model_cfg.image_size)
            self.sampling_mode = "random"
        else:
            self.transform = build_eval_transforms(data_cfg.augmentation, model_cfg.image_size)
            self.sampling_mode = "uniform"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        total_frames = get_total_frames(video_path)
        jitter = self.jitter_max_shift if self.split == "train" else 0
        indices = sample_clip_indices(
            total_frames, self.num_frames, self.stride, self.sampling_mode, jitter
        )
        frames = decode_video_clip(video_path, indices)
        clip = self.transform(frames)
        return clip, label
