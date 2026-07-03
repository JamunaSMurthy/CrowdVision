import csv

import numpy as np
import torch

from crowdvit.config import AugmentationConfig, DataConfig, ModelConfig
from crowdvit.data import manifest_dataset
from crowdvit.data.transforms import build_eval_transforms, build_train_transforms
from crowdvit.data.video_io import sample_clip_indices


def test_sample_clip_indices_uniform_is_deterministic():
    a = sample_clip_indices(300, 16, 4, mode="uniform")
    b = sample_clip_indices(300, 16, 4, mode="uniform")
    assert np.array_equal(a, b)
    assert len(a) == 16


def test_sample_clip_indices_short_video_pads():
    idx = sample_clip_indices(total_frames=5, num_frames=16, stride=4, mode="uniform")
    assert len(idx) == 16
    assert idx.max() < 5


def test_sample_clip_indices_jitter_stays_in_bounds():
    idx = sample_clip_indices(
        total_frames=50, num_frames=16, stride=2, mode="random", jitter_max_shift=3
    )
    assert idx.min() >= 0
    assert idx.max() < 50


def test_train_and_eval_transforms_produce_normalized_tensor():
    frames = (np.random.rand(8, 64, 96, 3) * 255).astype(np.uint8)
    cfg = AugmentationConfig()
    train_out = build_train_transforms(cfg, image_size=32)(frames)
    eval_out = build_eval_transforms(cfg, image_size=32)(frames)
    assert train_out.shape == (8, 3, 32, 32)
    assert eval_out.shape == (8, 3, 32, 32)
    assert train_out.dtype == torch.float32


def test_manifest_dataset_reads_csv_and_decodes(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["/fake/video_0.mp4", 0])
        writer.writerow(["/fake/video_1.mp4", 2])

    def fake_get_total_frames(path):
        return 40

    def fake_decode_video_clip(path, indices):
        return (np.random.rand(len(indices), 48, 48, 3) * 255).astype(np.uint8)

    monkeypatch.setattr(manifest_dataset, "get_total_frames", fake_get_total_frames)
    monkeypatch.setattr(manifest_dataset, "decode_video_clip", fake_decode_video_clip)

    model_cfg = ModelConfig(num_frames=8, image_size=32)
    data_cfg = DataConfig(frame_stride=2)

    ds = manifest_dataset.ManifestVideoDataset(manifest_path, model_cfg, data_cfg, split="train")
    assert len(ds) == 2

    clip, label = ds[0]
    assert clip.shape == (8, 3, 32, 32)
    assert label == 0

    clip1, label1 = ds[1]
    assert label1 == 2
