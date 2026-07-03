import pytest
import torch

from crowdvit.config import LossConfig, ModelConfig


@pytest.fixture
def tiny_model_cfg() -> ModelConfig:
    """A small-but-shape-correct config so unit tests run in milliseconds."""
    return ModelConfig(
        num_frames=8,
        image_size=32,
        patch_size=16,
        stem_channels=16,
        stem_descriptor_dim=32,
        hidden_dim=32,
        num_heads=4,
        kernel_dim=16,
        view_window_sizes=[4, 2],
        view_encoder_depth=1,
        fusion_encoder_depth=1,
        num_classes=7,
    )


@pytest.fixture
def tiny_loss_cfg() -> LossConfig:
    return LossConfig()


@pytest.fixture
def tiny_clip(tiny_model_cfg: ModelConfig) -> torch.Tensor:
    return torch.randn(
        2, tiny_model_cfg.num_frames, 3, tiny_model_cfg.image_size, tiny_model_cfg.image_size
    )
