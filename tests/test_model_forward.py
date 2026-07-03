import dataclasses

import pytest
import torch

from crowdvit.models.crowdvit import CrowdViT


def test_forward_output_shapes(tiny_model_cfg, tiny_clip):
    model = CrowdViT(tiny_model_cfg)
    out = model(tiny_clip)
    batch = tiny_clip.shape[0]
    assert out.logits.shape == (batch, tiny_model_cfg.num_classes)
    assert len(out.view_descriptors) == tiny_model_cfg.num_views
    assert all(d.shape == (batch, tiny_model_cfg.hidden_dim) for d in out.view_descriptors)
    assert out.pooled_feature.shape == (batch, tiny_model_cfg.hidden_dim)
    assert out.frame_sequence.shape[0] == batch
    assert out.frame_sequence.shape[2] == tiny_model_cfg.hidden_dim


@pytest.mark.parametrize("view_mode", ["adaptive", "fixed", "pyramid"])
def test_forward_all_view_modes(tiny_model_cfg, tiny_clip, view_mode):
    cfg = dataclasses.replace(tiny_model_cfg, view_mode=view_mode)
    model = CrowdViT(cfg)
    out = model(tiny_clip)
    assert torch.isfinite(out.logits).all()


@pytest.mark.parametrize(
    "fusion_type", ["kga", "concat_only", "performer_only", "graph_only", "full_cross_attention"]
)
def test_forward_all_fusion_types(tiny_model_cfg, tiny_clip, fusion_type):
    cfg = dataclasses.replace(tiny_model_cfg, fusion_type=fusion_type)
    model = CrowdViT(cfg)
    out = model(tiny_clip)
    assert torch.isfinite(out.logits).all()


@pytest.mark.parametrize("attention_type", ["performer", "linformer", "full"])
def test_forward_all_attention_types(tiny_model_cfg, tiny_clip, attention_type):
    cfg = dataclasses.replace(tiny_model_cfg, attention_type=attention_type)
    model = CrowdViT(cfg)
    out = model(tiny_clip)
    assert torch.isfinite(out.logits).all()


def test_forward_without_global_fusion_encoder(tiny_model_cfg, tiny_clip):
    cfg = dataclasses.replace(tiny_model_cfg, use_global_fusion_encoder=False)
    model = CrowdViT(cfg)
    out = model(tiny_clip)
    assert torch.isfinite(out.logits).all()


def test_single_view_model_runs(tiny_model_cfg, tiny_clip):
    cfg = dataclasses.replace(tiny_model_cfg, view_window_sizes=[4])
    model = CrowdViT(cfg)
    clip = torch.randn(2, cfg.num_frames, 3, cfg.image_size, cfg.image_size)
    out = model(clip)
    assert out.logits.shape == (2, cfg.num_classes)


def test_backward_pass_updates_parameters(tiny_model_cfg, tiny_clip):
    model = CrowdViT(tiny_model_cfg)
    out = model(tiny_clip)
    loss = out.logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_redraw_performer_projections_is_a_no_op_on_shapes(tiny_model_cfg, tiny_clip):
    model = CrowdViT(tiny_model_cfg)
    model.redraw_performer_projections()
    out = model(tiny_clip)
    assert out.logits.shape == (tiny_clip.shape[0], tiny_model_cfg.num_classes)
