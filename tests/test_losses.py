import torch

from crowdvit.config import LossConfig
from crowdvit.losses.classification_loss import ClassificationLoss
from crowdvit.losses.combined import CrowdViTLoss
from crowdvit.losses.temporal_smoothness import TemporalSmoothnessLoss
from crowdvit.losses.view_consistency import ViewConsistencyLoss
from crowdvit.models.crowdvit import CrowdViT


def test_classification_loss_matches_cross_entropy():
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))
    loss = ClassificationLoss()(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(loss, expected)


def test_temporal_smoothness_zero_for_constant_sequence():
    constant = torch.ones(2, 5, 8)
    loss = TemporalSmoothnessLoss()(constant)
    assert torch.allclose(loss, torch.zeros(()))


def test_temporal_smoothness_positive_for_varying_sequence():
    varying = torch.randn(2, 5, 8)
    loss = TemporalSmoothnessLoss()(varying)
    assert loss.item() > 0


def test_view_consistency_decreases_as_views_align():
    loss_fn = ViewConsistencyLoss(collapse_weight=0.0)  # isolate the alignment term
    aligned = [torch.ones(8, 16) + 0.01 * torch.randn(8, 16) for _ in range(3)]
    misaligned = [torch.randn(8, 16) for _ in range(3)]
    aligned_loss = loss_fn(aligned)
    misaligned_loss = loss_fn(misaligned)
    assert aligned_loss.item() < misaligned_loss.item()


def test_view_consistency_penalizes_collapse():
    loss_fn = ViewConsistencyLoss(collapse_weight=1.0, target_std=1.0)
    collapsed = [torch.ones(8, 16) for _ in range(3)]  # zero variance across batch
    loss = loss_fn(collapsed)
    assert loss.item() > 0  # collapse penalty should fire even though alignment is perfect


def test_combined_loss_respects_ablation_flags(tiny_model_cfg, tiny_clip):
    model = CrowdViT(tiny_model_cfg)
    targets = torch.randint(0, tiny_model_cfg.num_classes, (tiny_clip.shape[0],))
    out = model(tiny_clip)

    cfg_off = LossConfig(use_temporal_loss=False, use_view_loss=False)
    loss_out = CrowdViTLoss(cfg_off)(out, targets)
    assert torch.equal(loss_out.temp, torch.zeros(()))
    assert torch.equal(loss_out.view, torch.zeros(()))
    assert torch.allclose(loss_out.total, loss_out.ce)

    cfg_on = LossConfig(
        use_temporal_loss=True, use_view_loss=True, lambda_temp=0.1, lambda_view=0.1
    )
    loss_out_on = CrowdViTLoss(cfg_on)(out, targets)
    assert loss_out_on.total.item() != loss_out_on.ce.item()


def test_combined_loss_backward(tiny_model_cfg, tiny_clip):
    model = CrowdViT(tiny_model_cfg)
    targets = torch.randint(0, tiny_model_cfg.num_classes, (tiny_clip.shape[0],))
    out = model(tiny_clip)
    loss_out = CrowdViTLoss(LossConfig())(out, targets)
    loss_out.total.backward()
    assert any(p.grad is not None for p in model.parameters())
