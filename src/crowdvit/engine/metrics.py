"""Top-1/Top-5 accuracy and mean Average Precision (Section "Evaluation
Metrics" of the paper)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def accuracy_topk(
    logits: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 5)
) -> dict[int, float]:
    max_k = min(max(topk), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))

    results = {}
    for k in topk:
        k_eff = min(k, max_k)
        results[k] = correct[:, :k_eff].any(dim=1).float().mean().item() * 100.0
    return results


def mean_average_precision(probs: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    """probs: (N, C) predicted class probabilities, targets: (N,) integer labels."""
    one_hot = np.zeros((targets.shape[0], num_classes), dtype=np.int32)
    one_hot[np.arange(targets.shape[0]), targets] = 1

    ap_scores = []
    for c in range(num_classes):
        if one_hot[:, c].sum() == 0:
            continue
        ap_scores.append(average_precision_score(one_hot[:, c], probs[:, c]))

    return float(np.mean(ap_scores)) * 100.0 if ap_scores else 0.0
