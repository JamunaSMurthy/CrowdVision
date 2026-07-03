"""Full-dataset evaluation loop: runs the model over every batch in
``loader``, accumulates Top-1/Top-5 accuracy on the fly, and computes mean
Average Precision from the pooled softmax probabilities at the end.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from crowdvit.engine.metrics import accuracy_topk, mean_average_precision
from crowdvit.models.crowdvit import CrowdViT
from crowdvit.utils.distributed import is_main_process


@torch.no_grad()
def evaluate(
    model: CrowdViT,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    amp: bool = True,
) -> dict[str, float]:
    """Returns {"top1": ..., "top5": ..., "mAP": ...} (percentages) over the
    whole dataset served by ``loader``."""
    model.eval()
    all_probs, all_targets = [], []
    top1_sum, top5_sum, total = 0.0, 0.0, 0

    for clips, targets in tqdm(
        loader, desc="evaluating", disable=not is_main_process(), leave=False
    ):
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(clips)

        acc = accuracy_topk(output.logits, targets, topk=(1, 5))
        batch_size = targets.shape[0]
        top1_sum += acc[1] * batch_size
        top5_sum += acc[5] * batch_size
        total += batch_size

        probs = F.softmax(output.logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mAP = mean_average_precision(all_probs, all_targets, num_classes)

    return {
        "top1": top1_sum / total,
        "top5": top5_sum / total,
        "mAP": mAP,
    }
