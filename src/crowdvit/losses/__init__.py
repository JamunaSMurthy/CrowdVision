"""Training losses: classification cross-entropy, temporal smoothness,
cross-view consistency, and the combined weighted objective.
"""

from crowdvit.losses.combined import CrowdViTLoss, CrowdViTLossOutput

__all__ = ["CrowdViTLoss", "CrowdViTLossOutput"]
