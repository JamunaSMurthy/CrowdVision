"""Video data pipeline: clip sampling, decoding, clip-consistent
augmentation transforms, and the manifest-backed dataset/registry.
"""

from crowdvit.data.dataset_registry import build_dataset

__all__ = ["build_dataset"]
