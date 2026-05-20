"""Checkpoint inspection helpers shared by training, evaluation, and inference."""

from __future__ import annotations

import os
from typing import Optional

import torch


def infer_num_classes_from_checkpoint(model_path: Optional[str]) -> Optional[int]:
    """Infer classifier output size from a PyTorch checkpoint when possible."""
    if not model_path or not os.path.exists(model_path):
        return None

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        return None

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        return None

    classifier_suffixes = (
        "classifier.2.weight",
        "classifier.weight",
        "fc.weight",
        "head.weight",
        "heads.head.weight",
    )
    for suffix in classifier_suffixes:
        for key, value in state_dict.items():
            normalized_key = key.removeprefix("module.")
            if normalized_key.endswith(suffix) and hasattr(value, "ndim") and value.ndim >= 2:
                return int(value.shape[0])
    return None
