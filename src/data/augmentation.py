"""Albumentations pipeline placeholders for training and validation."""

from __future__ import annotations


def build_train_transforms() -> list[str]:
    """Describe the planned training augmentations."""
    return ["resize", "horizontal_flip", "color_jitter"]


def build_eval_transforms() -> list[str]:
    """Describe the planned validation and test transforms."""
    return ["resize", "normalize"]
