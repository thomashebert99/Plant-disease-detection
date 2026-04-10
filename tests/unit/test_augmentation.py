"""Tests for Albumentations preprocessing pipelines."""

from __future__ import annotations

import numpy as np

from src.data.augmentation import (
    IMAGE_SIZE,
    build_train_transform,
    build_val_transform,
    train_transform,
    val_transform,
)


def test_augmentation_output_shape() -> None:
    """Training augmentation should always return the fixed model input size."""

    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    result = train_transform(image=img)["image"]

    assert result.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_val_transform_output_shape_and_dtype() -> None:
    """Validation preprocessing should resize and normalize deterministically."""

    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    result = val_transform(image=img)["image"]

    assert result.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert result.dtype == np.float32


def test_val_transform_contains_only_resize_and_normalize() -> None:
    """Validation/test preprocessing must not include stochastic augmentation."""

    transform_names = [type(transform).__name__ for transform in build_val_transform().transforms]

    assert transform_names == ["Resize", "Normalize"]


def test_train_transform_contains_expected_augmentations() -> None:
    """Training preprocessing should include the domain-gap augmentations from the guide."""

    transform_names = [type(transform).__name__ for transform in build_train_transform().transforms]

    assert transform_names == [
        "Resize",
        "HorizontalFlip",
        "VerticalFlip",
        "RandomBrightnessContrast",
        "HueSaturationValue",
        "ShiftScaleRotate",
        "GaussianBlur",
        "CoarseDropout",
        "RandomShadow",
        "Normalize",
    ]
