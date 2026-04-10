"""Albumentations pipelines for training and validation/test preprocessing."""

from __future__ import annotations

import albumentations as A

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform() -> A.Compose:
    """Return the strong training augmentation pipeline used for model training."""

    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.4,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                p=0.3,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                p=0.4,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=20,
                max_width=20,
                p=0.2,
            ),
            A.RandomShadow(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_val_transform() -> A.Compose:
    """Return the validation/test pipeline without data augmentation."""

    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


train_transform = build_train_transform()
val_transform = build_val_transform()
