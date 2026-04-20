"""Image quality descriptors used for privacy-preserving CV monitoring."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


def analyze_image_bytes(image_bytes: bytes) -> dict[str, float | int]:
    """Return simple visual descriptors without storing the uploaded image."""

    with Image.open(BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        pixels = np.asarray(rgb_image, dtype=np.float32)

    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue

    return {
        "image_width": int(width),
        "image_height": int(height),
        "image_aspect_ratio": _round(width / height if height else 0.0),
        "image_file_size_kb": _round(len(image_bytes) / 1024),
        "brightness_mean": _round(float(np.mean(luminance))),
        "contrast_std": _round(float(np.std(luminance))),
        "sharpness_score": _round(_sharpness_score(luminance)),
        "saturation_mean": _round(_saturation_mean(red, green, blue)),
        "green_ratio": _round(_green_ratio(red, green, blue)),
        "brown_ratio": _round(_brown_ratio(red, green, blue)),
    }


def _sharpness_score(luminance: np.ndarray) -> float:
    """Approximate sharpness using luminance gradients; lower means blurrier."""

    if luminance.shape[0] < 2 or luminance.shape[1] < 2:
        return 0.0
    dy = np.diff(luminance, axis=0)
    dx = np.diff(luminance, axis=1)
    return float((np.var(dx) + np.var(dy)) / 2)


def _saturation_mean(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> float:
    """Return mean HSV-like saturation in [0, 1]."""

    max_channel = np.maximum.reduce([red, green, blue])
    min_channel = np.minimum.reduce([red, green, blue])
    saturation = np.divide(
        max_channel - min_channel,
        max_channel,
        out=np.zeros_like(max_channel),
        where=max_channel > 0,
    )
    return float(np.mean(saturation))


def _green_ratio(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> float:
    """Return the share of pixels dominated by green leaf tones."""

    green_pixels = (green > 60) & (green > red * 1.05) & (green > blue * 1.05)
    return float(np.mean(green_pixels))


def _brown_ratio(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> float:
    """Return a coarse share of brown/yellow lesion-like pixels."""

    brown_pixels = (
        (red > 70)
        & (green > 35)
        & (green < 180)
        & (blue < 140)
        & (red >= green * 0.85)
        & (green >= blue * 0.85)
    )
    return float(np.mean(brown_pixels))


def _round(value: Any) -> float:
    """Round numeric descriptors for stable JSONL logs."""

    return round(float(value), 4)
