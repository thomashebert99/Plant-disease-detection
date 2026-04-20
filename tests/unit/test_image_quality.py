"""Tests for privacy-preserving image quality descriptors."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from src.monitoring.image_quality import analyze_image_bytes


def test_analyze_image_bytes_returns_visual_descriptors() -> None:
    """Image descriptors should be derived without persisting image content."""

    buffer = BytesIO()
    Image.new("RGB", (20, 10), color=(80, 160, 40)).save(buffer, format="PNG")

    metrics = analyze_image_bytes(buffer.getvalue())

    assert metrics["image_width"] == 20
    assert metrics["image_height"] == 10
    assert metrics["image_aspect_ratio"] == 2.0
    assert metrics["brightness_mean"] > 0
    assert 0 <= metrics["green_ratio"] <= 1
