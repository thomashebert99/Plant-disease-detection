"""Grad-CAM generation placeholders."""

from __future__ import annotations

from pathlib import Path


def generate_gradcam(image_path: Path) -> dict[str, str]:
    """Return a minimal Grad-CAM job manifest."""
    return {"image_path": str(image_path), "status": "not_implemented"}
