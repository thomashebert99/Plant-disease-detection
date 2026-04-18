"""Tests for API image preprocessing."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.api.preprocessing import preprocess_image_bytes


def test_preprocess_image_bytes_returns_raw_rgb_batch() -> None:
    """Uploaded images are converted to raw 0-255 RGB tensors."""

    buffer = BytesIO()
    Image.new("RGB", (12, 8), color=(10, 20, 30)).save(buffer, format="PNG")

    batch = preprocess_image_bytes(buffer.getvalue(), image_size=32)

    assert batch.shape == (1, 32, 32, 3)
    assert batch.dtype == np.float32
    assert batch.min() >= 0
    assert batch.max() <= 255


def test_preprocess_image_bytes_rejects_invalid_image() -> None:
    """Invalid uploads produce a clear ValueError for the route to map to 400."""

    with pytest.raises(ValueError, match="image lisible"):
        preprocess_image_bytes(b"not an image")
