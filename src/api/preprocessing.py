"""Image preprocessing helpers used by the API."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

DEFAULT_IMAGE_SIZE = 224


def preprocess_image_bytes(
    image_bytes: bytes,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Convert uploaded image bytes to a raw RGB batch for Keras models."""

    if not image_bytes:
        raise ValueError("Le fichier image est vide.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB").resize((image_size, image_size))
            array = np.asarray(image, dtype="float32")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Le fichier fourni n'est pas une image lisible.") from exc

    return np.expand_dims(array, axis=0)
