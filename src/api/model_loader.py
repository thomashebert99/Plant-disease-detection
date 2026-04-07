"""Model loading and caching helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def load_model(model_dir: str) -> dict[str, str]:
    """Return placeholder metadata for a loaded model."""
    return {"model_dir": str(Path(model_dir))}
