"""Validation helpers for train and validation dataset splits."""

from __future__ import annotations

from pathlib import Path


def validate_split_directories(train_dir: Path, val_dir: Path) -> bool:
    """Check that both split directories exist."""
    return train_dir.exists() and val_dir.exists()
