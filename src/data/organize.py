"""Utilities to reorganize PlantVillage data into task-specific folders."""

from __future__ import annotations

from pathlib import Path


def organize_processed_dataset(source_dir: Path, target_dir: Path) -> dict[str, str]:
    """Return a simple manifest placeholder for a future reorganization pipeline."""
    return {"source": str(source_dir), "target": str(target_dir)}
