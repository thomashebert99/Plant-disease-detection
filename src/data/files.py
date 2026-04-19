"""Small filesystem helpers shared by dataset preparation scripts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def normalize_copy_mode(copy_mode: str) -> str:
    """Validate the file materialization mode used by data scripts."""

    normalized = copy_mode.lower().strip()
    if normalized not in {"copy", "hardlink"}:
        raise ValueError("copy_mode must be either 'copy' or 'hardlink'")
    return normalized


def materialize_file(source: Path, destination: Path, copy_mode: str) -> None:
    """Create a file by copy or hardlink, falling back to copy if linking fails."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    if copy_mode == "copy":
        shutil.copy2(source, destination)
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_prefixed_filename(prefix: str, filename: str) -> str:
    """Return a filename prefixed with a sanitized source label."""

    sanitized_prefix = prefix.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{sanitized_prefix}__{filename}"
