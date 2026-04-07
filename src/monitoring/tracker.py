"""Prediction logging utilities for observability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRACKING_FILE = Path("logs/predictions.jsonl")


def log_prediction(payload: dict[str, Any]) -> None:
    """Append a prediction event to a JSONL file."""
    TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TRACKING_FILE.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")
