"""Prediction logging utilities for lightweight API observability."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

TRACKING_FILE = Path("logs/predictions.jsonl")


def tracking_file() -> Path:
    """Return the JSONL file used to store prediction events."""

    configured_path = os.getenv("MONITORING_LOG_PATH")
    if configured_path:
        return Path(configured_path)
    return TRACKING_FILE


def log_prediction(payload: dict[str, Any]) -> None:
    """Append a prediction event to a JSONL file.

    The payload must not contain the uploaded image. The goal is to monitor the
    service behavior, not to store user data.
    """

    path = tracking_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_prediction_events(limit: int | None = None) -> list[dict[str, Any]]:
    """Read prediction events from the JSONL file, ignoring malformed lines."""

    path = tracking_file()
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)

    if limit is not None and limit >= 0:
        return events[-limit:]
    return events


def summarize_predictions() -> dict[str, Any]:
    """Return aggregate monitoring metrics for API prediction events."""

    events = read_prediction_events()
    prediction_events = [
        event for event in events if str(event.get("event_type", "")) == "prediction"
    ]

    statuses = [str(event.get("status", "")) for event in prediction_events]
    latencies = _numeric_values(prediction_events, "latency_ms")
    species_confidences = _numeric_values(prediction_events, "species_confidence")
    disease_confidences = _numeric_values(prediction_events, "disease_confidence")

    return {
        "enabled": True,
        "storage": "jsonl",
        "total_events": len(events),
        "total_predictions": len(prediction_events),
        "ok": statuses.count("ok"),
        "uncertain_species": statuses.count("uncertain_species"),
        "errors": statuses.count("error"),
        "average_latency_ms": _rounded_mean(latencies),
        "average_species_confidence": _rounded_mean(species_confidences),
        "average_disease_confidence": _rounded_mean(disease_confidences),
        "last_event_at": prediction_events[-1].get("timestamp") if prediction_events else None,
    }


def _numeric_values(events: list[dict[str, Any]], key: str) -> list[float]:
    """Extract valid numeric values from monitoring events."""

    values: list[float] = []
    for event in events:
        value = event.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _rounded_mean(values: list[float]) -> float | None:
    """Return a stable rounded mean for API responses."""

    if not values:
        return None
    return round(mean(values), 4)
