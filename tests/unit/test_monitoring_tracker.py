"""Tests for lightweight prediction monitoring utilities."""

from __future__ import annotations

from src.monitoring.tracker import log_prediction, read_prediction_events, summarize_predictions


def test_log_prediction_writes_jsonl_event() -> None:
    """Prediction events should be persisted as JSONL without extra setup."""

    log_prediction(
        {
            "event_type": "prediction",
            "timestamp": "2026-04-18T10:00:00+00:00",
            "endpoint": "/predict",
            "status": "ok",
            "species_confidence": 0.9,
            "disease_confidence": 0.8,
            "latency_ms": 123.45,
        }
    )

    events = read_prediction_events()

    assert len(events) == 1
    assert events[0]["endpoint"] == "/predict"
    assert events[0]["status"] == "ok"


def test_summarize_predictions_aggregates_service_metrics() -> None:
    """Monitoring summary should expose simple certification-friendly metrics."""

    log_prediction(
        {
            "event_type": "prediction",
            "timestamp": "2026-04-18T10:00:00+00:00",
            "endpoint": "/predict",
            "status": "ok",
            "species_confidence": 0.9,
            "disease_confidence": 0.8,
            "latency_ms": 100.0,
        }
    )
    log_prediction(
        {
            "event_type": "prediction",
            "timestamp": "2026-04-18T10:01:00+00:00",
            "endpoint": "/predict",
            "status": "uncertain_species",
            "species_confidence": 0.4,
            "latency_ms": 200.0,
        }
    )
    log_prediction(
        {
            "event_type": "prediction",
            "timestamp": "2026-04-18T10:02:00+00:00",
            "endpoint": "/predict",
            "status": "error",
            "latency_ms": 300.0,
        }
    )

    summary = summarize_predictions()

    assert summary["total_predictions"] == 3
    assert summary["ok"] == 1
    assert summary["uncertain_species"] == 1
    assert summary["errors"] == 1
    assert summary["average_latency_ms"] == 200.0
    assert summary["average_species_confidence"] == 0.65
    assert summary["average_disease_confidence"] == 0.8
    assert summary["last_event_at"] == "2026-04-18T10:02:00+00:00"
