"""Tests for lightweight prediction monitoring utilities."""

from __future__ import annotations

from src.monitoring.tracker import (
    feedback_file,
    log_feedback,
    log_prediction,
    read_feedback_events,
    read_prediction_events,
    summarize_feedback,
    summarize_predictions,
    tracking_file,
)


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


def test_monitoring_storage_dir_builds_default_jsonl_paths(monkeypatch, tmp_path) -> None:
    """A shared storage directory should define both monitoring JSONL files."""

    storage_dir = tmp_path / "monitoring"
    monkeypatch.delenv("MONITORING_LOG_PATH", raising=False)
    monkeypatch.delenv("FEEDBACK_LOG_PATH", raising=False)
    monkeypatch.setenv("MONITORING_STORAGE_DIR", str(storage_dir))

    assert tracking_file() == storage_dir / "predictions.jsonl"
    assert feedback_file() == storage_dir / "feedback.jsonl"


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
    assert summary["min_latency_ms"] == 100.0
    assert summary["max_latency_ms"] == 300.0
    assert summary["p95_latency_ms"] == 300.0
    assert summary["average_species_confidence"] == 0.65
    assert summary["average_disease_confidence"] == 0.8
    assert summary["error_rate"] == 0.3333
    assert summary["uncertain_rate"] == 0.3333
    assert summary["low_confidence_count"] == 1
    assert summary["last_event_at"] == "2026-04-18T10:02:00+00:00"


def test_summarize_predictions_detects_known_ood_like_shift(monkeypatch) -> None:
    """OOD-like images should be flagged as known domain shift, not as an unknown drift."""

    monkeypatch.setenv("MONITORING_DRIFT_MIN_EVENTS", "1")
    log_prediction(
        {
            "event_type": "prediction",
            "timestamp": "2026-04-18T10:00:00+00:00",
            "endpoint": "/predict",
            "status": "ok",
            "species": "tomato",
            "disease": "Late_Blight",
            "species_confidence": 0.72,
            "disease_confidence": 0.62,
            "latency_ms": 150.0,
            "brightness_mean": 108.0,
            "contrast_std": 58.0,
            "sharpness_score": 300.0,
            "saturation_mean": 0.34,
            "green_ratio": 0.34,
            "brown_ratio": 0.24,
        }
    )

    summary = summarize_predictions()

    assert summary["domain_shift"]["status"] == "ood_like"
    assert summary["domain_shift"]["risk_level"] == "watch"
    assert summary["domain_shift"]["closest_reference"] == "plantdoc_ood"


def test_feedback_summary_aggregates_user_disagreements() -> None:
    """Explicit user feedback should be stored separately from prediction monitoring."""

    log_feedback(
        {
            "event_type": "feedback",
            "timestamp": "2026-04-18T10:03:00+00:00",
            "verdict": "incorrect",
            "predicted_species": "tomato",
            "predicted_disease": "Late_Blight",
            "corrected_species": "potato",
            "corrected_disease": "Early_Blight",
        }
    )

    events = read_feedback_events()
    summary = summarize_feedback()

    assert len(events) == 1
    assert summary["total_feedback"] == 1
    assert summary["disagreement_rate"] == 1.0
    assert summary["corrected_species_distribution"] == {"potato": 1}


def test_user_feedback_adds_model_quality_shift_alert(monkeypatch) -> None:
    """Repeated user disagreements should become a quality-drift monitoring signal."""

    monkeypatch.setenv("MONITORING_FEEDBACK_MIN_EVENTS", "2")
    monkeypatch.setenv("MONITORING_MAX_DISAGREEMENT_RATE", "0.3")
    for timestamp, verdict in (
        ("2026-04-18T10:03:00+00:00", "incorrect"),
        ("2026-04-18T10:04:00+00:00", "incorrect"),
    ):
        log_feedback(
            {
                "event_type": "feedback",
                "timestamp": timestamp,
                "verdict": verdict,
                "predicted_species": "tomato",
                "predicted_disease": "Late_Blight",
            }
        )

    summary = summarize_predictions()

    assert summary["model_quality_shift"]["status"] == "quality_drift_suspected"
    assert summary["model_quality_shift"]["risk_level"] == "warning"
    assert any(alert["metric"] == "model_quality_shift" for alert in summary["alerts"])
