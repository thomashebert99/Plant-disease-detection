"""Prediction and feedback logging utilities for lightweight observability."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

TRACKING_FILE = Path("logs/predictions.jsonl")
FEEDBACK_FILE = Path("logs/feedback.jsonl")
STORAGE_DIR = Path("logs")
REFERENCE_FILE = Path(__file__).with_name("monitoring_reference.json")
CONFIDENCE_BINS = (
    (0.0, 0.5, "0-50%"),
    (0.5, 0.65, "50-65%"),
    (0.65, 0.8, "65-80%"),
    (0.8, 1.01, "80-100%"),
)
DRIFT_METRIC_KEYS = (
    "brightness_mean",
    "contrast_std",
    "sharpness_score",
    "saturation_mean",
    "green_ratio",
    "brown_ratio",
    "species_confidence",
    "disease_confidence",
)


def tracking_file() -> Path:
    """Return the JSONL file used to store prediction events."""

    configured_path = os.getenv("MONITORING_LOG_PATH")
    if configured_path:
        return Path(configured_path)
    return monitoring_storage_dir() / TRACKING_FILE.name


def feedback_file() -> Path:
    """Return the JSONL file used to store explicit user feedback."""

    configured_path = os.getenv("FEEDBACK_LOG_PATH")
    if configured_path:
        return Path(configured_path)
    return monitoring_storage_dir() / FEEDBACK_FILE.name


def monitoring_storage_dir() -> Path:
    """Return the directory used for monitoring JSONL storage."""

    configured_dir = os.getenv("MONITORING_STORAGE_DIR")
    if configured_dir:
        return Path(configured_dir)
    return STORAGE_DIR


def log_prediction(payload: dict[str, Any]) -> None:
    """Append a prediction event to a JSONL file.

    The payload must not contain the uploaded image. The goal is to monitor the
    service behavior, not to store user data.
    """

    path = tracking_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_feedback(payload: dict[str, Any]) -> None:
    """Append a user feedback event to a separate JSONL file."""

    path = feedback_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_prediction_events(limit: int | None = None) -> list[dict[str, Any]]:
    """Read prediction events from the JSONL file, ignoring malformed lines."""

    return _read_jsonl_events(tracking_file(), limit=limit)


def read_feedback_events(limit: int | None = None) -> list[dict[str, Any]]:
    """Read user feedback events from the JSONL file, ignoring malformed lines."""

    return _read_jsonl_events(feedback_file(), limit=limit)


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
    low_confidence_threshold = _env_float(
        "MONITORING_LOW_CONFIDENCE_THRESHOLD",
        _env_float("CONFIDENCE_THRESHOLD", 0.65),
    )
    low_confidence_count = _low_confidence_count(
        prediction_events,
        threshold=low_confidence_threshold,
    )
    feedback_summary = summarize_feedback()
    domain_shift_summary = summarize_domain_shift(prediction_events)

    summary: dict[str, Any] = {
        "enabled": True,
        "storage": "jsonl",
        "total_events": len(events),
        "total_predictions": len(prediction_events),
        "ok": statuses.count("ok"),
        "uncertain_species": statuses.count("uncertain_species"),
        "errors": statuses.count("error"),
        "ok_rate": _rate(statuses.count("ok"), len(prediction_events)),
        "uncertain_rate": _rate(statuses.count("uncertain_species"), len(prediction_events)),
        "error_rate": _rate(statuses.count("error"), len(prediction_events)),
        "low_confidence_count": low_confidence_count,
        "low_confidence_rate": _rate(low_confidence_count, len(prediction_events)),
        "low_confidence_threshold": low_confidence_threshold,
        "average_latency_ms": _rounded_mean(latencies),
        "min_latency_ms": _rounded_min(latencies),
        "max_latency_ms": _rounded_max(latencies),
        "p95_latency_ms": _rounded_percentile(latencies, 95),
        "average_species_confidence": _rounded_mean(species_confidences),
        "average_disease_confidence": _rounded_mean(disease_confidences),
        "species_distribution": _counter_mapping(prediction_events, "species"),
        "disease_distribution": _counter_mapping(prediction_events, "disease"),
        "healthy_ratio": _healthy_ratio(prediction_events),
        "species_confidence_histogram": _confidence_histogram(species_confidences),
        "disease_confidence_histogram": _confidence_histogram(disease_confidences),
        "feedback": feedback_summary,
        "domain_shift": domain_shift_summary,
        "model_quality_shift": summarize_model_quality_shift(
            feedback_summary,
            domain_shift_summary,
        ),
        "last_event_at": prediction_events[-1].get("timestamp") if prediction_events else None,
    }
    summary["alerts"] = build_alerts(summary)
    return summary


def summarize_feedback() -> dict[str, Any]:
    """Return aggregate metrics for explicit user feedback."""

    events = [
        event for event in read_feedback_events() if event.get("event_type") == "feedback"
    ]
    verdict_counts = _counter_mapping(events, "verdict")
    incorrect_count = verdict_counts.get("incorrect", 0)
    total = len(events)

    return {
        "total_feedback": total,
        "verdict_distribution": verdict_counts,
        "disagreement_rate": _rate(incorrect_count, total),
        "corrected_species_distribution": _counter_mapping(events, "corrected_species"),
        "corrected_disease_distribution": _counter_mapping(events, "corrected_disease"),
        "disputed_species_distribution": _counter_mapping(events, "predicted_species"),
        "disputed_disease_distribution": _counter_mapping(events, "predicted_disease"),
        "last_feedback_at": events[-1].get("timestamp") if events else None,
    }


def summarize_model_quality_shift(
    feedback_summary: dict[str, Any],
    domain_shift: dict[str, Any],
) -> dict[str, Any]:
    """Use user disagreement as a supervised signal for possible quality drift."""

    total_feedback = int(feedback_summary.get("total_feedback") or 0)
    disagreement_rate = float(feedback_summary.get("disagreement_rate") or 0.0)
    min_feedback = int(_env_float("MONITORING_FEEDBACK_MIN_EVENTS", 3))
    threshold = _env_float("MONITORING_MAX_DISAGREEMENT_RATE", 0.3)
    domain_status = str(domain_shift.get("status", "unknown"))

    summary: dict[str, Any] = {
        "status": "insufficient_feedback",
        "risk_level": "none",
        "total_feedback": total_feedback,
        "disagreement_rate": disagreement_rate,
        "minimum_feedback": min_feedback,
        "disagreement_threshold": threshold,
        "signals": [],
    }
    if total_feedback < min_feedback:
        return summary

    if disagreement_rate <= threshold:
        return {
            **summary,
            "status": "feedback_stable",
            "risk_level": "none",
        }

    status = "quality_drift_suspected"
    risk_level = "warning"
    if domain_status in {"ood_like", "reference_shift", "unknown_shift"}:
        status = "feedback_confirms_domain_risk"
    if domain_status == "unknown_shift":
        risk_level = "critical"

    return {
        **summary,
        "status": status,
        "risk_level": risk_level,
        "signals": [
            {
                "metric": "disagreement_rate",
                "value": disagreement_rate,
                "threshold": threshold,
                "message": "Le taux de désaccord utilisateur dépasse le seuil.",
            }
        ],
    }


def summarize_domain_shift(prediction_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare recent production descriptors to in-domain and known-OOD references."""

    window_size = int(_env_float("MONITORING_DRIFT_WINDOW", 50))
    min_events = int(_env_float("MONITORING_DRIFT_MIN_EVENTS", 5))
    recent_events = prediction_events[-window_size:] if window_size > 0 else prediction_events
    current_metrics = _current_metric_means(recent_events)
    reference = _load_reference()

    base_summary: dict[str, Any] = {
        "reference_available": bool(reference),
        "window_size": len(recent_events),
        "minimum_window_size": min_events,
        "current_metrics": current_metrics,
        "status": "insufficient_data",
        "risk_level": "none",
        "closest_reference": None,
        "distances": {},
        "signals": [],
    }
    if len(recent_events) < min_events or not reference:
        return base_summary

    distances: dict[str, float] = {}
    signals_by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain_key, domain in reference.get("domains", {}).items():
        distance, signals = _domain_distance(current_metrics, domain)
        if distance is not None:
            distances[domain_key] = distance
            signals_by_domain[domain_key] = signals

    if not distances:
        return base_summary

    closest_reference = min(distances, key=distances.get)
    closest_distance = distances[closest_reference]
    in_domain_distance = distances.get("plantvillage_in_domain", math.inf)
    ood_distance = distances.get("plantdoc_ood", math.inf)
    in_domain_threshold = _env_float("MONITORING_IN_DOMAIN_DISTANCE", 1.8)
    ood_threshold = _env_float("MONITORING_OOD_DISTANCE", 1.8)
    unknown_threshold = _env_float("MONITORING_UNKNOWN_DISTANCE", 2.8)

    status = "unknown_shift"
    risk_level = "critical"
    if closest_reference == "plantvillage_in_domain" and in_domain_distance <= in_domain_threshold:
        status = "in_domain"
        risk_level = "none"
    elif closest_reference == "plantdoc_ood" and ood_distance <= ood_threshold:
        status = "ood_like"
        risk_level = "watch"
    elif closest_distance <= unknown_threshold:
        status = "reference_shift"
        risk_level = "warning"

    prediction_drift = _prediction_drift_summary(recent_events, reference, closest_reference)

    return {
        **base_summary,
        "status": status,
        "risk_level": risk_level,
        "closest_reference": closest_reference,
        "distances": distances,
        "prediction_drift": prediction_drift,
        "signals": signals_by_domain.get(closest_reference, [])[:6],
    }


def build_alerts(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build human-readable monitoring alerts from aggregate metrics."""

    alerts: list[dict[str, Any]] = []
    _add_rate_alert(
        alerts,
        metric="error_rate",
        value=summary.get("error_rate"),
        threshold=_env_float("MONITORING_MAX_ERROR_RATE", 0.05),
        message="Taux d'erreur API élevé.",
    )
    _add_rate_alert(
        alerts,
        metric="uncertain_rate",
        value=summary.get("uncertain_rate"),
        threshold=_env_float("MONITORING_MAX_UNCERTAIN_RATE", 0.25),
        message="Trop de prédictions nécessitent une confirmation d'espèce.",
    )
    _add_min_alert(
        alerts,
        metric="average_disease_confidence",
        value=summary.get("average_disease_confidence"),
        threshold=_env_float("MONITORING_MIN_DISEASE_CONFIDENCE", 0.65),
        message="La confiance maladie moyenne est faible.",
    )
    _add_max_alert(
        alerts,
        metric="p95_latency_ms",
        value=summary.get("p95_latency_ms"),
        threshold=_env_float("MONITORING_MAX_P95_LATENCY_MS", 5000.0),
        message="La latence P95 dépasse le seuil attendu.",
    )

    domain_shift = summary.get("domain_shift")
    if isinstance(domain_shift, dict) and domain_shift.get("risk_level") in {"warning", "critical"}:
        alerts.append(
            {
                "level": domain_shift["risk_level"],
                "metric": "domain_shift",
                "message": "Le flux récent ne ressemble plus assez aux références connues.",
                "value": domain_shift.get("status"),
                "threshold": "in_domain_or_known_ood",
            }
        )
    model_quality_shift = summary.get("model_quality_shift")
    if isinstance(model_quality_shift, dict) and model_quality_shift.get("risk_level") in {
        "warning",
        "critical",
    }:
        alerts.append(
            {
                "level": model_quality_shift["risk_level"],
                "metric": "model_quality_shift",
                "message": "Les retours utilisateur signalent une possible dérive de qualité.",
                "value": model_quality_shift.get("disagreement_rate"),
                "threshold": model_quality_shift.get("disagreement_threshold"),
            }
        )
    return alerts


def _read_jsonl_events(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read JSONL events from a file, ignoring malformed lines."""

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


def _rounded_min(values: list[float]) -> float | None:
    """Return a rounded minimum value."""

    if not values:
        return None
    return round(min(values), 4)


def _rounded_max(values: list[float]) -> float | None:
    """Return a rounded maximum value."""

    if not values:
        return None
    return round(max(values), 4)


def _rounded_percentile(values: list[float], percentile: int) -> float | None:
    """Return a simple nearest-rank percentile."""

    if not values:
        return None
    sorted_values = sorted(values)
    index = math.ceil((percentile / 100) * len(sorted_values)) - 1
    index = min(max(index, 0), len(sorted_values) - 1)
    return round(sorted_values[index], 4)


def _counter_mapping(events: list[dict[str, Any]], key: str, *, limit: int = 10) -> dict[str, int]:
    """Return a stable top-count mapping for a categorical event key."""

    values = [str(event[key]) for event in events if event.get(key)]
    return dict(Counter(values).most_common(limit))


def _rate(count: int, total: int) -> float:
    """Return a rounded ratio with a stable zero fallback."""

    if total <= 0:
        return 0.0
    return round(count / total, 4)


def _low_confidence_count(events: list[dict[str, Any]], *, threshold: float) -> int:
    """Count predictions where at least one available confidence is below threshold."""

    count = 0
    for event in events:
        confidences = [
            value
            for value in (event.get("species_confidence"), event.get("disease_confidence"))
            if isinstance(value, int | float)
        ]
        if confidences and min(float(value) for value in confidences) < threshold:
            count += 1
    return count


def _confidence_histogram(values: list[float]) -> dict[str, int]:
    """Bucket confidence scores for dashboard charts."""

    histogram = {label: 0 for _, _, label in CONFIDENCE_BINS}
    for value in values:
        for lower, upper, label in CONFIDENCE_BINS:
            if lower <= value < upper:
                histogram[label] += 1
                break
    return histogram


def _healthy_ratio(events: list[dict[str, Any]]) -> float | None:
    """Return share of disease predictions labelled as healthy."""

    diseases = [str(event.get("disease")) for event in events if event.get("disease")]
    if not diseases:
        return None
    return _rate(diseases.count("Healthy"), len(diseases))


def _current_metric_means(events: list[dict[str, Any]]) -> dict[str, float]:
    """Compute recent means for drift-aware numeric descriptors."""

    metrics: dict[str, float] = {}
    for key in DRIFT_METRIC_KEYS:
        value = _rounded_mean(_numeric_values(events, key))
        if value is not None:
            metrics[key] = value
    return metrics


def _load_reference() -> dict[str, Any]:
    """Load the drift reference file, returning an empty dict when unavailable."""

    path = Path(os.getenv("MONITORING_REFERENCE_PATH", str(REFERENCE_FILE)))
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _domain_distance(
    current_metrics: dict[str, float],
    domain: dict[str, Any],
) -> tuple[float | None, list[dict[str, Any]]]:
    """Return mean z-distance between current metrics and one reference domain."""

    metrics = domain.get("metrics", {})
    distances: list[float] = []
    signals: list[dict[str, Any]] = []
    for key, current_value in current_metrics.items():
        reference_metric = metrics.get(key)
        if not isinstance(reference_metric, dict):
            continue
        reference_mean = reference_metric.get("mean")
        reference_std = reference_metric.get("std")
        if not isinstance(reference_mean, int | float) or not isinstance(reference_std, int | float):
            continue
        std = max(float(reference_std), 1e-6)
        z_score = abs(float(current_value) - float(reference_mean)) / std
        distances.append(z_score)
        if z_score >= 1.5:
            signals.append(
                {
                    "metric": key,
                    "current": round(float(current_value), 4),
                    "reference": round(float(reference_mean), 4),
                    "z_score": round(z_score, 4),
                    "level": "critical" if z_score >= 3.0 else "warning",
                    "direction": "above" if current_value > reference_mean else "below",
                }
            )
    if not distances:
        return None, []
    signals.sort(key=lambda signal: signal["z_score"], reverse=True)
    return round(mean(distances), 4), signals


def _prediction_drift_summary(
    events: list[dict[str, Any]],
    reference: dict[str, Any],
    domain_key: str,
) -> dict[str, Any]:
    """Compare recent predicted-class distributions to the closest reference."""

    domain = reference.get("domains", {}).get(domain_key, {})
    reference_distribution = domain.get("prediction_distribution", {})
    return {
        "species_distance": _distribution_distance(
            _distribution(events, "species"),
            reference_distribution.get("species", {}),
        ),
        "disease_distance": _distribution_distance(
            _distribution(events, "disease"),
            reference_distribution.get("disease", {}),
        ),
    }


def _distribution(events: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Return categorical distribution for one event key."""

    counts = Counter(str(event[key]) for event in events if event.get(key))
    total = sum(counts.values())
    if total == 0:
        return {}
    return {label: count / total for label, count in counts.items()}


def _distribution_distance(
    current: dict[str, float],
    reference: dict[str, Any],
) -> float | None:
    """Return total variation distance between two categorical distributions."""

    if not current or not reference:
        return None
    labels = set(current) | set(reference)
    distance = 0.5 * sum(
        abs(float(current.get(label, 0.0)) - float(reference.get(label, 0.0)))
        for label in labels
    )
    return round(distance, 4)


def _add_rate_alert(
    alerts: list[dict[str, Any]],
    *,
    metric: str,
    value: Any,
    threshold: float,
    message: str,
) -> None:
    """Append an alert when a rate exceeds its threshold."""

    _add_max_alert(alerts, metric=metric, value=value, threshold=threshold, message=message)


def _add_max_alert(
    alerts: list[dict[str, Any]],
    *,
    metric: str,
    value: Any,
    threshold: float,
    message: str,
) -> None:
    """Append an alert when a metric is above a maximum threshold."""

    if isinstance(value, int | float) and value > threshold:
        alerts.append(
            {
                "level": "warning",
                "metric": metric,
                "message": message,
                "value": round(float(value), 4),
                "threshold": threshold,
            }
        )


def _add_min_alert(
    alerts: list[dict[str, Any]],
    *,
    metric: str,
    value: Any,
    threshold: float,
    message: str,
) -> None:
    """Append an alert when a metric is below a minimum threshold."""

    if isinstance(value, int | float) and value < threshold:
        alerts.append(
            {
                "level": "warning",
                "metric": metric,
                "message": message,
                "value": round(float(value), 4),
                "threshold": threshold,
            }
        )


def _env_float(name: str, default: float) -> float:
    """Read a numeric environment variable with a safe fallback."""

    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
