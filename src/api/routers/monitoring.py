"""Monitoring endpoints for lightweight prediction observability."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.monitoring.tracker import read_prediction_events, summarize_predictions

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/summary")
async def monitoring_summary() -> dict[str, object]:
    """Return aggregated prediction monitoring metrics."""

    return summarize_predictions()


@router.get("/events")
async def monitoring_events(
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, object]:
    """Return the latest prediction events without storing uploaded images."""

    events = [
        event
        for event in read_prediction_events(limit=limit)
        if str(event.get("event_type", "")) == "prediction"
    ]
    return {"events": events, "count": len(events), "limit": limit}
