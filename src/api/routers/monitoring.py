"""Monitoring endpoints for lightweight prediction observability."""

from __future__ import annotations

from fastapi import APIRouter

from src.monitoring.tracker import summarize_predictions

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/summary")
async def monitoring_summary() -> dict[str, object]:
    """Return aggregated prediction monitoring metrics."""

    return summarize_predictions()
