"""User feedback endpoint for iterative model monitoring."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from src.api.schemas import FeedbackRequest, FeedbackResponse
from src.monitoring.tracker import log_feedback

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest) -> FeedbackResponse:
    """Store user feedback without storing the uploaded image."""

    payload = feedback.model_dump()
    event = {
        "event_type": "feedback",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    log_feedback(event)
    return FeedbackResponse(
        stored=True,
        message="Retour enregistré sans conservation de l'image.",
    )
