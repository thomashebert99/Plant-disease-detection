"""Prediction endpoints."""

from fastapi import APIRouter

from src.api.schemas import PredictResponse

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
def predict() -> PredictResponse:
    """Return a placeholder prediction payload."""
    return PredictResponse(species="unknown", disease="unknown", confidence=0.0)
