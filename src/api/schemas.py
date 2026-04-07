"""Pydantic models for API inputs and outputs."""

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """Basic API response schema."""

    species: str = Field(..., examples=["tomato"])
    disease: str = Field(..., examples=["early_blight"])
    confidence: float = Field(..., ge=0.0, le=1.0)
