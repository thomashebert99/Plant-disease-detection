"""Pydantic schemas for the prediction API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SpeciesEnum(str, Enum):
    """Species supported by the disease routing models."""

    tomato = "tomato"
    apple = "apple"
    grape = "grape"
    corn = "corn"
    potato = "potato"
    pepper = "pepper"
    strawberry = "strawberry"


class PredictionStatus(str, Enum):
    """Prediction status returned to the frontend."""

    ok = "ok"
    uncertain_species = "uncertain_species"


class SpeciesSource(str, Enum):
    """Whether the species was detected or provided by the user."""

    auto = "auto"
    manual = "manual"


class PredictionCandidate(BaseModel):
    """One candidate class returned by a model ensemble."""

    label: str = Field(..., examples=["tomato"])
    confidence: float = Field(..., ge=0.0, le=1.0)


class SpeciesResult(BaseModel):
    """Species prediction payload."""

    species: str = Field(..., examples=["tomato"])
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: SpeciesSource = Field(..., examples=["auto"])
    top_predictions: list[PredictionCandidate] = Field(default_factory=list)


class DiseaseResult(BaseModel):
    """Disease prediction payload."""

    disease: str = Field(..., examples=["Late_Blight"])
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_predictions: list[PredictionCandidate] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    """Full prediction response returned by `/predict`."""

    status: PredictionStatus
    species: SpeciesResult
    disease: DiseaseResult | None = None
    action_required: str | None = None


class ModelsInfoResponse(BaseModel):
    """Model configuration and loading status."""

    config_available: bool
    source: str
    complete: bool | None = None
    complete_tasks: list[str] = Field(default_factory=list)
    missing_tasks: list[str] = Field(default_factory=list)
    loaded_model_cache_size: int = 0
    tasks: dict[str, dict[str, object]] = Field(default_factory=dict)
    error: str | None = None


class FeedbackVerdict(str, Enum):
    """User feedback options for the last prediction."""

    correct = "correct"
    incorrect = "incorrect"
    unsure = "unsure"


class FeedbackRequest(BaseModel):
    """Feedback payload stored without the uploaded image."""

    verdict: FeedbackVerdict
    prediction_status: str | None = None
    predicted_species: str | None = None
    predicted_disease: str | None = None
    corrected_species: str | None = None
    corrected_disease: str | None = None
    comment: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    """Response returned after persisting user feedback."""

    stored: bool
    message: str
