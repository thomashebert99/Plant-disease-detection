"""Prediction endpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.model_loader import (
    ModelConfigError,
    ModelNotAvailableError,
    predict_task,
)
from src.api.preprocessing import preprocess_image_bytes
from src.api.schemas import (
    DiseaseResult,
    PredictionResponse,
    PredictionStatus,
    SpeciesEnum,
    SpeciesResult,
    SpeciesSource,
)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    species: SpeciesEnum | None = Form(None),
) -> PredictionResponse:
    """Predict species when needed, then route to the matching disease model."""

    image_batch = await _read_image_batch(file)

    if species is None:
        species_prediction = _predict_species_result(image_batch, source=SpeciesSource.auto)
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
        if species_prediction.confidence < threshold:
            return PredictionResponse(
                status=PredictionStatus.uncertain_species,
                species=species_prediction,
                disease=None,
                gradcam_base64=None,
                action_required=(
                    "Espèce détectée avec une confiance insuffisante. "
                    "Merci de confirmer l'espèce avant le diagnostic."
                ),
            )
        detected_species = species_prediction.species
    else:
        detected_species = species.value
        species_prediction = SpeciesResult(
            species=detected_species,
            confidence=1.0,
            source=SpeciesSource.manual,
        )

    disease_prediction = _predict_disease_result(detected_species, image_batch)
    return PredictionResponse(
        status=PredictionStatus.ok,
        species=species_prediction,
        disease=disease_prediction,
        gradcam_base64=None,
        action_required=None,
    )


@router.post("/species", response_model=SpeciesResult)
async def predict_species(file: UploadFile = File(...)) -> SpeciesResult:
    """Detect only the plant species."""

    image_batch = await _read_image_batch(file)
    return _predict_species_result(image_batch, source=SpeciesSource.auto)


@router.post("/disease", response_model=PredictionResponse)
async def predict_disease(
    file: UploadFile = File(...),
    species: SpeciesEnum = Form(...),
) -> PredictionResponse:
    """Diagnose disease for a user-provided species."""

    image_batch = await _read_image_batch(file)
    species_result = SpeciesResult(
        species=species.value,
        confidence=1.0,
        source=SpeciesSource.manual,
    )
    disease_result = _predict_disease_result(species.value, image_batch)
    return PredictionResponse(
        status=PredictionStatus.ok,
        species=species_result,
        disease=disease_result,
        gradcam_base64=None,
        action_required=None,
    )


async def _read_image_batch(file: UploadFile) -> object:
    """Read and preprocess an uploaded image."""

    try:
        return preprocess_image_bytes(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _predict_species_result(image_batch: object, *, source: SpeciesSource) -> SpeciesResult:
    """Run the species ensemble and convert its result to an API schema."""

    try:
        prediction = predict_task("species", image_batch)
    except (ModelConfigError, ModelNotAvailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SpeciesResult(
        species=prediction.label,
        confidence=prediction.confidence,
        source=source,
    )


def _predict_disease_result(species: str, image_batch: object) -> DiseaseResult:
    """Run the disease ensemble matching the selected species."""

    try:
        prediction = predict_task(species, image_batch)
    except (ModelConfigError, ModelNotAvailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return DiseaseResult(
        disease=prediction.label,
        confidence=prediction.confidence,
    )
