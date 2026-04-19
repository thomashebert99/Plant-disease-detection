"""Prediction endpoints."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from src.api.model_loader import (
    ModelConfigError,
    ModelNotAvailableError,
    model_source,
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
from src.monitoring.tracker import log_prediction

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    species: SpeciesEnum | None = Form(None),
) -> PredictionResponse:
    """Predict species when needed, then route to the matching disease model."""

    start_time = time.perf_counter()
    mode = SpeciesSource.manual if species is not None else SpeciesSource.auto
    species_prediction: SpeciesResult | None = None

    try:
        image_batch = await _read_image_batch(file)

        if species is None:
            species_prediction = _predict_species_result(image_batch, source=SpeciesSource.auto)
            threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
            if species_prediction.confidence < threshold:
                response = PredictionResponse(
                    status=PredictionStatus.uncertain_species,
                    species=species_prediction,
                    disease=None,
                    action_required=(
                        "Espèce détectée avec une confiance insuffisante. "
                        "Merci de confirmer l'espèce avant le diagnostic."
                    ),
                )
                _log_prediction_response(
                    endpoint="/predict",
                    mode=mode,
                    response=response,
                    start_time=start_time,
                )
                return response
            detected_species = species_prediction.species
        else:
            detected_species = species.value
            species_prediction = SpeciesResult(
                species=detected_species,
                confidence=1.0,
                source=SpeciesSource.manual,
            )

        disease_prediction = _predict_disease_result(detected_species, image_batch)
        response = PredictionResponse(
            status=PredictionStatus.ok,
            species=species_prediction,
            disease=disease_prediction,
            action_required=None,
        )
        _log_prediction_response(
            endpoint="/predict",
            mode=mode,
            response=response,
            start_time=start_time,
        )
        return response
    except HTTPException as exc:
        _log_prediction_error(
            endpoint="/predict",
            mode=mode,
            start_time=start_time,
            exception=exc,
            species_result=species_prediction,
        )
        raise


@router.post("/species", response_model=SpeciesResult)
async def predict_species(file: UploadFile = File(...)) -> SpeciesResult:
    """Detect only the plant species."""

    start_time = time.perf_counter()
    try:
        image_batch = await _read_image_batch(file)
        result = _predict_species_result(image_batch, source=SpeciesSource.auto)
        _log_prediction_event(
            {
                "endpoint": "/predict/species",
                "mode": SpeciesSource.auto.value,
                "status": PredictionStatus.ok.value,
                "species": result.species,
                "species_confidence": result.confidence,
                "latency_ms": _elapsed_ms(start_time),
            }
        )
        return result
    except HTTPException as exc:
        _log_prediction_error(
            endpoint="/predict/species",
            mode=SpeciesSource.auto,
            start_time=start_time,
            exception=exc,
        )
        raise


@router.post("/disease", response_model=PredictionResponse)
async def predict_disease(
    file: UploadFile = File(...),
    species: SpeciesEnum = Form(...),
) -> PredictionResponse:
    """Diagnose disease for a user-provided species."""

    start_time = time.perf_counter()
    species_result = SpeciesResult(
        species=species.value,
        confidence=1.0,
        source=SpeciesSource.manual,
    )
    try:
        image_batch = await _read_image_batch(file)
        disease_result = _predict_disease_result(species.value, image_batch)
        response = PredictionResponse(
            status=PredictionStatus.ok,
            species=species_result,
            disease=disease_result,
            action_required=None,
        )
        _log_prediction_response(
            endpoint="/predict/disease",
            mode=SpeciesSource.manual,
            response=response,
            start_time=start_time,
        )
        return response
    except HTTPException as exc:
        _log_prediction_error(
            endpoint="/predict/disease",
            mode=SpeciesSource.manual,
            start_time=start_time,
            exception=exc,
            species_result=species_result,
        )
        raise


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
        top_predictions=prediction.top_predictions,
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
        top_predictions=prediction.top_predictions,
    )


def _log_prediction_response(
    *,
    endpoint: str,
    mode: SpeciesSource,
    response: PredictionResponse,
    start_time: float,
) -> None:
    """Log a successful or uncertain prediction response without image data."""

    payload: dict[str, Any] = {
        "endpoint": endpoint,
        "mode": mode.value,
        "status": response.status.value,
        "species": response.species.species,
        "species_confidence": response.species.confidence,
        "species_source": response.species.source.value,
        "latency_ms": _elapsed_ms(start_time),
    }
    if response.disease is not None:
        payload["disease"] = response.disease.disease
        payload["disease_confidence"] = response.disease.confidence

    _log_prediction_event(payload)


def _log_prediction_error(
    *,
    endpoint: str,
    mode: SpeciesSource,
    start_time: float,
    exception: HTTPException,
    species_result: SpeciesResult | None = None,
) -> None:
    """Log a prediction error in a compact, non-sensitive form."""

    payload: dict[str, Any] = {
        "endpoint": endpoint,
        "mode": mode.value,
        "status": "error",
        "error_code": exception.status_code,
        "error_message": str(exception.detail),
        "latency_ms": _elapsed_ms(start_time),
    }
    if species_result is not None:
        payload["species"] = species_result.species
        payload["species_confidence"] = species_result.confidence
        payload["species_source"] = species_result.source.value

    _log_prediction_event(payload)


def _log_prediction_event(payload: dict[str, Any]) -> None:
    """Add common metadata and write a prediction monitoring event."""

    event = {
        "event_type": "prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_source": _safe_model_source(),
        **_json_ready(payload),
    }

    try:
        log_prediction(event)
    except Exception as exc:  # pragma: no cover - logging must never break inference.
        logger.warning("Monitoring prediction ignore apres echec: {}", exc)


def _elapsed_ms(start_time: float) -> float:
    """Return elapsed time in milliseconds."""

    return round((time.perf_counter() - start_time) * 1000, 2)


def _safe_model_source() -> str:
    """Return model source without letting monitoring raise API errors."""

    try:
        return model_source()
    except Exception:
        return os.getenv("MODEL_SOURCE", "local")


def _json_ready(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert enum values to plain JSON-compatible values."""

    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Enum):
            cleaned[key] = value.value
        else:
            cleaned[key] = value
    return cleaned
