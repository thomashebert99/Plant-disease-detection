"""Integration tests for prediction endpoints."""

from __future__ import annotations

from io import BytesIO

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.api.main import app
from src.api.model_loader import PredictionResult, clear_model_caches
from src.api.routers import predict as predict_router


def make_png_bytes() -> bytes:
    """Build a tiny valid RGB PNG image."""

    buffer = BytesIO()
    Image.new("RGB", (16, 16), color=(80, 160, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_predict_without_model_config_returns_503(monkeypatch) -> None:
    """Prediction should fail cleanly until notebook 05 produces the config."""

    monkeypatch.setenv("ENSEMBLE_CONFIG_PATH", "/tmp/does-not-exist/ensemble_config.json")
    clear_model_caches()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/predict",
            files={"file": ("leaf.png", make_png_bytes(), "image/png")},
            data={"species": "tomato"},
        )

    assert response.status_code == 503
    assert "Configuration d'ensemble introuvable" in response.json()["detail"]


@pytest.mark.asyncio
async def test_predict_species_without_model_config_returns_503(monkeypatch) -> None:
    """Species-only prediction uses the same model availability contract."""

    monkeypatch.setenv("ENSEMBLE_CONFIG_PATH", "/tmp/does-not-exist/ensemble_config.json")
    clear_model_caches()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/predict/species",
            files={"file": ("leaf.png", make_png_bytes(), "image/png")},
        )

    assert response.status_code == 503
    assert "Configuration d'ensemble introuvable" in response.json()["detail"]


@pytest.mark.asyncio
async def test_predict_logs_monitoring_summary(monkeypatch) -> None:
    """A successful prediction should be visible in the monitoring summary."""

    def fake_predict_task(task: str, image_batch: object) -> PredictionResult:
        if task == "species":
            return PredictionResult(
                task="species",
                label="tomato",
                class_index=0,
                confidence=0.92,
                probabilities=[0.92, 0.08],
                model_count=3,
                top_predictions=[
                    {"label": "tomato", "confidence": 0.92},
                    {"label": "apple", "confidence": 0.08},
                ],
            )
        return PredictionResult(
            task=task,
            label="Late_Blight",
            class_index=1,
            confidence=0.88,
            probabilities=[0.12, 0.88],
            model_count=3,
            top_predictions=[
                {"label": "Late_Blight", "confidence": 0.88},
                {"label": "Healthy", "confidence": 0.12},
            ],
        )

    monkeypatch.setattr(predict_router, "predict_task", fake_predict_task)
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.65")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        predict_response = await client.post(
            "/predict",
            files={"file": ("leaf.png", make_png_bytes(), "image/png")},
        )
        monitoring_response = await client.get("/monitoring/summary")

    assert predict_response.status_code == 200
    prediction_payload = predict_response.json()
    assert prediction_payload["status"] == "ok"
    assert prediction_payload["species"]["top_predictions"][0] == {
        "label": "tomato",
        "confidence": 0.92,
    }
    assert prediction_payload["disease"]["top_predictions"][0] == {
        "label": "Late_Blight",
        "confidence": 0.88,
    }

    assert monitoring_response.status_code == 200
    summary = monitoring_response.json()
    assert summary["total_predictions"] == 1
    assert summary["ok"] == 1
    assert summary["uncertain_species"] == 0
    assert summary["errors"] == 0
    assert summary["average_species_confidence"] == 0.92
    assert summary["average_disease_confidence"] == 0.88
