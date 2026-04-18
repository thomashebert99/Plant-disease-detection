"""Integration tests for prediction endpoints."""

from __future__ import annotations

from io import BytesIO

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.api.main import app
from src.api.model_loader import clear_model_caches


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
