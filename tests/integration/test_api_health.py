"""Integration tests for health and model metadata endpoints."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.api.model_loader import clear_model_caches


@pytest.mark.asyncio
async def test_api_health() -> None:
    """The health endpoint should stay lightweight and always available."""

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_models_info_without_config(monkeypatch) -> None:
    """The API should start before the final ensemble config exists."""

    monkeypatch.setenv("ENSEMBLE_CONFIG_PATH", "/tmp/does-not-exist/ensemble_config.json")
    clear_model_caches()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/models/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["config_available"] is False
    assert payload["source"] == "local"
    assert "Configuration d'ensemble introuvable" in payload["error"]
