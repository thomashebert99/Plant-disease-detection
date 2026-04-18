"""Integration tests for API error handling."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app


@pytest.mark.asyncio
async def test_predict_rejects_invalid_image() -> None:
    """Unreadable uploads should return a client error, not a server crash."""

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/predict",
            files={"file": ("leaf.txt", b"not an image", "text/plain")},
            data={"species": "tomato"},
        )

    assert response.status_code == 400
    assert "image lisible" in response.json()["detail"]
