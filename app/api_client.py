"""HTTP client helpers used by the Streamlit frontend."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any
from urllib import error, request

import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 60


@dataclass(slots=True)
class ApiResponse:
    """HTTP response returned by the API client."""

    status_code: int
    payload: dict[str, Any]


def get_api_url() -> str:
    """Return the API base URL from the environment."""

    return os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")


def call_predict_api(
    *,
    api_url: str,
    image_bytes: bytes,
    filename: str,
    species: str | None,
) -> ApiResponse:
    """Call `POST /predict` with multipart form data."""

    fields = {"species": species} if species else {}
    body, content_type = build_multipart_body(
        fields=fields,
        file_field="file",
        filename=filename,
        file_bytes=image_bytes,
    )
    http_request = request.Request(
        f"{api_url}/predict",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    return send_json_request(http_request, timeout=REQUEST_TIMEOUT_SECONDS)


def get_api_health(api_url: str) -> ApiResponse:
    """Call `GET /health`."""

    return cached_get_api_health(api_url)


def get_models_info(api_url: str) -> ApiResponse:
    """Call `GET /models/info`."""

    return cached_get_models_info(api_url)


def get_monitoring_summary(api_url: str) -> ApiResponse:
    """Call `GET /monitoring/summary`."""

    return cached_get_monitoring_summary(api_url)


@st.cache_data(ttl=10, show_spinner=False)
def cached_get_api_health(api_url: str) -> ApiResponse:
    """Call `GET /health` with a short cache to keep the UI responsive."""

    http_request = request.Request(f"{api_url}/health", method="GET")
    return send_json_request(http_request, timeout=3)


@st.cache_data(ttl=10, show_spinner=False)
def cached_get_models_info(api_url: str) -> ApiResponse:
    """Call `GET /models/info` with a short cache to avoid repeated polling."""

    http_request = request.Request(f"{api_url}/models/info", method="GET")
    return send_json_request(http_request, timeout=5)


@st.cache_data(ttl=5, show_spinner=False)
def cached_get_monitoring_summary(api_url: str) -> ApiResponse:
    """Call `GET /monitoring/summary` with a short cache for the dashboard."""

    http_request = request.Request(f"{api_url}/monitoring/summary", method="GET")
    return send_json_request(http_request, timeout=5)


def send_json_request(http_request: request.Request, *, timeout: int) -> ApiResponse:
    """Send an HTTP request and decode a JSON response."""

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            payload = decode_json_body(response.read())
            return ApiResponse(status_code=response.status, payload=payload)
    except error.HTTPError as exc:
        payload = decode_json_body(exc.read())
        return ApiResponse(status_code=exc.code, payload=payload)
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return ApiResponse(
            status_code=0,
            payload={"detail": f"Impossible de joindre le service : {exc}"},
        )


def decode_json_body(body: bytes) -> dict[str, Any]:
    """Decode an HTTP body as JSON with a readable fallback for proxy errors."""

    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        compact_text = " ".join(text.split())
        detail = compact_text[:500] if compact_text else "Réponse non JSON vide."
        return {"detail": detail}

    if isinstance(payload, dict):
        return payload
    return {"detail": str(payload)}


def build_multipart_body(
    *,
    fields: dict[str, str | None],
    file_field: str,
    filename: str,
    file_bytes: bytes,
) -> tuple[bytes, str]:
    """Build a multipart/form-data request body without an extra dependency."""

    boundary = f"----plant-disease-detection-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        if value is None:
            continue
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                f"{value}\r\n".encode(),
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
            ).encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
