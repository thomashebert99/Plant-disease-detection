"""Tests for DagsHub/MLflow configuration."""

from __future__ import annotations

import mlflow
import pytest

from src.core.mlflow_config import REQUIRED_MLFLOW_ENV_VARS, setup_mlflow


def test_setup_mlflow_requires_tracking_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail fast when the DagsHub credentials are not loaded."""

    for name in REQUIRED_MLFLOW_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="Variables MLflow manquantes"):
        setup_mlflow(load_env=False)


def test_setup_mlflow_sets_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the standard MLflow environment variables for DagsHub."""

    previous_tracking_uri = mlflow.get_tracking_uri()
    tracking_uri = "https://dagshub.com/user/repo.mlflow"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "token")

    try:
        configured_uri = setup_mlflow(load_env=False)

        assert configured_uri == tracking_uri
        assert mlflow.get_tracking_uri() == tracking_uri
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
