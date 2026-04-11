"""MLflow configuration for the hosted DagsHub tracking server."""

from __future__ import annotations

import os

import mlflow
from dotenv import load_dotenv
from loguru import logger

REQUIRED_MLFLOW_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)


def setup_mlflow(
    *,
    experiment_name: str | None = None,
    load_env: bool = True,
) -> str:
    """Configure MLflow to log runs to the DagsHub tracking URI from the environment."""

    if load_env:
        load_dotenv()

    missing_vars = [name for name in REQUIRED_MLFLOW_ENV_VARS if not os.getenv(name)]
    if missing_vars:
        missing = ", ".join(missing_vars)
        raise RuntimeError(f"Variables MLflow manquantes: {missing}")

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    logger.info("MLflow configure vers {}", tracking_uri)
    return tracking_uri
