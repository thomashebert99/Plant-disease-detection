import os

import dagshub
from loguru import logger

import mlflow


def setup_mlflow() -> None:
    """
    Configure MLflow pour pointer vers DagsHub.
    Fonctionne identiquement en local et en production (Cloud Run).
    """
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USER"),
        repo_name=os.getenv("DAGSHUB_REPO"),
        mlflow=True,
    )
    logger.info(f"MLflow configuré → {mlflow.get_tracking_uri()}")
