"""Load selected Keras ensembles and run soft-vote predictions.

The API does not choose models itself. It consumes the configuration produced by
`notebooks/05_ensemble_selection.ipynb`, then loads either local checkpoints
during development or Hugging Face Hub artifacts during deployment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "models" / "ensemble_config.json"
DEFAULT_HUB_CONFIG_FILENAME = "ensemble_config.json"
SUPPORTED_MODEL_SOURCES = {"local", "hub"}


class ModelConfigError(RuntimeError):
    """Raised when the ensemble configuration cannot be read or is invalid."""


class ModelNotAvailableError(RuntimeError):
    """Raised when a configured task or model artifact is unavailable."""


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Prediction returned by a task ensemble."""

    task: str
    label: str
    class_index: int
    confidence: float
    probabilities: list[float]
    model_count: int


def model_source() -> str:
    """Return the configured model source: local checkpoints or HF Hub."""

    source = os.getenv("MODEL_SOURCE", "local").strip().lower()
    if source not in SUPPORTED_MODEL_SOURCES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_SOURCES))
        raise ModelConfigError(f"MODEL_SOURCE invalide: {source}. Valeurs: {supported}")
    return source


def ensemble_config_path() -> Path:
    """Return the local ensemble config path used in development."""

    configured_path = os.getenv("ENSEMBLE_CONFIG_PATH")
    if configured_path:
        return _resolve_project_path(configured_path)
    return DEFAULT_CONFIG_PATH


def is_model_config_available() -> bool:
    """Return whether the configured ensemble config can be loaded."""

    try:
        load_ensemble_config()
    except ModelConfigError:
        return False
    return True


def get_models_info() -> dict[str, Any]:
    """Return lightweight metadata for health checks and `/models/info`."""

    try:
        config = load_ensemble_config()
    except ModelConfigError as exc:
        return {
            "config_available": False,
            "source": model_source(),
            "error": str(exc),
            "tasks": {},
        }

    tasks = {}
    for task_name, task_payload in config.get("tasks", {}).items():
        models = task_payload.get("models", [])
        tasks[task_name] = {
            "task_type": task_payload.get("task_type"),
            "display_name": task_payload.get("display_name", task_name),
            "strategy": task_payload.get("strategy"),
            "class_count": len(task_payload.get("class_names", [])),
            "model_count": len(models),
            "architectures": [model.get("architecture") for model in models],
        }

    return {
        "config_available": True,
        "source": model_source(),
        "complete": bool(config.get("complete", False)),
        "complete_tasks": config.get("complete_tasks", []),
        "missing_tasks": config.get("missing_tasks", []),
        "loaded_model_cache_size": _load_keras_model.cache_info().currsize,
        "tasks": tasks,
    }


@lru_cache(maxsize=4)
def load_ensemble_config() -> dict[str, Any]:
    """Load the selected ensemble configuration.

    In local mode, the config is read from `models/ensemble_config.json`.
    In hub mode, the config is downloaded from `HF_REPO_ID`.
    """

    source = model_source()
    if source == "hub":
        config_path = _download_from_hub(
            os.getenv("ENSEMBLE_CONFIG_HUB_FILENAME", DEFAULT_HUB_CONFIG_FILENAME)
        )
    else:
        config_path = ensemble_config_path()

    if not config_path.exists():
        raise ModelConfigError(
            f"Configuration d'ensemble introuvable: {config_path}. "
            "Lance le notebook 05 pour générer models/ensemble_config.json."
        )

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelConfigError(f"Configuration JSON invalide: {config_path}") from exc

    _validate_config(config)
    return config


def predict_task(task: str, image_batch: np.ndarray) -> PredictionResult:
    """Predict one task with the configured soft-vote ensemble."""

    config = load_ensemble_config()
    task_payload = _get_task_payload(config, task)
    class_names = task_payload["class_names"]
    model_entries = task_payload["models"]

    _validate_image_batch(image_batch)

    probabilities = []
    for model_entry in model_entries:
        model = get_model(model_entry)
        prediction = model.predict(image_batch, verbose=0)[0]
        if len(prediction) != len(class_names):
            raise ModelConfigError(
                f"Sortie modèle incohérente pour {task}: "
                f"{len(prediction)} probas pour {len(class_names)} classes."
            )
        probabilities.append(prediction)

    average_proba = np.mean(probabilities, axis=0)
    class_index = int(np.argmax(average_proba))
    confidence = float(average_proba[class_index])

    return PredictionResult(
        task=task,
        label=str(class_names[class_index]),
        class_index=class_index,
        confidence=confidence,
        probabilities=[float(value) for value in average_proba],
        model_count=len(model_entries),
    )


def get_model(model_entry: dict[str, Any]) -> "tf.keras.Model":
    """Load one configured Keras checkpoint, using an in-memory cache."""

    architecture = str(model_entry.get("architecture", ""))
    artifact_path = resolve_model_artifact(model_entry)
    return _load_keras_model(architecture, str(artifact_path))


def resolve_model_artifact(model_entry: dict[str, Any]) -> Path:
    """Resolve a model artifact path from local disk or Hugging Face Hub."""

    if model_source() == "hub":
        hub_filename = model_entry.get("hub_filename")
        if not hub_filename:
            raise ModelNotAvailableError(
                "Entrée modèle sans `hub_filename`. "
                "Publie les modèles avec scripts/push_models_to_hub.py."
            )
        return _download_from_hub(str(hub_filename))

    checkpoint_path = model_entry.get("checkpoint_path")
    if not checkpoint_path:
        raise ModelNotAvailableError("Entrée modèle sans `checkpoint_path`.")

    resolved_path = _resolve_project_path(str(checkpoint_path))
    if not resolved_path.exists():
        raise ModelNotAvailableError(f"Checkpoint introuvable: {resolved_path}")
    return resolved_path


def clear_model_caches() -> None:
    """Clear config and Keras model caches, mainly for tests and notebooks."""

    load_ensemble_config.cache_clear()
    _load_keras_model.cache_clear()


@lru_cache(maxsize=16)
def _load_keras_model(architecture: str, artifact_path: str) -> "tf.keras.Model":
    """Load a Keras model with architecture-specific custom objects."""

    path = Path(artifact_path)
    if not path.exists():
        raise ModelNotAvailableError(f"Artifact modèle introuvable: {path}")

    import tensorflow as tf

    logger.info("Chargement du modèle {} depuis {}", architecture, path)
    return tf.keras.models.load_model(
        str(path),
        compile=False,
        safe_mode=False,
        custom_objects=_custom_objects_for_architecture(architecture),
    )


def _custom_objects_for_architecture(architecture: str) -> dict[str, Any]:
    """Return Keras custom objects needed to reload serialized preprocessors."""

    if architecture not in {"DenseNet121", "DenseNet169", "ResNet50V2", "ResNet101V2"}:
        return {}

    import tensorflow as tf

    from src.models.build import densenet_preprocess_input, resnet_v2_preprocess_input

    if architecture in {"DenseNet121", "DenseNet169"}:
        return {
            "preprocess_input": tf.keras.applications.densenet.preprocess_input,
            "densenet_preprocess_input": densenet_preprocess_input,
            "plant_disease>densenet_preprocess_input": densenet_preprocess_input,
        }

    return {
        "preprocess_input": tf.keras.applications.resnet_v2.preprocess_input,
        "resnet_v2_preprocess_input": resnet_v2_preprocess_input,
        "plant_disease>resnet_v2_preprocess_input": resnet_v2_preprocess_input,
    }


def _download_from_hub(filename: str) -> Path:
    """Download one file from the configured Hugging Face Hub model repo."""

    repo_id = os.getenv("HF_REPO_ID")
    if not repo_id:
        raise ModelConfigError("HF_REPO_ID est requis quand MODEL_SOURCE=hub.")

    from huggingface_hub import hf_hub_download

    token = os.getenv("HF_TOKEN") or None
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, token=token))


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the minimal contract expected from notebook 05."""

    if not isinstance(config.get("tasks"), dict) or not config["tasks"]:
        raise ModelConfigError("Configuration d'ensemble sans section `tasks`.")

    for task_name, task_payload in config["tasks"].items():
        class_names = task_payload.get("class_names")
        models = task_payload.get("models")
        if not isinstance(class_names, list) or not class_names:
            raise ModelConfigError(f"Tâche {task_name} sans `class_names`.")
        if not isinstance(models, list) or not models:
            raise ModelConfigError(f"Tâche {task_name} sans modèles sélectionnés.")
        for model_entry in models:
            if not model_entry.get("architecture"):
                raise ModelConfigError(f"Modèle de {task_name} sans architecture.")


def _get_task_payload(config: dict[str, Any], task: str) -> dict[str, Any]:
    """Return one task payload from the config or raise a clear error."""

    tasks = config.get("tasks", {})
    if task not in tasks:
        available = ", ".join(sorted(tasks))
        raise ModelNotAvailableError(f"Tâche modèle indisponible: {task}. Disponibles: {available}")
    return tasks[task]


def _validate_image_batch(image_batch: np.ndarray) -> None:
    """Ensure the API sends raw resized RGB batches to Keras."""

    if not isinstance(image_batch, np.ndarray):
        raise ValueError("image_batch doit être un numpy.ndarray.")
    if image_batch.ndim != 4 or image_batch.shape[-1] != 3:
        raise ValueError("image_batch doit avoir la forme (batch, height, width, 3).")
    if image_batch.shape[0] < 1:
        raise ValueError("image_batch doit contenir au moins une image.")


def _resolve_project_path(path_value: str) -> Path:
    """Resolve model paths across local notebooks, Docker and HF configs.

    Notebook 05 may write absolute checkpoint paths from the training machine.
    In Docker, the same `models/` directory is mounted under `/app/models`, so
    we remap any path containing a `models` segment to the current project root.
    """

    path = Path(path_value)
    if path.exists():
        return path

    if not path.is_absolute():
        return PROJECT_ROOT / path

    parts = path.parts
    if "models" in parts:
        models_index = parts.index("models")
        return PROJECT_ROOT.joinpath(*parts[models_index:])

    return path
