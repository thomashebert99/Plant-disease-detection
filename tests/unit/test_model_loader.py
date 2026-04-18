"""Tests for API model loading and soft-vote prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.api import model_loader


class FakeModel:
    """Small stand-in for a Keras model."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.array([probabilities], dtype="float32")

    def predict(self, image_batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Return deterministic probabilities."""

        assert verbose == 0
        assert image_batch.ndim == 4
        return self.probabilities


@pytest.fixture(autouse=True)
def clear_loader_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep env-driven loader caches isolated between tests."""

    monkeypatch.delenv("MODEL_SOURCE", raising=False)
    monkeypatch.delenv("ENSEMBLE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    model_loader.clear_model_caches()
    yield
    try:
        model_loader.clear_model_caches()
    except AttributeError:
        model_loader.load_ensemble_config.cache_clear()


def write_config(tmp_path: Path, models: list[dict[str, Any]]) -> Path:
    """Write a minimal ensemble config for tests."""

    config_path = tmp_path / "ensemble_config.json"
    payload = {
        "version": 1,
        "complete": True,
        "complete_tasks": ["species"],
        "missing_tasks": [],
        "tasks": {
            "species": {
                "task_type": "species",
                "display_name": "Espèce",
                "strategy": "soft_vote_mean_probabilities",
                "image_size": 224,
                "class_names": ["apple", "tomato"],
                "models": models,
            }
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_models_info_reports_missing_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The API can start before notebook 05 has produced the final config."""

    monkeypatch.setenv("ENSEMBLE_CONFIG_PATH", str(tmp_path / "missing.json"))

    info = model_loader.get_models_info()

    assert info["config_available"] is False
    assert info["source"] == "local"
    assert "Configuration d'ensemble introuvable" in info["error"]


def test_predict_task_uses_soft_vote(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Average model probabilities and return the selected class label."""

    first_checkpoint = tmp_path / "first.keras"
    second_checkpoint = tmp_path / "second.keras"
    first_checkpoint.write_text("fake", encoding="utf-8")
    second_checkpoint.write_text("fake", encoding="utf-8")
    config_path = write_config(
        tmp_path,
        [
            {
                "run_name": "first",
                "architecture": "EfficientNetB0",
                "checkpoint_path": str(first_checkpoint),
                "selected_rank": 1,
            },
            {
                "run_name": "second",
                "architecture": "ConvNeXtTiny",
                "checkpoint_path": str(second_checkpoint),
                "selected_rank": 2,
            },
        ],
    )
    monkeypatch.setenv("ENSEMBLE_CONFIG_PATH", str(config_path))

    def fake_load_model(architecture: str, artifact_path: str) -> FakeModel:
        if architecture == "EfficientNetB0":
            return FakeModel([0.70, 0.30])
        assert Path(artifact_path) == second_checkpoint
        return FakeModel([0.20, 0.80])

    monkeypatch.setattr(model_loader, "_load_keras_model", fake_load_model)

    result = model_loader.predict_task(
        "species",
        np.zeros((1, 224, 224, 3), dtype="float32"),
    )

    assert result.label == "tomato"
    assert result.class_index == 1
    assert result.confidence == pytest.approx(0.55)
    assert result.probabilities == pytest.approx([0.45, 0.55])
    assert result.model_count == 2
    assert [candidate["label"] for candidate in result.top_predictions] == ["tomato", "apple"]
    assert [candidate["confidence"] for candidate in result.top_predictions] == pytest.approx(
        [0.55, 0.45]
    )


def test_hub_mode_requires_hub_filename(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hub deployment configs must include remote filenames."""

    checkpoint = tmp_path / "model.keras"
    checkpoint.write_text("fake", encoding="utf-8")
    config_path = write_config(
        tmp_path,
        [
            {
                "run_name": "first",
                "architecture": "EfficientNetB0",
                "checkpoint_path": str(checkpoint),
                "selected_rank": 1,
            }
        ],
    )
    monkeypatch.setenv("ENSEMBLE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MODEL_SOURCE", "hub")
    monkeypatch.setenv("HF_REPO_ID", "user/repo")

    with pytest.raises(model_loader.ModelNotAvailableError, match="hub_filename"):
        model_loader.resolve_model_artifact(
            {
                "run_name": "first",
                "architecture": "EfficientNetB0",
            }
        )


def test_resolve_model_artifact_remaps_training_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Docker can load configs written with absolute paths from the training machine."""

    project_root = tmp_path / "project"
    checkpoint = project_root / "models" / "species" / "run" / "best_model.keras"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("fake", encoding="utf-8")

    training_machine_path = (
        "/home/user/code/Plant-disease-detection/"
        "models/species/run/best_model.keras"
    )
    monkeypatch.setattr(model_loader, "PROJECT_ROOT", project_root)

    resolved = model_loader.resolve_model_artifact(
        {
            "run_name": "first",
            "architecture": "EfficientNetB0",
            "checkpoint_path": training_machine_path,
        }
    )

    assert resolved == checkpoint
