"""Tests for two-phase training helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.models.train import (
    TrainingConfig,
    build_callbacks,
    build_checkpoint_path,
    _log_model_to_mlflow,
    train_model,
    unfreeze_last_backbone_layers,
)


def test_build_checkpoint_path_sanitizes_run_name(tmp_path: Path) -> None:
    """Checkpoint folders should be safe for task names coming from notebooks."""

    checkpoint_path = build_checkpoint_path(
        architecture="MobileNetV3Small",
        task="tomato diseases",
        checkpoint_root=tmp_path,
    )

    assert checkpoint_path == tmp_path / "MobileNetV3Small_tomato_diseases" / "best_model.keras"


def test_build_callbacks_create_checkpoint_parent(tmp_path: Path) -> None:
    """ModelCheckpoint needs its parent folder before training starts."""

    checkpoint_path = tmp_path / "nested" / "best_model.keras"

    callbacks = build_callbacks(
        config=TrainingConfig(),
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.parent.is_dir()
    assert [type(callback).__name__ for callback in callbacks] == [
        "EarlyStopping",
        "ReduceLROnPlateau",
        "ModelCheckpoint",
    ]


def test_unfreeze_last_backbone_layers_keeps_batchnorm_frozen() -> None:
    """Fine-tuning should not accidentally thaw BatchNorm statistics."""

    model = _build_tiny_transfer_model()

    trainable_layers = unfreeze_last_backbone_layers(model=model, layer_count=4)
    backbone = model.get_layer("tiny_backbone")

    assert trainable_layers == 2
    assert backbone.get_layer("conv_1").trainable is True
    assert backbone.get_layer("batch_norm").trainable is False
    assert backbone.get_layer("conv_2").trainable is True


def test_unfreeze_last_backbone_layers_rejects_flat_model() -> None:
    """The helper expects the builder shape from src.models.build."""

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(8, 8, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )

    with pytest.raises(ValueError, match="Aucun backbone"):
        unfreeze_last_backbone_layers(model=model)


def test_train_model_runs_two_phases_without_mlflow(tmp_path: Path) -> None:
    """A tiny offline training run validates compilation, callbacks, and histories."""

    model = _build_tiny_transfer_model()
    train_data = _build_tiny_dataset()
    val_data = _build_tiny_dataset()
    config = TrainingConfig(
        phase1_epochs=1,
        phase2_epochs=1,
        fine_tune_layers=2,
        checkpoint_root=tmp_path,
        early_stopping_patience=1,
        reduce_lr_patience=1,
    )

    result = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        architecture="TinyNet",
        task="species",
        config=config,
        log_to_mlflow=False,
    )

    assert result["checkpoint_path"] == tmp_path / "TinyNet_species" / "best_model.keras"
    assert result["fine_tuned_layers"] == 1
    assert "loss" in result["phase1_history"]
    assert "loss" in result["phase2_history"]


def test_train_model_passes_class_weight_to_both_phases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Class weights from the EDA should be used in both training phases."""

    model = _build_tiny_transfer_model()
    class_weight = {0: 0.75, 1: 1.25}
    fit_class_weights = []

    def fake_fit(*args: object, **kwargs: object) -> tf.keras.callbacks.History:
        fit_class_weights.append(kwargs["class_weight"])
        history = tf.keras.callbacks.History()
        history.history = {"loss": [1.0], "val_loss": [1.1]}
        return history

    monkeypatch.setattr(model, "fit", fake_fit)

    train_model(
        model=model,
        train_data=_build_tiny_dataset(),
        val_data=_build_tiny_dataset(),
        architecture="TinyNet",
        task="species",
        config=TrainingConfig(
            phase1_epochs=1,
            phase2_epochs=1,
            fine_tune_layers=2,
            checkpoint_root=tmp_path,
        ),
        class_weight=class_weight,
        log_to_mlflow=False,
    )

    assert fit_class_weights == [class_weight, class_weight]


def test_train_model_can_run_phase1_only(tmp_path: Path) -> None:
    """Screening runs should be able to stop before fine-tuning."""

    model = _build_tiny_transfer_model()
    config = TrainingConfig(
        phase1_epochs=1,
        phase2_epochs=0,
        checkpoint_root=tmp_path,
        early_stopping_patience=1,
        reduce_lr_patience=1,
    )

    result = train_model(
        model=model,
        train_data=_build_tiny_dataset(),
        val_data=_build_tiny_dataset(),
        architecture="TinyNet",
        task="species_screening",
        config=config,
        log_to_mlflow=False,
    )

    assert result["fine_tuned_layers"] == 0
    assert "loss" in result["phase1_history"]
    assert result["phase2_history"] == {}


def test_train_model_can_run_phase2_only_from_existing_model(tmp_path: Path) -> None:
    """Fine-tuning final candidates can resume from a phase-1 checkpoint."""

    model = _build_tiny_transfer_model()
    config = TrainingConfig(
        phase1_epochs=0,
        phase2_epochs=1,
        fine_tune_layers=2,
        checkpoint_root=tmp_path,
        early_stopping_patience=1,
        reduce_lr_patience=1,
    )

    result = train_model(
        model=model,
        train_data=_build_tiny_dataset(),
        val_data=_build_tiny_dataset(),
        architecture="TinyNet",
        task="species_finetune",
        config=config,
        log_to_mlflow=False,
    )

    assert result["phase1_history"] == {}
    assert "loss" in result["phase2_history"]
    assert result["fine_tuned_layers"] == 1


def test_train_model_rejects_empty_training_plan(tmp_path: Path) -> None:
    """A benchmark call should request at least one training phase."""

    with pytest.raises(ValueError, match="Au moins une phase"):
        train_model(
            model=_build_tiny_transfer_model(),
            train_data=_build_tiny_dataset(),
            val_data=_build_tiny_dataset(),
            architecture="TinyNet",
            task="species",
            config=TrainingConfig(
                phase1_epochs=0,
                phase2_epochs=0,
                checkpoint_root=tmp_path,
            ),
            log_to_mlflow=False,
        )


def test_mlflow_model_upload_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote artifact upload issue should not invalidate a completed fit."""

    model = _build_tiny_transfer_model()

    class FakeTensorflowFlavor:
        @staticmethod
        def log_model(*args: object, **kwargs: object) -> None:
            raise RuntimeError("network broke during artifact upload")

    class FakeMlflow:
        tensorflow = FakeTensorflowFlavor()

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())

    _log_model_to_mlflow(model)


def _build_tiny_transfer_model() -> tf.keras.Model:
    """Return a small nested-backbone classifier."""

    backbone_inputs = tf.keras.Input(shape=(8, 8, 3), name="backbone_image")
    x = tf.keras.layers.Conv2D(2, kernel_size=3, padding="same", name="conv_1")(
        backbone_inputs
    )
    x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
    x = tf.keras.layers.Conv2D(2, kernel_size=3, padding="same", name="conv_2")(x)
    backbone = tf.keras.Model(backbone_inputs, x, name="tiny_backbone")
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(8, 8, 3), name="image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="tiny_classifier")


def _build_tiny_dataset() -> tf.data.Dataset:
    """Return a deterministic toy classification dataset."""

    images = np.random.default_rng(42).random((4, 8, 8, 3), dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.int32)

    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)
