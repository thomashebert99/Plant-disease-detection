"""Training helpers for two-phase transfer learning."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf
from loguru import logger


@dataclass(slots=True)
class TrainingConfig:
    """Configuration shared by benchmark training runs."""

    phase1_epochs: int = 5
    phase2_epochs: int = 10
    phase1_learning_rate: float = 1e-3
    phase2_learning_rate: float = 1e-5
    fine_tune_layers: int = 50
    checkpoint_root: Path = Path("models")
    loss: str = "sparse_categorical_crossentropy"
    metrics: tuple[str, ...] = ("accuracy",)
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    monitor: str = "val_loss"
    checkpoint_monitor: str = "val_accuracy"


def train_model(
    model: tf.keras.Model,
    train_data: Any,
    val_data: Any,
    *,
    architecture: str,
    task: str,
    config: TrainingConfig | None = None,
    class_weight: Mapping[int, float] | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Train a transfer-learning model with optional frozen and fine-tuning phases."""

    config = config or TrainingConfig()
    if config.phase1_epochs < 0 or config.phase2_epochs < 0:
        raise ValueError("Les nombres d'epochs doivent etre positifs ou nuls.")
    if config.phase1_epochs == 0 and config.phase2_epochs == 0:
        raise ValueError("Au moins une phase d'entrainement doit avoir des epochs.")

    checkpoint_path = build_checkpoint_path(
        architecture=architecture,
        task=task,
        checkpoint_root=config.checkpoint_root,
    )

    phase1_history: dict[str, list[float]] = {}
    if config.phase1_epochs > 0:
        logger.info("Phase 1: entrainement de la tete pour {} / {}", task, architecture)
        compile_model(
            model=model,
            learning_rate=config.phase1_learning_rate,
            loss=config.loss,
            metrics=config.metrics,
        )
        phase1_fit_history = _fit_phase(
            model=model,
            train_data=train_data,
            val_data=val_data,
            architecture=architecture,
            task=task,
            phase="phase1",
            epochs=config.phase1_epochs,
            callbacks=build_callbacks(config=config, checkpoint_path=checkpoint_path),
            class_weight=class_weight,
            log_to_mlflow=log_to_mlflow,
            log_model=False,
        )
        phase1_history = phase1_fit_history.history

    trainable_layers = 0
    phase2_history: dict[str, list[float]] = {}
    if config.phase2_epochs > 0:
        trainable_layers = unfreeze_last_backbone_layers(
            model=model,
            layer_count=config.fine_tune_layers,
        )
        logger.info(
            "Phase 2: fine-tuning de {} couches du backbone pour {} / {}",
            trainable_layers,
            task,
            architecture,
        )
        compile_model(
            model=model,
            learning_rate=config.phase2_learning_rate,
            loss=config.loss,
            metrics=config.metrics,
        )
        phase2_fit_history = _fit_phase(
            model=model,
            train_data=train_data,
            val_data=val_data,
            architecture=architecture,
            task=task,
            phase="phase2",
            epochs=config.phase2_epochs,
            callbacks=build_callbacks(config=config, checkpoint_path=checkpoint_path),
            class_weight=class_weight,
            log_to_mlflow=log_to_mlflow,
            log_model=True,
        )
        phase2_history = phase2_fit_history.history

    return {
        "checkpoint_path": checkpoint_path,
        "fine_tuned_layers": trainable_layers,
        "phase1_history": phase1_history,
        "phase2_history": phase2_history,
    }


def compile_model(
    *,
    model: tf.keras.Model,
    learning_rate: float,
    loss: str,
    metrics: tuple[str, ...],
) -> None:
    """Compile a model for a benchmark phase."""

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=list(metrics),
    )


def build_callbacks(
    *,
    config: TrainingConfig,
    checkpoint_path: Path,
) -> list[tf.keras.callbacks.Callback]:
    """Build fresh callbacks for a training phase."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=config.monitor,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=config.monitor,
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=config.checkpoint_monitor,
            save_best_only=True,
        ),
    ]


def build_checkpoint_path(
    *,
    architecture: str,
    task: str,
    checkpoint_root: Path,
) -> Path:
    """Return the checkpoint path used by benchmark runs."""

    run_name = f"{_safe_name(architecture)}_{_safe_name(task)}"
    return checkpoint_root / run_name / "best_model.keras"


def unfreeze_last_backbone_layers(model: tf.keras.Model, layer_count: int = 50) -> int:
    """Unfreeze the last trainable-friendly layers of the nested backbone model."""

    if layer_count < 1:
        raise ValueError("layer_count doit etre superieur ou egal a 1.")

    backbone = find_backbone(model)
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    trainable_layers = 0
    for layer in backbone.layers[-layer_count:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization) or not layer.weights:
            continue
        layer.trainable = True
        trainable_layers += 1

    return trainable_layers


def find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Return the nested Keras Applications backbone from a classifier model."""

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer

    raise ValueError("Aucun backbone Keras imbrique trouve dans le modele.")


def _fit_phase(
    *,
    model: tf.keras.Model,
    train_data: Any,
    val_data: Any,
    architecture: str,
    task: str,
    phase: str,
    epochs: int,
    callbacks: list[tf.keras.callbacks.Callback],
    class_weight: Mapping[int, float] | None,
    log_to_mlflow: bool,
    log_model: bool,
) -> tf.keras.callbacks.History:
    """Fit one training phase and optionally log it to MLflow."""

    run_context = _mlflow_run(
        run_name=f"{task}_{architecture}_{phase}",
        enabled=log_to_mlflow,
    )
    with run_context:
        if log_to_mlflow:
            _log_phase_params(
                architecture=architecture,
                task=task,
                phase=phase,
                epochs=epochs,
                class_weight=class_weight,
            )

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
        )

        if log_to_mlflow:
            _log_history_metrics(history=history, phase=phase)
            if log_model:
                _log_model_to_mlflow(model)

    return history


def _mlflow_run(*, run_name: str, enabled: bool) -> Any:
    """Return an MLflow run context, or a no-op context when disabled."""

    if not enabled:
        return nullcontext()

    import mlflow

    return mlflow.start_run(run_name=run_name)


def _log_phase_params(
    *,
    architecture: str,
    task: str,
    phase: str,
    epochs: int,
    class_weight: Mapping[int, float] | None,
) -> None:
    """Log basic phase metadata to MLflow."""

    import mlflow

    mlflow.log_params(
        {
            "architecture": architecture,
            "task": task,
            "phase": phase,
            "epochs": epochs,
        }
    )
    if class_weight:
        mlflow.log_params(
            {f"class_weight_{label}": float(weight) for label, weight in class_weight.items()}
        )


def _log_history_metrics(*, history: tf.keras.callbacks.History, phase: str) -> None:
    """Log final and best validation metrics to MLflow."""

    import mlflow

    for name, values in history.history.items():
        if not values:
            continue
        mlflow.log_metric(f"{phase}_{name}", float(values[-1]))
        if name.startswith("val_"):
            best_value = min(values) if "loss" in name else max(values)
            mlflow.log_metric(f"{phase}_best_{name}", float(best_value))


def _log_model_to_mlflow(model: tf.keras.Model) -> None:
    """Log the final model artifact to MLflow without failing the training run."""

    import mlflow

    try:
        mlflow.tensorflow.log_model(model, artifact_path="model")
    except Exception as exc:  # pragma: no cover - exact MLflow errors depend on the backend.
        logger.warning(
            "Upload MLflow du modele ignore apres echec: {}. "
            "Le checkpoint local reste disponible.",
            exc,
        )


def _safe_name(value: str) -> str:
    """Return a filesystem-friendly run name fragment."""

    return value.strip().replace("/", "_").replace(" ", "_")
