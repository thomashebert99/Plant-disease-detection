"""Model construction helpers for transfer learning benchmarks."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import tensorflow as tf

DEFAULT_INPUT_SHAPE = (224, 224, 3)

AVAILABLE_ARCHITECTURES: dict[str, Callable[..., tf.keras.Model]] = {
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    "EfficientNetB1": tf.keras.applications.EfficientNetB1,
    "EfficientNetB3": tf.keras.applications.EfficientNetB3,
    "ResNet50V2": tf.keras.applications.ResNet50V2,
    "ResNet101V2": tf.keras.applications.ResNet101V2,
    "MobileNetV3Small": tf.keras.applications.MobileNetV3Small,
    "MobileNetV3Large": tf.keras.applications.MobileNetV3Large,
    "ConvNeXtTiny": tf.keras.applications.ConvNeXtTiny,
    "ConvNeXtSmall": tf.keras.applications.ConvNeXtSmall,
    "DenseNet121": tf.keras.applications.DenseNet121,
    "DenseNet169": tf.keras.applications.DenseNet169,
}

RECOMMENDED_SCREENING_ARCHITECTURES = (
    "MobileNetV3Small",
    "MobileNetV3Large",
    "EfficientNetB0",
    "EfficientNetB1",
    "ConvNeXtTiny",
    "DenseNet121",
)

DEFERRED_ARCHITECTURES = (
    "EfficientNetB3",
    "ResNet50V2",
    "ResNet101V2",
    "ConvNeXtSmall",
    "DenseNet169",
)


@tf.keras.utils.register_keras_serializable(package="plant_disease")
def densenet_preprocess_input(inputs: tf.Tensor) -> tf.Tensor:
    """Serializable wrapper around the DenseNet preprocessing function."""

    return tf.keras.applications.densenet.preprocess_input(inputs)


@tf.keras.utils.register_keras_serializable(package="plant_disease")
def resnet_v2_preprocess_input(inputs: tf.Tensor) -> tf.Tensor:
    """Serializable wrapper around the ResNetV2 preprocessing function."""

    return tf.keras.applications.resnet_v2.preprocess_input(inputs)


EXTERNAL_PREPROCESSORS: dict[str, Callable[[tf.Tensor], tf.Tensor]] = {
    "DenseNet121": densenet_preprocess_input,
    "DenseNet169": densenet_preprocess_input,
    "ResNet50V2": resnet_v2_preprocess_input,
    "ResNet101V2": resnet_v2_preprocess_input,
}


def list_available_architectures() -> list[str]:
    """Return the names accepted by `build_model`."""

    return sorted(AVAILABLE_ARCHITECTURES)


def list_recommended_screening_architectures() -> list[str]:
    """Return the architecture shortlist used for fast benchmark screening."""

    return list(RECOMMENDED_SCREENING_ARCHITECTURES)


def build_model(
    architecture: str,
    num_classes: int,
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    dropout_rate: float = 0.3,
    *,
    weights: str | None = "imagenet",
    dense_units: int = 256,
    preprocess_input: bool = True,
) -> tf.keras.Model:
    """Build a transfer-learning classifier with a frozen Keras Applications backbone.

    The returned model is intentionally not compiled: the training step owns the
    optimizer, loss, metrics, and fine-tuning schedule.
    """

    if architecture not in AVAILABLE_ARCHITECTURES:
        raise ValueError(
            f"Architecture inconnue : {architecture}. "
            f"Disponibles : {list_available_architectures()}"
        )
    if num_classes < 2:
        raise ValueError("num_classes doit etre superieur ou egal a 2.")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate doit etre dans l'intervalle [0, 1).")
    if dense_units < 1:
        raise ValueError("dense_units doit etre superieur ou egal a 1.")

    backbone_builder = AVAILABLE_ARCHITECTURES[architecture]
    base_model = backbone_builder(
        **_build_backbone_kwargs(
            backbone_builder=backbone_builder,
            input_shape=input_shape,
            weights=weights,
            preprocess_input=preprocess_input,
        )
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = _preprocess_inputs(inputs, architecture=architecture, enabled=preprocess_input)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="classifier_dense")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{architecture}_classifier")


def _build_backbone_kwargs(
    *,
    backbone_builder: Callable[..., tf.keras.Model],
    input_shape: tuple[int, int, int],
    weights: str | None,
    preprocess_input: bool,
) -> dict[str, Any]:
    """Build common Keras Applications kwargs."""

    kwargs: dict[str, Any] = {
        "include_top": False,
        "weights": weights,
        "input_shape": input_shape,
    }

    if "include_preprocessing" in inspect.signature(backbone_builder).parameters:
        kwargs["include_preprocessing"] = preprocess_input

    return kwargs


def _preprocess_inputs(
    inputs: tf.Tensor,
    *,
    architecture: str,
    enabled: bool,
) -> tf.Tensor:
    """Apply external Keras preprocessing when the backbone has no built-in layer."""

    if not enabled or architecture not in EXTERNAL_PREPROCESSORS:
        return inputs

    preprocessor = EXTERNAL_PREPROCESSORS[architecture]
    return tf.keras.layers.Lambda(preprocessor, name=f"{architecture}_preprocess")(inputs)
