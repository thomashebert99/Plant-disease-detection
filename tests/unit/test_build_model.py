"""Tests for transfer-learning model builders."""

from __future__ import annotations

from pathlib import Path

import pytest
import tensorflow as tf

from src.models import build as model_build


def test_build_model_uses_frozen_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build a small classifier around a fake backbone."""

    captured_kwargs: dict[str, object] = {}

    def fake_backbone(**kwargs: object) -> tf.keras.Model:
        captured_kwargs.update(kwargs)
        inputs = tf.keras.Input(shape=kwargs["input_shape"])
        outputs = tf.keras.layers.Conv2D(4, kernel_size=3, padding="same")(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="fake_backbone")

    monkeypatch.setitem(model_build.AVAILABLE_ARCHITECTURES, "FakeNet", fake_backbone)

    model = model_build.build_model(
        architecture="FakeNet",
        num_classes=3,
        input_shape=(32, 32, 3),
        dropout_rate=0.2,
        weights=None,
        dense_units=8,
    )

    assert model.output_shape == (None, 3)
    assert model.get_layer("fake_backbone").trainable is False
    assert captured_kwargs == {
        "include_top": False,
        "weights": None,
        "input_shape": (32, 32, 3),
    }


def test_build_model_uses_internal_preprocessing_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let compatible backbones consume raw 0-255 images by default."""

    captured_kwargs: dict[str, object] = {}

    def fake_backbone_with_preprocessing(
        *,
        include_top: bool,
        weights: str | None,
        input_shape: tuple[int, int, int],
        include_preprocessing: bool = True,
    ) -> tf.keras.Model:
        captured_kwargs.update(
            {
                "include_top": include_top,
                "weights": weights,
                "input_shape": input_shape,
                "include_preprocessing": include_preprocessing,
            }
        )
        inputs = tf.keras.Input(shape=input_shape)
        outputs = tf.keras.layers.Conv2D(4, kernel_size=3, padding="same")(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="fake_preprocessed_backbone")

    monkeypatch.setitem(
        model_build.AVAILABLE_ARCHITECTURES,
        "FakePreprocessedNet",
        fake_backbone_with_preprocessing,
    )

    model_build.build_model(
        architecture="FakePreprocessedNet",
        num_classes=2,
        input_shape=(32, 32, 3),
        weights=None,
    )

    assert captured_kwargs["include_preprocessing"] is True


def test_build_model_can_disable_internal_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow custom data pipelines to own preprocessing when needed."""

    captured_kwargs: dict[str, object] = {}

    def fake_backbone_with_preprocessing(
        *,
        include_top: bool,
        weights: str | None,
        input_shape: tuple[int, int, int],
        include_preprocessing: bool = True,
    ) -> tf.keras.Model:
        captured_kwargs.update(
            {
                "include_top": include_top,
                "weights": weights,
                "input_shape": input_shape,
                "include_preprocessing": include_preprocessing,
            }
        )
        inputs = tf.keras.Input(shape=input_shape)
        outputs = tf.keras.layers.Conv2D(4, kernel_size=3, padding="same")(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="fake_custom_backbone")

    monkeypatch.setitem(
        model_build.AVAILABLE_ARCHITECTURES,
        "FakeCustomPreprocessedNet",
        fake_backbone_with_preprocessing,
    )

    model_build.build_model(
        architecture="FakeCustomPreprocessedNet",
        num_classes=2,
        input_shape=(32, 32, 3),
        weights=None,
        preprocess_input=False,
    )

    assert captured_kwargs["include_preprocessing"] is False


def test_external_preprocessor_model_can_be_reloaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep Lambda-based external preprocessing compatible with Keras reloads."""

    def fake_backbone(**kwargs: object) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=kwargs["input_shape"])
        outputs = tf.keras.layers.Conv2D(4, kernel_size=3, padding="same")(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="fake_external_backbone")

    monkeypatch.setitem(model_build.AVAILABLE_ARCHITECTURES, "FakeExternalNet", fake_backbone)
    monkeypatch.setitem(
        model_build.EXTERNAL_PREPROCESSORS,
        "FakeExternalNet",
        model_build.densenet_preprocess_input,
    )

    model = model_build.build_model(
        architecture="FakeExternalNet",
        num_classes=2,
        input_shape=(32, 32, 3),
        weights=None,
    )
    model_path = tmp_path / "model.keras"

    model.save(model_path)
    reloaded_model = tf.keras.models.load_model(model_path, compile=False)

    assert reloaded_model.output_shape == (None, 2)


def test_build_model_rejects_unknown_architecture() -> None:
    """Fail fast when a benchmark asks for an unsupported backbone."""

    with pytest.raises(ValueError, match="Architecture inconnue"):
        model_build.build_model("UnknownNet", num_classes=2, weights=None)


def test_recommended_screening_architectures_are_available() -> None:
    """The fast benchmark shortlist should only contain supported builders."""

    available = set(model_build.list_available_architectures())

    assert set(model_build.list_recommended_screening_architectures()) <= available
    assert model_build.list_recommended_screening_architectures() == [
        "MobileNetV3Small",
        "MobileNetV3Large",
        "EfficientNetB0",
        "EfficientNetB1",
        "ConvNeXtTiny",
        "DenseNet121",
    ]
