"""Model construction helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    """Configuration for a classification model."""

    backbone: str = "EfficientNetB3"
    num_classes: int = 2


def build_model(config: ModelConfig) -> dict[str, int | str]:
    """Return a serializable placeholder model description."""
    return {"backbone": config.backbone, "num_classes": config.num_classes}
