"""Training loop placeholders."""

from __future__ import annotations


def train_model(model_name: str, epochs: int = 1) -> dict[str, int | str]:
    """Return minimal training metadata for future implementation."""
    return {"model_name": model_name, "epochs": epochs, "status": "not_implemented"}
