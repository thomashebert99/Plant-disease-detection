"""Dataset download entrypoints using the Kaggle API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DatasetSource:
    """Metadata describing a remote dataset source."""

    name: str
    kaggle_slug: str
    output_dir: str


def get_dataset_sources() -> list[DatasetSource]:
    """List the datasets expected by the project."""
    return [
        DatasetSource(
            name="plantvillage",
            kaggle_slug="vipoooool/new-plant-diseases-dataset",
            output_dir="data/raw/plantvillage",
        ),
        DatasetSource(
            name="plantdoc",
            kaggle_slug="nirmalsankalana/plantdoc-dataset",
            output_dir="data/raw/plantdoc",
        ),
    ]
