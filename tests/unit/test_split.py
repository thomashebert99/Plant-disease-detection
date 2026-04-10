"""Tests for split materialization helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.split import build_train_val_splits, validate_split_directories


def test_validate_split_directories_checks_train_and_val_presence(tmp_path: Path) -> None:
    """Both train and valid directories must exist."""

    train_dir = tmp_path / "train"
    val_dir = tmp_path / "valid"
    train_dir.mkdir()
    val_dir.mkdir()

    assert validate_split_directories(train_dir, val_dir) is True
    assert validate_split_directories(train_dir, tmp_path / "missing") is False


def test_build_train_val_splits_preserves_provided_kaggle_split(tmp_path: Path) -> None:
    """The split step should materialize the provided directories without resplitting."""

    raw_dir = tmp_path / "raw" / "plantvillage"
    processed_dir = tmp_path / "processed"
    _write_image(raw_dir / "train" / "Tomato___Early_blight" / "train_a.JPG", "train")
    _write_image(raw_dir / "valid" / "Tomato___healthy" / "val_a.JPG", "val")

    report = build_train_val_splits(
        source_dir=raw_dir,
        target_dir=processed_dir,
        copy_mode="copy",
        random_state=42,
    )

    assert report["split_strategy"] == "dataset_provided_split"
    assert report["resplit_performed"] is False
    assert report["random_state"] == 42
    assert report["processed_images"] == 2
    assert report["species"]["train"] == {"tomato": 1}
    assert report["species"]["val"] == {"tomato": 1}
    assert (processed_dir / "species" / "train" / "tomato").exists()
    assert (processed_dir / "tomato" / "val" / "Healthy" / "val_a.JPG").exists()


def _write_image(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
