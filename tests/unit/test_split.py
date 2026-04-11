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


def test_build_train_val_splits_creates_validation_and_test_holdout(tmp_path: Path) -> None:
    """The split step should build a stratified 70/15/15 holdout."""

    raw_dir = tmp_path / "raw" / "plantvillage"
    processed_dir = tmp_path / "processed"
    (raw_dir / "train").mkdir(parents=True)
    for index in range(10):
        _write_image(
            raw_dir / "valid" / "Tomato___healthy" / f"tomato_{index}.JPG",
            f"tomato-{index}",
        )

    report = build_train_val_splits(
        source_dir=raw_dir,
        target_dir=processed_dir,
        copy_mode="copy",
        random_state=42,
    )

    assert report["split_strategy"] == "stratified_70_15_15_holdout"
    assert report["resplit_performed"] is True
    assert report["random_state"] == 42
    assert report["split_ratios"] == {"train": 0.70, "val": 0.15, "test": 0.15}
    assert report["processed_images"] == 10
    assert report["species"]["train"] == {"tomato": 7}
    assert report["species"]["val"] == {"tomato": 2}
    assert report["species"]["test"] == {"tomato": 1}
    assert (processed_dir / "species" / "train" / "tomato").exists()
    assert len(list((processed_dir / "tomato" / "train" / "Healthy").iterdir())) == 7
    assert len(list((processed_dir / "tomato" / "val" / "Healthy").iterdir())) == 2
    assert len(list((processed_dir / "tomato" / "test" / "Healthy").iterdir())) == 1


def _write_image(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
