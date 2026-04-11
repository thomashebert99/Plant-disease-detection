"""Tests for PlantVillage/PlantDoc dataset organization helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.organize import organize_processed_dataset


def test_organize_processed_dataset_restructures_plantvillage_and_builds_ood(
    tmp_path: Path,
) -> None:
    """The organization step should build processed splits and aligned OOD data."""

    raw_dir = tmp_path / "raw" / "plantvillage"
    processed_dir = tmp_path / "processed"
    plantdoc_dir = tmp_path / "raw" / "plantdoc"
    test_ood_dir = tmp_path / "test_ood"

    _write_image(
        raw_dir / "train" / "Tomato___Early_blight" / "tomato_train_a.JPG",
        "tomato-train",
    )
    for index in range(10):
        _write_image(
            raw_dir / "valid" / "Tomato___healthy" / f"tomato_healthy_{index}.JPG",
            f"tomato-healthy-{index}",
        )
    _write_image(
        raw_dir / "train" / "Apple___Black_rot" / "apple_train_a.JPG",
        "apple-train",
    )
    _write_image(
        raw_dir / "train" / "Blueberry___healthy" / "blueberry_train_a.JPG",
        "ignored",
    )
    _write_image(
        plantdoc_dir / "test" / "Tomato_Early_blight_leaf" / "ood_1.jpg",
        "ood-train",
    )

    report = organize_processed_dataset(
        source_dir=raw_dir,
        target_dir=processed_dir,
        plantdoc_dir=plantdoc_dir,
        test_ood_dir=test_ood_dir,
        copy_mode="copy",
    )

    assert report["processed_images"] == 12
    assert report["ignored_labels"] == ["Blueberry___healthy"]
    assert report["species"]["train"] == {"apple": 1, "tomato": 8}
    assert report["species"]["val"] == {"tomato": 2}
    assert report["species"]["test"] == {"tomato": 1}
    assert report["diseases"]["train"]["tomato"] == {
        "Early_Blight": 1,
        "Healthy": 7,
    }
    assert report["diseases"]["val"]["tomato"] == {"Healthy": 2}
    assert report["diseases"]["test"]["tomato"] == {"Healthy": 1}
    assert report["test_ood"]["processed_images"] == 1
    assert report["test_ood"]["species"]["tomato"] == {"Early_Blight": 1}

    assert (
        processed_dir
        / "species"
        / "train"
        / "tomato"
        / "Tomato___Early_blight__tomato_train_a.JPG"
    ).exists()
    assert (processed_dir / "tomato" / "train" / "Early_Blight" / "tomato_train_a.JPG").exists()
    assert not (processed_dir / "species" / "train" / "blueberry").exists()
    assert any((processed_dir / "tomato" / "test" / "Healthy").iterdir())
    assert (test_ood_dir / "tomato" / "Early_Blight").exists()


def _write_image(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
