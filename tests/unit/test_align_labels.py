"""Tests for PlantDoc label alignment helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.align_labels import align_plantdoc_dataset, get_label_mapping


def test_get_label_mapping_normalizes_dataset_and_guide_labels() -> None:
    """Directory-style and guide-style labels should resolve to the same target."""

    mapping = get_label_mapping()

    assert mapping["tomato early blight leaf"] == ("tomato", "Early_Blight")
    assert mapping["tomato leaf late blight"] == ("tomato", "Late_Blight")
    assert mapping["corn leaf blight"] == ("corn", "Northern_Leaf_Blight")
    assert mapping["bell pepper leaf spot"] == ("pepper", "Bacterial_Spot")


def test_align_plantdoc_dataset_builds_test_ood_and_ignores_unknown_labels(
    tmp_path: Path,
) -> None:
    """Aligned PlantDoc images should be copied into species/class folders."""

    plantdoc_dir = tmp_path / "raw" / "plantdoc"
    test_ood_dir = tmp_path / "test_ood"
    _write_image(
        plantdoc_dir / "test" / "Tomato_Early_blight_leaf" / "img1.jpg",
        "ood-1",
    )
    _write_image(
        plantdoc_dir / "train" / "Tomato_leaf" / "img2.jpg",
        "ood-2",
    )
    _write_image(
        plantdoc_dir / "test" / "Unknown_label" / "img3.jpg",
        "ood-3",
    )

    report = align_plantdoc_dataset(
        source_dir=plantdoc_dir,
        target_dir=test_ood_dir,
        copy_mode="copy",
    )

    assert report["available"] is True
    assert report["processed_images"] == 2
    assert report["species"]["tomato"] == {"Early_Blight": 1, "Healthy": 1}
    assert report["ignored_labels"] == ["Unknown_label"]
    assert next((test_ood_dir / "tomato" / "Early_Blight").iterdir()).name.startswith(
        "Tomato_Early_blight_leaf__"
    )


def _write_image(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
