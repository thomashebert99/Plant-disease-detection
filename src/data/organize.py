"""Utilities to reorganize PlantVillage data into task-specific folders."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from src.data.align_labels import build_test_ood_dataset
from src.data.files import build_prefixed_filename, materialize_file, normalize_copy_mode

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
RAW_SPLITS = ("train", "valid")
TARGET_SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RAW_TO_PROJECT_LABELS = {
    "Apple___Apple_scab": ("apple", "Apple_Scab"),
    "Apple___Black_rot": ("apple", "Black_Rot"),
    "Apple___Cedar_apple_rust": ("apple", "Cedar_Apple_Rust"),
    "Apple___healthy": ("apple", "Healthy"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "corn",
        "Cercospora_Leaf_Spot",
    ),
    "Corn_(maize)___Common_rust_": ("corn", "Common_Rust"),
    "Corn_(maize)___Northern_Leaf_Blight": ("corn", "Northern_Leaf_Blight"),
    "Corn_(maize)___healthy": ("corn", "Healthy"),
    "Grape___Black_rot": ("grape", "Black_Rot"),
    "Grape___Esca_(Black_Measles)": ("grape", "Esca_Black_Measles"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("grape", "Leaf_Blight"),
    "Grape___healthy": ("grape", "Healthy"),
    "Pepper,_bell___Bacterial_spot": ("pepper", "Bacterial_Spot"),
    "Pepper,_bell___healthy": ("pepper", "Healthy"),
    "Potato___Early_blight": ("potato", "Early_Blight"),
    "Potato___Late_blight": ("potato", "Late_Blight"),
    "Potato___healthy": ("potato", "Healthy"),
    "Strawberry___Leaf_scorch": ("strawberry", "Leaf_Scorch"),
    "Strawberry___healthy": ("strawberry", "Healthy"),
    "Tomato___Bacterial_spot": ("tomato", "Bacterial_Spot"),
    "Tomato___Early_blight": ("tomato", "Early_Blight"),
    "Tomato___Late_blight": ("tomato", "Late_Blight"),
    "Tomato___Leaf_Mold": ("tomato", "Leaf_Mold"),
    "Tomato___Septoria_leaf_spot": ("tomato", "Septoria_Leaf_Spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("tomato", "Spider_Mites"),
    "Tomato___Target_Spot": ("tomato", "Target_Spot"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("tomato", "Yellow_Leaf_Curl_Virus"),
    "Tomato___Tomato_mosaic_virus": ("tomato", "Mosaic_Virus"),
    "Tomato___healthy": ("tomato", "Healthy"),
}
RETAINED_SPECIES = tuple(sorted({species for species, _ in RAW_TO_PROJECT_LABELS.values()}))
DEFAULT_RAW_PLANTVILLAGE_DIR = Path("data/raw/plantvillage")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_RAW_PLANTDOC_DIR = Path("data/raw/plantdoc")
DEFAULT_TEST_OOD_DIR = Path("data/test_ood")
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True, slots=True)
class PlantVillageImage:
    """Describe one PlantVillage image and the normalized labels it maps to."""

    source_path: Path
    raw_split: str
    source_label: str
    species: str
    disease: str


def organize_processed_dataset(
    source_dir: Path = DEFAULT_RAW_PLANTVILLAGE_DIR,
    target_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    plantdoc_dir: Path = DEFAULT_RAW_PLANTDOC_DIR,
    test_ood_dir: Path = DEFAULT_TEST_OOD_DIR,
    copy_mode: str = "hardlink",
    clean: bool = False,
) -> dict[str, Any]:
    """Build the processed datasets for species classification and disease routing."""

    plantdoc_dir = plantdoc_dir.resolve()
    test_ood_dir = test_ood_dir.resolve()
    report = build_processed_splits(
        source_dir=source_dir,
        target_dir=target_dir,
        copy_mode=copy_mode,
        clean=clean,
    )
    report["plantdoc_source"] = str(plantdoc_dir)
    report["test_ood_target"] = str(test_ood_dir)

    report["test_ood"] = build_test_ood_dataset(
        source_dir=plantdoc_dir,
        target_dir=test_ood_dir,
        copy_mode=copy_mode,
        clean=clean,
    )
    return _serialize_report(report)


def build_processed_splits(
    source_dir: Path = DEFAULT_RAW_PLANTVILLAGE_DIR,
    target_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    copy_mode: str = "hardlink",
    clean: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Materialize a deterministic stratified 70/15/15 split from PlantVillage."""

    copy_mode = normalize_copy_mode(copy_mode)
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()

    if clean:
        _clear_processed_directories(target_dir)

    _validate_plantvillage_layout(source_dir)
    _ensure_processed_roots(target_dir)

    report = _build_empty_processed_report(
        source_dir=source_dir,
        target_dir=target_dir,
        copy_mode=copy_mode,
        random_state=random_state,
    )

    source_images = [
        image
        for raw_split in RAW_SPLITS
        for image in _iter_plantvillage_images(source_dir / raw_split, raw_split)
    ]
    split_images = _split_images_by_source_label(
        source_images,
        random_state=random_state,
    )

    for target_split, images in split_images.items():
        for image in images:
            _materialize_processed_image(
                image=image,
                target_dir=target_dir,
                target_split=target_split,
                copy_mode=copy_mode,
            )

            report["processed_images"] += 1
            report["species"][target_split][image.species] += 1
            report["diseases"][target_split][image.species][image.disease] += 1

    report["ignored_labels"] = sorted(_discover_ignored_labels(source_dir))
    _log_processed_summary(report)
    return _serialize_report(report)


def _iter_plantvillage_images(split_dir: Path, raw_split: str) -> list[PlantVillageImage]:
    images: list[PlantVillageImage] = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        normalized = RAW_TO_PROJECT_LABELS.get(class_dir.name)
        if normalized is None:
            continue

        species, disease = normalized
        for image_path in sorted(path for path in class_dir.iterdir() if _is_image_file(path)):
            images.append(
                PlantVillageImage(
                    source_path=image_path,
                    raw_split=raw_split,
                    source_label=class_dir.name,
                    species=species,
                    disease=disease,
                )
            )
    return images


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _split_images_by_source_label(
    images: list[PlantVillageImage],
    *,
    random_state: int,
) -> dict[str, list[PlantVillageImage]]:
    grouped_images: dict[str, list[PlantVillageImage]] = defaultdict(list)
    for image in images:
        grouped_images[image.source_label].append(image)

    split_images: dict[str, list[PlantVillageImage]] = {"train": [], "val": [], "test": []}
    for source_label, label_images in sorted(grouped_images.items()):
        shuffled_images = sorted(label_images, key=lambda image: image.source_path.name)
        random.Random(f"{random_state}:{source_label}").shuffle(shuffled_images)

        train_count, val_count, test_count = _split_counts(len(shuffled_images))

        split_images["train"].extend(shuffled_images[:train_count])
        split_images["val"].extend(shuffled_images[train_count : train_count + val_count])
        split_images["test"].extend(shuffled_images[train_count + val_count :])

    return split_images


def _split_counts(image_count: int) -> tuple[int, int, int]:
    if image_count < 1:
        return 0, 0, 0
    if image_count == 1:
        return 1, 0, 0
    if image_count == 2:
        return 1, 1, 0

    train_count = max(1, round(image_count * SPLIT_RATIOS["train"]))
    val_count = max(1, round(image_count * SPLIT_RATIOS["val"]))
    test_count = image_count - train_count - val_count

    if test_count < 1:
        train_count -= 1 - test_count
        test_count = 1

    if train_count < 1:
        val_count -= 1 - train_count
        train_count = 1

    return train_count, val_count, test_count


def _materialize_processed_image(
    *,
    image: PlantVillageImage,
    target_dir: Path,
    target_split: str,
    copy_mode: str,
) -> None:
    species_destination = (
        target_dir
        / "species"
        / target_split
        / image.species
        / build_prefixed_filename(image.source_label, image.source_path.name)
    )
    disease_destination = (
        target_dir
        / image.species
        / target_split
        / image.disease
        / image.source_path.name
    )
    materialize_file(image.source_path, species_destination, copy_mode)
    materialize_file(image.source_path, disease_destination, copy_mode)


def _validate_plantvillage_layout(source_dir: Path) -> None:
    missing_splits = [
        split_name for split_name in RAW_SPLITS if not (source_dir / split_name).is_dir()
    ]
    if missing_splits:
        missing = ", ".join(missing_splits)
        raise FileNotFoundError(
            f"PlantVillage incomplet dans {source_dir}: split(s) manquant(s) {missing}"
        )


def _ensure_processed_roots(target_dir: Path) -> None:
    for split in TARGET_SPLITS:
        for species in RETAINED_SPECIES:
            (target_dir / "species" / split / species).mkdir(parents=True, exist_ok=True)
            (target_dir / species / split).mkdir(parents=True, exist_ok=True)


def _clear_processed_directories(target_dir: Path) -> None:
    managed_directories = [
        target_dir / "species",
        *(target_dir / species for species in RETAINED_SPECIES),
    ]
    for directory in managed_directories:
        if directory.exists():
            shutil.rmtree(directory)


def _clear_managed_directories(target_dir: Path, test_ood_dir: Path) -> None:
    _clear_processed_directories(target_dir)
    if test_ood_dir.exists():
        shutil.rmtree(test_ood_dir)


def _discover_ignored_labels(source_dir: Path) -> set[str]:
    ignored_labels: set[str] = set()
    for raw_split in RAW_SPLITS:
        split_dir = source_dir / raw_split
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir() and class_dir.name not in RAW_TO_PROJECT_LABELS:
                ignored_labels.add(class_dir.name)
    return ignored_labels


def _build_empty_processed_report(
    *,
    source_dir: Path,
    target_dir: Path,
    copy_mode: str,
    random_state: int,
) -> dict[str, Any]:
    return {
        "source": str(source_dir),
        "target": str(target_dir),
        "copy_mode": copy_mode,
        "split_strategy": "stratified_70_15_15_holdout",
        "resplit_performed": True,
        "random_state": random_state,
        "split_ratios": SPLIT_RATIOS,
        "processed_images": 0,
        "ignored_labels": [],
        "species": {
            split: defaultdict(int) for split in TARGET_SPLITS
        },
        "diseases": {
            split: defaultdict(Counter) for split in TARGET_SPLITS
        },
    }


def _serialize_report(report: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(report)

    if "species" in serialized:
        serialized["species"] = {
            split: dict(sorted(counts.items()))
            for split, counts in serialized["species"].items()
        }

    if "diseases" in serialized:
        serialized["diseases"] = {
            split: {
                species: dict(sorted(class_counts.items()))
                for species, class_counts in sorted(species_counts.items())
            }
            for split, species_counts in serialized["diseases"].items()
        }

    if "test_ood" in serialized and isinstance(serialized["test_ood"], dict):
        serialized["test_ood"] = _serialize_report(serialized["test_ood"])

    return serialized


def _log_processed_summary(report: dict[str, Any]) -> None:
    logger.info(
        "Organisation PlantVillage terminée: {} images retenues vers {}",
        report["processed_images"],
        report["target"],
    )
    if report["ignored_labels"]:
        logger.info("Labels PlantVillage ignorés: {}", ", ".join(report["ignored_labels"]))

    for split, species_counts in report["species"].items():
        logger.info("Split {} - espèces: {}", split, dict(sorted(species_counts.items())))

    for split, species_counts in report["diseases"].items():
        flattened = {
            species: dict(sorted(class_counts.items()))
            for species, class_counts in sorted(species_counts.items())
        }
        logger.info("Split {} - classes maladie: {}", split, flattened)
def main() -> None:
    """CLI entrypoint for dataset organization."""

    parser = argparse.ArgumentParser(
        description="Réorganise PlantVillage vers data/processed et prépare test_ood."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_RAW_PLANTVILLAGE_DIR,
        help="Dossier racine de PlantVillage contenant train/ et valid/.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Dossier de sortie pour les datasets train/val réorganisés.",
    )
    parser.add_argument(
        "--plantdoc-dir",
        type=Path,
        default=DEFAULT_RAW_PLANTDOC_DIR,
        help="Dossier source PlantDoc pour préparer le test OOD.",
    )
    parser.add_argument(
        "--test-ood-dir",
        type=Path,
        default=DEFAULT_TEST_OOD_DIR,
        help="Dossier de sortie pour le test OOD aligné.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Mode de matérialisation des fichiers dans les dossiers cibles.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Supprime les dossiers gérés avant de reconstruire les sorties.",
    )
    args = parser.parse_args()

    report = organize_processed_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        plantdoc_dir=args.plantdoc_dir,
        test_ood_dir=args.test_ood_dir,
        copy_mode=args.copy_mode,
        clean=args.clean,
    )
    logger.info("Manifeste final: {}", report)


if __name__ == "__main__":
    main()
