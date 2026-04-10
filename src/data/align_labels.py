"""Helpers to align PlantDoc labels with project labels and build the OOD set."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_RAW_PLANTDOC_DIR = Path("data/raw/plantdoc")
DEFAULT_TEST_OOD_DIR = Path("data/test_ood")

# Correspondance PlantDoc -> labels projet
PLANTDOC_TO_PROJECT: dict[str, tuple[str, str]] = {
    # Tomate
    "Tomato Early blight leaf": ("tomato", "Early_Blight"),
    "Tomato Late blight leaf": ("tomato", "Late_Blight"),
    "Tomato leaf late blight": ("tomato", "Late_Blight"),
    "Tomato leaf": ("tomato", "Healthy"),
    "Tomato leaf bacterial spot": ("tomato", "Bacterial_Spot"),
    "Tomato leaf mosaic virus": ("tomato", "Mosaic_Virus"),
    "Tomato leaf yellow virus": ("tomato", "Yellow_Leaf_Curl_Virus"),
    "Tomato mold leaf": ("tomato", "Leaf_Mold"),
    "Tomato Septoria leaf spot": ("tomato", "Septoria_Leaf_Spot"),
    "Tomato two spotted spider mites leaf": ("tomato", "Spider_Mites"),
    # Pommier
    "Apple Scab Leaf": ("apple", "Apple_Scab"),
    "Apple leaf": ("apple", "Healthy"),
    "Apple rust leaf": ("apple", "Cedar_Apple_Rust"),
    # Vigne
    "Grape leaf": ("grape", "Healthy"),
    "Grape leaf black rot": ("grape", "Black_Rot"),
    # Mais
    "Corn rust leaf": ("corn", "Common_Rust"),
    "Corn Gray leaf spot": ("corn", "Cercospora_Leaf_Spot"),
    "Corn leaf blight": ("corn", "Northern_Leaf_Blight"),
    # Poivron
    "Bell_pepper leaf": ("pepper", "Healthy"),
    "Bell_pepper leaf spot": ("pepper", "Bacterial_Spot"),
    # Pomme de terre
    "Potato leaf": ("potato", "Healthy"),
    "Potato leaf early blight": ("potato", "Early_Blight"),
    "Potato leaf late blight": ("potato", "Late_Blight"),
    # Fraisier
    "Strawberry leaf": ("strawberry", "Healthy"),
}


def get_label_mapping() -> dict[str, tuple[str, str]]:
    """Return a normalized PlantDoc-to-project mapping."""

    return {
        _normalize_label(source_label): target
        for source_label, target in PLANTDOC_TO_PROJECT.items()
    }


def align_plantdoc_dataset(
    source_dir: Path = DEFAULT_RAW_PLANTDOC_DIR,
    target_dir: Path = DEFAULT_TEST_OOD_DIR,
    *,
    copy_mode: str = "hardlink",
    clean: bool = False,
) -> dict[str, Any]:
    """Copy aligned PlantDoc images into `data/test_ood/{species}/{class}/`."""

    copy_mode = _normalize_copy_mode(copy_mode)
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()

    report: dict[str, Any] = {
        "source": str(source_dir),
        "target": str(target_dir),
        "copy_mode": copy_mode,
        "available": False,
        "processed_images": 0,
        "species": defaultdict(Counter),
        "ignored_labels": [],
        "reason": None,
    }

    if clean and target_dir.exists():
        shutil.rmtree(target_dir)

    if not source_dir.exists():
        report["reason"] = "plantdoc_source_missing"
        logger.warning("PlantDoc introuvable, préparation du test OOD ignorée: {}", source_dir)
        return _serialize_alignment_report(report)

    image_files = list(_iter_plantdoc_images(source_dir))
    if not image_files:
        report["reason"] = "no_images_found"
        logger.warning("Aucune image PlantDoc détectée dans {}", source_dir)
        return _serialize_alignment_report(report)

    normalized_mapping = get_label_mapping()
    if not normalized_mapping:
        report["reason"] = "label_mapping_missing"
        logger.warning("Le mapping PlantDoc est vide, préparation du test OOD ignorée.")
        return _serialize_alignment_report(report)

    report["available"] = True
    ignored_labels: set[str] = set()

    for image_path in image_files:
        raw_label = image_path.parent.name
        normalized_label = normalized_mapping.get(_normalize_label(raw_label))
        if normalized_label is None:
            ignored_labels.add(raw_label)
            continue

        species, disease = normalized_label
        destination = (
            target_dir
            / species
            / disease
            / _build_prefixed_filename(raw_label, image_path.name)
        )
        _materialize_file(image_path, destination, copy_mode)

        report["processed_images"] += 1
        report["species"][species][disease] += 1

    report["ignored_labels"] = sorted(ignored_labels)
    _log_alignment_summary(report)
    return _serialize_alignment_report(report)


def build_test_ood_dataset(
    source_dir: Path = DEFAULT_RAW_PLANTDOC_DIR,
    target_dir: Path = DEFAULT_TEST_OOD_DIR,
    *,
    copy_mode: str = "hardlink",
    clean: bool = False,
) -> dict[str, Any]:
    """Backward-compatible wrapper used by the organization step."""

    return align_plantdoc_dataset(
        source_dir=source_dir,
        target_dir=target_dir,
        copy_mode=copy_mode,
        clean=clean,
    )


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.replace("_", " ").strip()).lower()


def _iter_plantdoc_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _normalize_copy_mode(copy_mode: str) -> str:
    normalized = copy_mode.lower().strip()
    if normalized not in {"copy", "hardlink"}:
        raise ValueError("copy_mode must be either 'copy' or 'hardlink'")
    return normalized


def _materialize_file(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    if copy_mode == "copy":
        shutil.copy2(source, destination)
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _build_prefixed_filename(prefix: str, filename: str) -> str:
    sanitized_prefix = prefix.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{sanitized_prefix}__{filename}"


def _serialize_alignment_report(report: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(report)
    serialized["species"] = {
        species: dict(sorted(class_counts.items()))
        for species, class_counts in sorted(serialized["species"].items())
    }
    return serialized


def _log_alignment_summary(report: dict[str, Any]) -> None:
    if not report["available"]:
        return

    logger.info(
        "Préparation PlantDoc terminée: {} images alignées vers {}",
        report["processed_images"],
        report["target"],
    )
    logger.info(
        "Répartition OOD par espèce/classe: {}",
        {
            species: dict(sorted(class_counts.items()))
            for species, class_counts in sorted(report["species"].items())
        },
    )
    if report["ignored_labels"]:
        logger.info("Labels PlantDoc ignorés: {}", ", ".join(report["ignored_labels"]))


def main() -> None:
    """CLI entrypoint for the PlantDoc label-alignment step."""

    parser = argparse.ArgumentParser(
        description="Aligne les labels PlantDoc et construit data/test_ood."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_RAW_PLANTDOC_DIR,
        help="Dossier racine PlantDoc contenant train/ et/ou test/.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TEST_OOD_DIR,
        help="Dossier cible pour le test OOD aligné.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Mode de matérialisation des fichiers.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Supprime data/test_ood avant reconstruction.",
    )
    args = parser.parse_args()

    report = align_plantdoc_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        copy_mode=args.copy_mode,
        clean=args.clean,
    )
    logger.info("Manifeste OOD: {}", report)


if __name__ == "__main__":
    main()
