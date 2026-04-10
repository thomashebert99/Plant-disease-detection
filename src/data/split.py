"""Utilities for materializing and validating the provided PlantVillage split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from loguru import logger

from src.data.organize import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RAW_PLANTVILLAGE_DIR,
    build_processed_splits,
)


def validate_split_directories(train_dir: Path, val_dir: Path) -> bool:
    """Check that both provided split directories exist."""

    return train_dir.is_dir() and val_dir.is_dir()


def build_train_val_splits(
    source_dir: Path = DEFAULT_RAW_PLANTVILLAGE_DIR,
    target_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    copy_mode: str = "hardlink",
    clean: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Materialize the Kaggle-provided train/valid split into `data/processed/`."""

    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()
    train_dir = source_dir / "train"
    val_dir = source_dir / "valid"

    if not validate_split_directories(train_dir, val_dir):
        raise FileNotFoundError(
            f"PlantVillage incomplet dans {source_dir}: dossiers train/ et valid/ requis"
        )

    logger.info(
        "Le split Kaggle est conservé tel quel; random_state={} est journalisé pour reproductibilité uniquement.",
        random_state,
    )
    return build_processed_splits(
        source_dir=source_dir,
        target_dir=target_dir,
        copy_mode=copy_mode,
        clean=clean,
        random_state=random_state,
    )


def main() -> None:
    """CLI entrypoint for the split materialization step."""

    parser = argparse.ArgumentParser(
        description="Matérialise le split train/valid Kaggle vers data/processed."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_RAW_PLANTVILLAGE_DIR,
        help="Dossier racine PlantVillage contenant train/ et valid/.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Dossier de sortie data/processed.",
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
        help="Supprime les dossiers gérés dans data/processed avant reconstruction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Journalisé pour reproductibilité, sans re-splitting aléatoire.",
    )
    args = parser.parse_args()

    report = build_train_val_splits(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        copy_mode=args.copy_mode,
        clean=args.clean,
        random_state=args.random_state,
    )
    logger.info("Manifeste split: {}", report)


if __name__ == "__main__":
    main()
