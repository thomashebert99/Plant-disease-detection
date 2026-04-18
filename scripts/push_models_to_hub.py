"""Upload the selected ensemble models and config to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "models" / "ensemble_config.json"
DEFAULT_OUTPUT_CONFIG_PATH = PROJECT_ROOT / "models" / "ensemble" / "ensemble_config_hf.json"
DEFAULT_REMOTE_CONFIG_PATH = "ensemble_config.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Upload checkpoints selected by notebooks/05_ensemble_selection.ipynb "
            "and publish a Hub-ready ensemble_config.json."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Local ensemble config produced by notebook 05.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=DEFAULT_OUTPUT_CONFIG_PATH,
        help="Local copy of the Hub-ready config with hub_filename fields.",
    )
    parser.add_argument(
        "--remote-config-path",
        default=DEFAULT_REMOTE_CONFIG_PATH,
        help="Path of the config file inside the Hugging Face repo.",
    )
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_REPO_ID"),
        help="Hugging Face repo id, for example username/phyto-diagnose-models.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the Hub config without uploading files.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the local ensemble config."""

    config_path = config_path.resolve()
    if not config_path.exists():
        raise SystemExit(f"Config introuvable: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("complete", False):
        raise SystemExit(
            "La configuration n'est pas complète. "
            "Lance le notebook 05 après la fin de tous les fine-tunings."
        )
    return config


def build_hub_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the config with deterministic Hub filenames."""

    hub_config = deepcopy(config)
    hub_config["model_source"] = "huggingface_hub"

    for task_name, task_payload in hub_config["tasks"].items():
        for model_entry in task_payload["models"]:
            checkpoint_path = resolve_checkpoint_path(model_entry["checkpoint_path"])
            model_entry["checkpoint_path"] = str(checkpoint_path)
            model_entry["hub_filename"] = build_remote_model_path(task_name, model_entry)

    return hub_config


def upload_hub_config(
    *,
    config: dict[str, Any],
    output_config_path: Path,
    remote_config_path: str,
    repo_id: str,
    token: str | None,
    dry_run: bool,
) -> None:
    """Write the Hub config locally and upload it unless this is a dry run."""

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.info("Config Hub écrite dans {}", output_config_path)

    if dry_run:
        logger.info("Dry run: upload de la config ignoré.")
        return

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(output_config_path),
        path_in_repo=remote_config_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    logger.info("Config uploadée vers {}/{}", repo_id, remote_config_path)


def upload_model_artifacts(
    *,
    config: dict[str, Any],
    repo_id: str,
    token: str | None,
    dry_run: bool,
) -> None:
    """Upload all selected Keras checkpoints to their configured Hub paths."""

    api = HfApi(token=token)
    if not dry_run:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    for task_payload in config["tasks"].values():
        for model_entry in task_payload["models"]:
            checkpoint_path = resolve_checkpoint_path(model_entry["checkpoint_path"])
            remote_path = model_entry["hub_filename"]

            if not checkpoint_path.exists():
                raise SystemExit(f"Checkpoint introuvable: {checkpoint_path}")

            if dry_run:
                logger.info("Dry run: {} -> {}", checkpoint_path, remote_path)
                continue

            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            logger.info("Uploadé: {} -> {}/{}", checkpoint_path, repo_id, remote_path)


def build_remote_model_path(task_name: str, model_entry: dict[str, Any]) -> str:
    """Build a stable path in the Hub repo for one selected checkpoint."""

    selected_rank = int(model_entry.get("selected_rank", 0))
    run_name = safe_name(str(model_entry.get("run_name", "model")))
    architecture = safe_name(str(model_entry.get("architecture", "architecture")))
    return f"models/{safe_name(task_name)}/{selected_rank:02d}_{architecture}_{run_name}.keras"


def resolve_checkpoint_path(path_value: str) -> Path:
    """Resolve a checkpoint path from the project root when relative."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_name(value: str) -> str:
    """Return a filename-safe fragment while keeping names readable."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")


def main() -> None:
    """Upload selected models and the Hub-ready config."""

    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    if not args.repo_id:
        raise SystemExit("HF_REPO_ID manquant. Utilise --repo-id ou la variable HF_REPO_ID.")
    if not args.token and not args.dry_run:
        raise SystemExit("HF_TOKEN manquant. Utilise --token ou la variable HF_TOKEN.")

    config = load_config(args.config)
    hub_config = build_hub_config(config)
    upload_model_artifacts(
        config=hub_config,
        repo_id=args.repo_id,
        token=args.token,
        dry_run=args.dry_run,
    )
    upload_hub_config(
        config=hub_config,
        output_config_path=args.output_config,
        remote_config_path=args.remote_config_path,
        repo_id=args.repo_id,
        token=args.token,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
