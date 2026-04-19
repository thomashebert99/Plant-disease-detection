"""Log the final ensemble selection to MLflow without retraining models."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
DEFAULT_EXPERIMENT_NAME = "Plant Disease Detection"
DEFAULT_RUN_NAME = "final_ensemble_selection"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "models" / "ensemble_config.json"
DEFAULT_ENSEMBLE_DIR = PROJECT_ROOT / "models" / "ensemble"

REQUIRED_ARTIFACTS = (
    "final_selection_summary.csv",
    "ensemble_evaluation.csv",
    "ensemble_gain_summary.csv",
    "final_decisions.csv",
    "selection_strategy_comparison.csv",
    "selection_summary.csv",
)
GENERATED_ARTIFACT_NAMES = (
    "final_selection_summary.md",
    "plots/test_vs_ood_f1_by_task.png",
    "plots/test_gain_f1_by_task.png",
    "plots/test_f1_ensemble_by_task.png",
    "plots/architecture_family_distribution.png",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a post-hoc MLflow run describing the final ensemble selection. "
            "This script never retrains models."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the summary without creating an MLflow run.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="MLflow run name.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to models/ensemble_config.json.",
    )
    parser.add_argument(
        "--ensemble-dir",
        type=Path,
        default=DEFAULT_ENSEMBLE_DIR,
        help="Path to models/ensemble.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    summary = build_summary(
        config_path=resolve_path(args.config_path),
        ensemble_dir=resolve_path(args.ensemble_dir),
    )

    if args.dry_run:
        print_dry_run(summary)
        return

    log_to_mlflow(
        summary=summary,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )


def resolve_path(path: Path) -> Path:
    """Resolve a path from the project root when needed."""

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_summary(*, config_path: Path, ensemble_dir: Path) -> dict[str, Any]:
    """Build params, metrics and artifacts to log."""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration introuvable: {config_path}")
    if not ensemble_dir.exists():
        raise FileNotFoundError(f"Dossier ensemble introuvable: {ensemble_dir}")

    artifact_paths = collect_artifacts(config_path=config_path, ensemble_dir=ensemble_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = config.get("tasks", {})
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Configuration sans section tasks exploitable.")

    model_counts = {
        task_name: len(task_payload.get("models", []))
        for task_name, task_payload in tasks.items()
    }
    class_counts = {
        task_name: len(task_payload.get("class_names", []))
        for task_name, task_payload in tasks.items()
    }

    params: dict[str, str | int | float | bool] = {
        "selection_strategy": str(config.get("selection_policy") or "top3_max2_family"),
        "ensemble_strategy": "soft_vote_mean_probabilities",
        "num_tasks": len(tasks),
        "total_checkpoints": sum(model_counts.values()),
        "models_per_task_min": min(model_counts.values()),
        "models_per_task_max": max(model_counts.values()),
        "artifact_store": "Hugging Face Hub",
        "model_format": "keras",
        "retraining_performed": False,
        "source": "post_hoc_final_selection",
    }
    for task_name in sorted(tasks):
        task_payload = tasks[task_name]
        architectures = [
            str(model.get("architecture"))
            for model in task_payload.get("models", [])
            if model.get("architecture")
        ]
        params[f"task_{task_name}_model_count"] = model_counts[task_name]
        params[f"task_{task_name}_class_count"] = class_counts[task_name]
        params[f"task_{task_name}_architectures"] = ", ".join(architectures)

    metrics = build_metrics(ensemble_dir)
    tags = build_tags()
    return {
        "config_path": config_path,
        "ensemble_dir": ensemble_dir,
        "artifact_paths": artifact_paths,
        "params": params,
        "metrics": metrics,
        "tags": tags,
    }


def collect_artifacts(*, config_path: Path, ensemble_dir: Path) -> list[Path]:
    """Return existing artifacts that should be attached to the MLflow run."""

    artifacts = [config_path]
    hf_config = ensemble_dir / "ensemble_config_hf.json"
    if hf_config.exists():
        artifacts.append(hf_config)

    missing = []
    for filename in REQUIRED_ARTIFACTS:
        path = ensemble_dir / filename
        if path.exists():
            artifacts.append(path)
        else:
            missing.append(filename)

    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Artefacts de selection manquants: {missing_text}")

    evaluations_dir = ensemble_dir / "evaluations"
    if evaluations_dir.exists():
        artifacts.append(evaluations_dir)

    return artifacts


def build_metrics(ensemble_dir: Path) -> dict[str, float]:
    """Compute compact global metrics from existing CSV outputs."""

    metrics: dict[str, float] = {}

    evaluation_path = ensemble_dir / "ensemble_evaluation.csv"
    evaluation = pd.read_csv(evaluation_path)
    for dataset in ("test", "ood"):
        ensemble_rows = evaluation[
            (evaluation["dataset"] == dataset)
            & (evaluation["model_type"] == "ensemble_soft_vote")
        ]
        if ensemble_rows.empty:
            continue
        for column in ("accuracy", "balanced_accuracy", "f1_macro", "log_loss", "ms_per_image"):
            add_metric(metrics, f"{dataset}_ensemble_{column}_mean", ensemble_rows[column].mean())
            add_metric(metrics, f"{dataset}_ensemble_{column}_min", ensemble_rows[column].min())
            add_metric(metrics, f"{dataset}_ensemble_{column}_max", ensemble_rows[column].max())
        for _, row in ensemble_rows.iterrows():
            task = normalize_metric_key(row["task"])
            add_metric(metrics, f"{dataset}_ensemble_f1_macro_{task}", row["f1_macro"])
            add_metric(metrics, f"{dataset}_ensemble_accuracy_{task}", row["accuracy"])
            add_metric(metrics, f"{dataset}_ensemble_log_loss_{task}", row["log_loss"])
            add_metric(metrics, f"{dataset}_ensemble_ms_per_image_{task}", row["ms_per_image"])

    gain_path = ensemble_dir / "ensemble_gain_summary.csv"
    gains = pd.read_csv(gain_path)
    for dataset in ("test", "ood"):
        gain_rows = gains[gains["dataset"] == dataset]
        if gain_rows.empty:
            continue
        for column in ("gain_f1_macro", "gain_accuracy", "gain_log_loss"):
            add_metric(metrics, f"{dataset}_{column}_mean", gain_rows[column].mean())
            add_metric(metrics, f"{dataset}_{column}_min", gain_rows[column].min())
            add_metric(metrics, f"{dataset}_{column}_max", gain_rows[column].max())
        for _, row in gain_rows.iterrows():
            task = normalize_metric_key(row["task"])
            add_metric(metrics, f"{dataset}_gain_f1_macro_{task}", row["gain_f1_macro"])
            add_metric(metrics, f"{dataset}_gain_accuracy_{task}", row["gain_accuracy"])
            add_metric(metrics, f"{dataset}_gain_log_loss_{task}", row["gain_log_loss"])

    decisions_path = ensemble_dir / "final_decisions.csv"
    decisions = pd.read_csv(decisions_path)
    add_metric(metrics, "final_tasks_count", len(decisions))
    add_metric(metrics, "final_model_count_mean", decisions["model_count"].mean())
    add_metric(metrics, "final_gain_f1_macro_test_mean", decisions["gain_f1_macro_test"].mean())
    add_metric(metrics, "final_gain_log_loss_test_mean", decisions["gain_log_loss_test"].mean())

    selection_path = ensemble_dir / "final_selection_summary.csv"
    selection = pd.read_csv(selection_path)
    if "architecture_family" in selection.columns:
        for family, count in selection["architecture_family"].value_counts().items():
            family_key = normalize_metric_key(family)
            add_metric(metrics, f"selected_architecture_family_count_{family_key}", count)

    return metrics


def build_tags() -> dict[str, str]:
    """Return descriptive MLflow tags for filtering and oral defense."""

    return {
        "project": "plant-disease-detection",
        "stage": "final_selection",
        "retraining": "false",
        "tracking_scope": "post_hoc_decision_trace",
        "selection_strategy": "top3_max2_family",
        "model_source": "keras_checkpoints",
        "artifact_store": "hugging_face_hub",
        "train_dataset": "PlantVillage",
        "ood_dataset": "PlantDoc",
        "deployment_api": "FastAPI",
        "deployment_frontend": "Streamlit",
        "service_monitoring": "JSONL",
        "mlflow_scope": "experiment_tracking",
    }


def add_metric(metrics: dict[str, float], name: str, value: Any) -> None:
    """Add a finite metric value."""

    if pd.isna(value):
        return
    metrics[name] = float(value)


def normalize_metric_key(value: Any) -> str:
    """Return a metric-safe lowercase key."""

    key = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip().lower())
    return key.strip("_") or "unknown"


def print_dry_run(summary: dict[str, Any]) -> None:
    """Print what would be logged to MLflow."""

    print("Dry run: aucun run MLflow ne sera cree.")
    print(f"Config: {summary['config_path']}")
    print(f"Dossier ensemble: {summary['ensemble_dir']}")
    print("\nParametres:")
    for key, value in sorted(summary["params"].items()):
        print(f"  {key}: {value}")
    print("\nMetriques:")
    for key, value in sorted(summary["metrics"].items()):
        print(f"  {key}: {value:.6f}")
    print("\nTags:")
    for key, value in sorted(summary["tags"].items()):
        print(f"  {key}: {value}")
    print("\nArtefacts:")
    for path in summary["artifact_paths"]:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print("\nArtefacts generes:")
    for name in GENERATED_ARTIFACT_NAMES:
        print(f"  {name}")


def write_generated_artifacts(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    """Write Markdown and PNG artifacts for a readable MLflow run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ensemble_dir = summary["ensemble_dir"]
    evaluation = pd.read_csv(ensemble_dir / "ensemble_evaluation.csv")
    gains = pd.read_csv(ensemble_dir / "ensemble_gain_summary.csv")
    selection = pd.read_csv(ensemble_dir / "final_selection_summary.csv")
    decisions = pd.read_csv(ensemble_dir / "final_decisions.csv")

    artifact_paths = [
        write_markdown_summary(
            output_dir / "final_selection_summary.md",
            summary=summary,
            evaluation=evaluation,
            gains=gains,
            decisions=decisions,
        )
    ]
    artifact_paths.extend(
        [
            plot_test_vs_ood_f1(evaluation, plots_dir / "test_vs_ood_f1_by_task.png"),
            plot_test_gain_f1(gains, plots_dir / "test_gain_f1_by_task.png"),
            plot_test_f1_by_task(evaluation, plots_dir / "test_f1_ensemble_by_task.png"),
            plot_architecture_families(
                selection,
                plots_dir / "architecture_family_distribution.png",
            ),
        ]
    )
    return artifact_paths


def write_markdown_summary(
    path: Path,
    *,
    summary: dict[str, Any],
    evaluation: pd.DataFrame,
    gains: pd.DataFrame,
    decisions: pd.DataFrame,
) -> Path:
    """Write a human-readable decision summary."""

    test_rows = ensemble_rows(evaluation, "test")
    ood_rows = ensemble_rows(evaluation, "ood")
    test_gains = gains[gains["dataset"] == "test"]
    ood_gains = gains[gains["dataset"] == "ood"]

    lines = [
        "# Final Ensemble Selection",
        "",
        "This MLflow run summarizes the final model-selection decision without retraining.",
        "",
        "## Selection",
        "",
        f"- Strategy: `{summary['params']['selection_strategy']}`",
        f"- Ensemble strategy: `{summary['params']['ensemble_strategy']}`",
        f"- Tasks: `{summary['params']['num_tasks']}`",
        f"- Total checkpoints: `{summary['params']['total_checkpoints']}`",
        f"- Models per task: `{summary['params']['models_per_task_min']}`",
        f"- Retraining performed: `{summary['params']['retraining_performed']}`",
        "",
        "## Global Metrics",
        "",
        "| Scope | F1 macro mean | F1 macro min | Accuracy mean | Log loss mean |",
        "|---|---:|---:|---:|---:|",
        metric_row("PlantVillage test", test_rows),
        metric_row("PlantDoc OOD", ood_rows),
        "",
        "## Soft-Vote Gain",
        "",
        "| Scope | Mean F1 gain | Min F1 gain | Max F1 gain |",
        "|---|---:|---:|---:|",
        gain_row("PlantVillage test", test_gains),
        gain_row("PlantDoc OOD", ood_gains),
        "",
        "## Final Tasks",
        "",
        "| Task | Models | F1 gain on test | Log-loss gain on test |",
        "|---|---:|---:|---:|",
    ]

    for _, row in decisions.sort_values("task").iterrows():
        lines.append(
            "| {task} | {model_count:.0f} | {gain_f1:.6f} | {gain_loss:.6f} |".format(
                task=row["task"],
                model_count=float(row["model_count"]),
                gain_f1=float(row["gain_f1_macro_test"]),
                gain_loss=float(row["gain_log_loss_test"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The in-distribution scores are very high on PlantVillage, while the OOD scores on PlantDoc are much lower. This run is therefore a trace of the final prototype decision, not evidence of agronomic certification.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def metric_row(label: str, rows: pd.DataFrame) -> str:
    """Return one Markdown row for global metric summary."""

    if rows.empty:
        return f"| {label} | n/a | n/a | n/a | n/a |"
    return "| {label} | {f1_mean:.6f} | {f1_min:.6f} | {acc_mean:.6f} | {loss_mean:.6f} |".format(
        label=label,
        f1_mean=rows["f1_macro"].mean(),
        f1_min=rows["f1_macro"].min(),
        acc_mean=rows["accuracy"].mean(),
        loss_mean=rows["log_loss"].mean(),
    )


def gain_row(label: str, rows: pd.DataFrame) -> str:
    """Return one Markdown row for soft-vote gains."""

    if rows.empty:
        return f"| {label} | n/a | n/a | n/a |"
    return "| {label} | {mean:.6f} | {min:.6f} | {max:.6f} |".format(
        label=label,
        mean=rows["gain_f1_macro"].mean(),
        min=rows["gain_f1_macro"].min(),
        max=rows["gain_f1_macro"].max(),
    )


def ensemble_rows(evaluation: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Return final ensemble rows for one dataset."""

    return evaluation[
        (evaluation["dataset"] == dataset)
        & (evaluation["model_type"] == "ensemble_soft_vote")
    ].copy()


def plot_test_vs_ood_f1(evaluation: pd.DataFrame, path: Path) -> Path:
    """Plot PlantVillage test F1 against PlantDoc OOD F1 by task."""

    rows = ensemble_rows(evaluation, "test")[["task", "f1_macro"]].rename(
        columns={"f1_macro": "PlantVillage test"}
    )
    ood = ensemble_rows(evaluation, "ood")[["task", "f1_macro"]].rename(
        columns={"f1_macro": "PlantDoc OOD"}
    )
    data = rows.merge(ood, on="task", how="outer").sort_values("task")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = data.plot(
        x="task",
        y=["PlantVillage test", "PlantDoc OOD"],
        kind="bar",
        figsize=(11, 5),
        ylim=(0, 1.05),
        rot=35,
    )
    ax.set_title("F1 macro: in-distribution vs out-of-distribution")
    ax.set_xlabel("Task")
    ax.set_ylabel("F1 macro")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_test_gain_f1(gains: pd.DataFrame, path: Path) -> Path:
    """Plot soft-vote F1 gains on the PlantVillage test set."""

    data = gains[gains["dataset"] == "test"].sort_values("task")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in data["gain_f1_macro"]]
    ax.bar(data["task"], data["gain_f1_macro"], color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Soft-vote F1 macro gain on PlantVillage test")
    ax.set_xlabel("Task")
    ax.set_ylabel("F1 macro gain")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_test_f1_by_task(evaluation: pd.DataFrame, path: Path) -> Path:
    """Plot final ensemble test F1 by task."""

    data = ensemble_rows(evaluation, "test").sort_values("f1_macro", ascending=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(data["task"], data["f1_macro"], color="#1f77b4")
    ax.set_xlim(0.98, 1.001)
    ax.set_title("Final ensemble F1 macro on PlantVillage test")
    ax.set_xlabel("F1 macro")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_architecture_families(selection: pd.DataFrame, path: Path) -> Path:
    """Plot selected architecture families."""

    column = "architecture_family" if "architecture_family" in selection.columns else "architecture"
    counts = selection[column].value_counts().sort_values(ascending=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(counts.index, counts.values, color="#9467bd")
    ax.set_title("Selected model families across final checkpoints")
    ax.set_xlabel("Number of selected checkpoints")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def log_to_mlflow(
    *,
    summary: dict[str, Any],
    experiment_name: str,
    run_name: str,
) -> None:
    """Create the MLflow run and attach params, metrics and artifacts."""

    import mlflow

    from src.core.mlflow_config import setup_mlflow

    setup_mlflow(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(summary["tags"])
        mlflow.log_params(summary["params"])
        mlflow.log_metrics(summary["metrics"])

        for path in summary["artifact_paths"]:
            if path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path=path.name)
            else:
                mlflow.log_artifact(str(path), artifact_path="final_selection")

        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_dir = Path(tmp_dir) / "generated"
            write_generated_artifacts(summary, generated_dir)
            mlflow.log_artifacts(str(generated_dir), artifact_path="final_selection_visuals")

    print(f"Run MLflow cree: {run_name}")


if __name__ == "__main__":
    main()
