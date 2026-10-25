"""Utilities for MLflow tracking and artifact logging."""

import json
import os
import subprocess
from pathlib import Path as FilePath
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

MLFLOW_EXPERIMENT_NAME = "mibici-equipo44"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
TEAM_TAG = "equipo44"


def configure_mlflow() -> None:
    """Configure the MLflow tracking URI and experiment when starting a run locally."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # MLflow Projects may already have created a run in the selected experiment.
    if not os.getenv("MLFLOW_RUN_ID") and mlflow.active_run() is None:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("MLflow URI: {}", mlflow.get_tracking_uri())
    logger.info("Configured MLflow experiment: {}", MLFLOW_EXPERIMENT_NAME)


def get_git_commit() -> str:
    """Return the current Git commit, or 'unknown' outside a Git repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _safe_value(value: Any) -> Any:
    """Convert complex values to strings that MLflow parameters can store."""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def log_model_with_signature(model: Any,input_example: pd.DataFrame | np.ndarray,artifact_path: str = "model",) -> None:
    """Log a fitted scikit-learn model with inferred signature and input example."""
    import mlflow.sklearn
    from mlflow.models import infer_signature

    example = input_example
    if isinstance(example, pd.DataFrame):
        example = example.head(5).copy()
    else:
        example = np.asarray(example)[:5]

    predictions = model.predict(example)
    signature = infer_signature(example, predictions)

    # artifact_path is compatible with a broad range of MLflow versions.
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=example,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )


def log_mlflow_run(
    run_name: str,
    params: dict,
    metrics: dict,
    tags: dict,
    artifact_paths: list | None = None,
    nested: bool | None = None,
    model: Any | None = None,
    input_example: pd.DataFrame | np.ndarray | None = None,
    artifact_dirs: dict[str, list[str | FilePath]] | None = None,
    input_dataset: Any | None = None,
) -> None:
    """Log parameters, metrics, tags, files, and optionally a model to MLflow.

    `artifact_paths` remains supported for existing callers; those files are logged
    under `artifacts/`. Use `artifact_dirs` to place files under named folders,
    e.g. {"plots": [plot_path], "data": [manifest_path], "config": [config_path]}.
    """
    commit = get_git_commit()

    if nested is None:
        nested = (
            mlflow.active_run() is not None
            or bool(os.getenv("MLFLOW_RUN_ID"))
        )

    run_context = (
        mlflow.start_run(run_name=run_name, nested=True)
        if nested
        else mlflow.start_run(run_name=run_name)
    )

    with run_context:
        if input_dataset is not None:
            mlflow.log_input(input_dataset, context="training")

        safe_params = {**params, "git_commit": commit}
        for key, value in safe_params.items():
            mlflow.log_param(key, _safe_value(value))

        safe_tags = {**tags,"team": TEAM_TAG,"project": "MiBici","git_commit": commit,"model_run_name": run_name,}
        mlflow.set_tags({key: str(value) for key, value in safe_tags.items()})

        numeric_metrics = {}
        for key, value in metrics.items():
            if (
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                and np.isfinite(value)
            ):
                metric_name = (
                    key if key.startswith(("test_", "data_")) else f"test_{key}"
                )
                numeric_metrics[metric_name] = float(value)
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

        for artifact in artifact_paths or []:
            _log_existing_file(artifact, "artifacts")

        for destination, paths in (artifact_dirs or {}).items():
            for artifact in paths:
                _log_existing_file(artifact, destination)

        if model is not None:
            if input_example is None:
                logger.warning(
                    "Model was not logged with MLflow flavor because input_example "
                    "was not supplied for run '{}'.",
                    run_name,
                )
            else:
                log_model_with_signature(model, input_example, artifact_path="model")

        logger.info("MLflow results logged for: {}", run_name)


def _log_existing_file(path: str | FilePath, artifact_path: str) -> None:
    """Log an existing file into an MLflow artifact subdirectory."""
    file_path = FilePath(path)
    if file_path.is_file():
        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
    else:
        logger.warning("Artifact not found; not logged: {}", file_path)
