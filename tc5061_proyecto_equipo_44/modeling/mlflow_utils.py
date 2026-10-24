"""Utils used for MLflow administration in code."""

import json
import os
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from loguru import logger



MLFLOW_EXPERIMENT_NAME = "mibici-equipo44"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

TEAM_TAG = "equipo44"


def configure_mlflow() -> None:
    """Configure local MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # MLflow Projects may provide an active run/experiment. Reuse it when present.
    if not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("MLflow URI: {}", mlflow.get_tracking_uri())
    logger.info("MLflow experiment: {}", MLFLOW_EXPERIMENT_NAME)


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


def log_mlflow_run(
    run_name: str,
    params: dict,
    metrics: dict,
    tags: dict,
    artifact_paths: list | None = None,
) -> None:
    """Log parameters, metrics, tags and existing artifacts to MLflow."""
    commit = get_git_commit()

    active_run = mlflow.active_run()
    projects_run_id = os.getenv("MLFLOW_RUN_ID")
    if active_run is not None or projects_run_id:
        # Create one child run per selected model under the MLflow Projects run.
        with mlflow.start_run(run_name=run_name, nested=True):
            _log_run_contents(run_name, params, metrics, tags, artifact_paths, commit)
    else:
        with mlflow.start_run(run_name=run_name):
            _log_run_contents(run_name, params, metrics, tags, artifact_paths, commit)


def _log_run_contents(
    run_name: str,
    params: dict,
    metrics: dict,
    tags: dict,
    artifact_paths: list | None,
    commit: str,
) -> None:
    safe_params = {**params, "git_commit": commit}
    for key, value in safe_params.items():
        if isinstance(value, (list, tuple, dict)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        mlflow.log_param(key, value)

    safe_tags = {
        **tags,
        "team": TEAM_TAG,
        "project": "MiBici",
        "git_commit": commit,
        "model_run_name": run_name,
    }
    mlflow.set_tags({key: str(value) for key, value in safe_tags.items()})

    numeric_metrics = {}
    for key, value in metrics.items():
        if (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
        ):
            metric_name = key if key.startswith(("test_", "data_")) else f"test_{key}"
            numeric_metrics[metric_name] = float(value)
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)

    for artifact in artifact_paths or []:
        artifact_path = Path(artifact)
        if artifact_path.is_file():
            mlflow.log_artifact(str(artifact_path))
        else:
            logger.warning("Artifact not found; not logged: {}", artifact_path)
    logger.info("MLflow results logged for: {}", run_name)