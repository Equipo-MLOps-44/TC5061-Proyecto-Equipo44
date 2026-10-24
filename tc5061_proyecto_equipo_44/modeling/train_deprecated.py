"""Train baseline models for P1, P2 and P3 and track them in MLflow."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tc5061_proyecto_equipo_44.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

MLFLOW_EXPERIMENT_NAME = "mibici-equipo44"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

RANDOM_STATE = 0
TEST_SIZE = 0.30
N_CLUSTERS = 3
TEAM_TAG = "equipo44"

HOURLY_PATH = PROCESSED_DATA_DIR / "hourly_features.parquet"
PROFILES_PATH = PROCESSED_DATA_DIR / "station_profiles.parquet"


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


def dataset_metadata(path: Path) -> dict:
    """Collect basic metadata about the input dataset."""
    stat = path.stat()

    return {
        "dataset_file": path.name,
        "dataset_size_bytes": stat.st_size,
        "dataset_modified_time": pd.Timestamp(
            stat.st_mtime, unit="s"
        ).isoformat(),
    }


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


def chronological_split(
    df: pd.DataFrame,
    timestamp_column: str = "hour",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split records chronologically, keeping timestamps in one partition."""
    unique_times = np.sort(df[timestamp_column].dropna().unique())

    if len(unique_times) < 2:
        raise ValueError(
            "At least two distinct timestamps are required."
        )

    split_index = int(len(unique_times) * (1 - TEST_SIZE))
    split_index = min(max(split_index, 1), len(unique_times) - 1)
    split_time = pd.Timestamp(unique_times[split_index])

    train = df[df[timestamp_column] < split_time].copy()
    test = df[df[timestamp_column] >= split_time].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Chronological split produced an empty partition."
        )

    return train, test, split_time


def train_p1(model_name: str = "baseline", n_estimators: int = 300, max_depth: int = 15, random_state: int = 0) -> dict:
    """
    P1: estimate hourly station departures using a mean baseline.

    The planned feature matrices are prepared for the future predictive
    model, but the DummyRegressor baseline does not use those features.
    """
    p1_features = [
        "Origen_Id",
        "departures_lag_1h",
        "departures_lag_24h",
        "departures_lag_168h",
        "departures_rolling_mean_24h",
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    df = pd.read_parquet(HOURLY_PATH)

    required_columns = {"Origen_Id", "hour", "departures"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"P1 is missing required columns: {sorted(missing_columns)}"
        )

    missing_features = [
        column for column in p1_features if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"P1 is missing planned features: {missing_features}"
        )

    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df["departures"] = pd.to_numeric(
        df["departures"], errors="coerce"
    )
    df = df.dropna(
        subset=["Origen_Id", "hour", "departures"]
    )
    df = df.sort_values(
        ["hour", "Origen_Id"]
    ).reset_index(drop=True)

    train, test, split_time = chronological_split(df)

    y_train = train["departures"].to_numpy()
    y_test = test["departures"].to_numpy()

    # Prepare feature matrices for the future predictive model.
    X_train = train[p1_features].copy()
    X_test = test[p1_features].copy()

    # Station ID is categorical.
    X_train["Origen_Id"] = X_train["Origen_Id"].astype("string")
    X_test["Origen_Id"] = X_test["Origen_Id"].astype("string")

    numeric_features = [
        feature for feature in p1_features
        if feature != "Origen_Id"
    ]

    for feature in numeric_features:
        X_train[feature] = pd.to_numeric(
            X_train[feature], errors="coerce"
        )
        X_test[feature] = pd.to_numeric(
            X_test[feature], errors="coerce"
        )

    logger.info(
        "P1 feature matrices: X_train={}, X_test={}",
        X_train.shape,
        X_test.shape,
    )

    # Exclude station ID from RF inputs to avoid treating IDs as continuous values.
    numeric_features = [feature for feature in p1_features if feature != "Origen_Id"]
    X_train_model = X_train[numeric_features].apply(pd.to_numeric, errors="coerce")
    X_test_model = X_test[numeric_features].apply(pd.to_numeric, errors="coerce")
    medians = X_train_model.median().fillna(0)
    X_train_model = X_train_model.fillna(medians)
    X_test_model = X_test_model.fillna(medians)

    if model_name == "baseline":
        model = DummyRegressor(strategy="mean")
        model.fit(np.zeros((len(y_train), 1)), y_train)
        predictions = model.predict(np.zeros((len(y_test), 1)))
        model_filename = "p1_mean_baseline.joblib"
        model_type = "mean_baseline"
    elif model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1,
        )
        model.fit(X_train_model, y_train)
        predictions = model.predict(X_test_model)
        model_filename = "p1_random_forest.joblib"
        model_type = "random_forest_regressor"
    else:
        raise ValueError("P1 model must be 'baseline' or 'rf'.")

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / model_filename
    joblib.dump(model, model_path)

    results = {
        "task": "P1_regression",
        "target": "departures",
        "model": model_name,
        "split": "chronological",
        "split_time": str(split_time),
        "test_size": TEST_SIZE,
        "train_rows": len(train),
        "test_rows": len(test),
        "planned_model_features": p1_features,
        "feature_matrices_prepared": True,
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "mae": mae,
        "rmse": rmse,
    }

    log_mlflow_run(
        run_name=f"P1_{model_name}",
        params={
            "task": "P1_regression",
            "target": "departures",
            "model": model_name,
            "n_estimators": n_estimators if model_name == "rf" else "not_applicable",
            "max_depth": max_depth if model_name == "rf" else "not_applicable",
            "random_state": random_state,
            "test_size": TEST_SIZE,
            "split": "chronological",
            "split_time": str(split_time),
            "train_rows": len(train),
            "test_rows": len(test),
            "planned_features": p1_features,
            "feature_matrices_prepared": True,
            "station_id_treated_as": "categorical",
            "data_start_time": str(df["hour"].min()),
            "data_end_time": str(df["hour"].max()),
            **dataset_metadata(HOURLY_PATH),
        },
        metrics={
            "mae": mae,
            "rmse": rmse,
        },
        tags={
            "problem": "P1",
            "question": "demand_prediction",
            "model_type": model_type,
        },
        artifact_paths=[model_path],
    )

    logger.success("P1 {} completed.", model_name)
    return results


def train_p2(model_name: str = "baseline", n_estimators: int = 300, max_depth: int = 15, class_weight: str = "balanced", random_state: int = 0) -> dict | None:
    """
    P2: majority-class baselines for station risk indicators.

    Risk labels are derived from estimated inventory in feature engineering.
    They are proxies, not measurements of actual station inventory.
    """
    df = pd.read_parquet(HOURLY_PATH)

    risk_targets = ["risk_no_bikes", "risk_no_docks"]
    available_targets = [
        target for target in risk_targets
        if target in df.columns
    ]

    if not available_targets:
        logger.warning(
            "P2 skipped: risk targets are not present in {}. "
            "Run feature engineering first.",
            HOURLY_PATH,
        )
        return None

    candidate_features = [
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "departures_lag_1h",
        "departures_lag_24h",
        "departures_lag_168h",
        "departures_rolling_mean_24h",
        "arrivals_lag_1h",
        "net_flow_lag_1h",
        "dpcapacity",
    ]

    feature_columns = [
        column for column in candidate_features
        if column in df.columns
    ]

    if not feature_columns:
        raise ValueError("P2 has no eligible predictor columns.")

    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])
    df = df.sort_values(
        ["hour", "Origen_Id"]
    ).reset_index(drop=True)

    train_all, test_all, split_time = chronological_split(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for target in available_targets:
        train = train_all.dropna(subset=[target]).copy()
        test = test_all.dropna(subset=[target]).copy()

        if train.empty or test.empty:
            logger.warning(
                "P2 target '{}' skipped because a split is empty.",
                target,
            )
            continue

        if train[target].nunique() < 2:
            logger.warning(
                "P2 target '{}' skipped: training data contains "
                "fewer than two classes.",
                target,
            )
            continue

        X_train = train[feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        X_test = test[feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )

        y_train = train[target].astype(int)
        y_test = test[target].astype(int)

        # Learn imputation values from training data only.
        medians = X_train.median().fillna(0)
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        if model_name == "baseline":
            model = DummyClassifier(strategy="most_frequent", random_state=random_state)
            model_type = "majority_class_baseline"
        elif model_name == "rf":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=None if class_weight.lower() in {"none", "null", ""} else class_weight,
                random_state=random_state,
                n_jobs=-1,
            )
            model_type = "random_forest_classifier"
        else:
            raise ValueError("P2 model must be 'baseline' or 'rf'.")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        labels = sorted(
            set(y_train.unique()).union(set(y_test.unique()))
        )

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
        macro_recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        metrics = {
            "task": "P2_classification",
            "target": target,
            "model_type": "majority_class_baseline",
            "features": feature_columns,
            "split": "chronological_by_timestamp",
            "split_time": str(split_time),
            "test_size": TEST_SIZE,
            "train_rows": len(train),
            "test_rows": len(test),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "macro_recall": macro_recall,
            "labels": [int(label) for label in labels],
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=labels
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=labels,
                output_dict=True,
                zero_division=0,
            ),
        }

        if target == "risk_no_bikes":
            model_filename = f"p2_no_bikes_{model_name}.joblib"
        else:
            model_filename = f"p2_no_docks_{model_name}.joblib"

        model_path = MODELS_DIR / model_filename

        joblib.dump(
            {
                "model": model,
                "features": feature_columns,
                "medians": medians,
            },
            model_path,
        )

        log_mlflow_run(
            run_name=f"P2_{target}_{model_name}",
            params={
                "task": "P2_classification",
                "target": target,
                "model": model_name,
                "n_estimators": n_estimators if model_name == "rf" else "not_applicable",
                "max_depth": max_depth if model_name == "rf" else "not_applicable",
                "class_weight": class_weight if model_name == "rf" else "not_applicable",
                "random_state": random_state,
                "test_size": TEST_SIZE,
                "split": "chronological_by_timestamp",
                "split_time": str(split_time),
                "train_rows": len(train),
                "test_rows": len(test),
                "features": feature_columns,
                "data_start_time": str(df["hour"].min()),
                "data_end_time": str(df["hour"].max()),
                **dataset_metadata(HOURLY_PATH),
            },
            metrics={
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "macro_recall": macro_recall,
            },
            tags={
                "problem": "P2",
                "question": target,
                "model_type": model_type,
            },
            artifact_paths=[model_path],
        )

        results[target] = metrics
        logger.success(
            "P2 majority baseline completed for '{}'.",
            target,
        )

    return results or None


def train_p3(n_clusters: int = 3, random_state: int = 0, n_init: int = 10) -> dict:
    """P3: cluster station profiles using KMeans with three clusters."""
    df = pd.read_parquet(PROFILES_PATH)

    excluded_columns = {"Origen_Id", "id"}
    feature_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError(
            "P3 has no numeric station-profile features."
        )

    X = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    X = X.fillna(X.median()).fillna(0)

    if len(X) <= n_clusters:
        raise ValueError(f"P3 needs more than {n_clusters} station records.")

    if len(X.drop_duplicates()) < n_clusters:
        raise ValueError("P3 has fewer distinct profiles than clusters.")

    model = make_pipeline(
        StandardScaler(),
        KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        ),
    )

    labels = model.fit_predict(X)
    scaled_X = model.named_steps["standardscaler"].transform(X)

    silhouette = silhouette_score(scaled_X, labels)

    cluster_sizes = {
        str(int(cluster)): int(count)
        for cluster, count in (
            pd.Series(labels).value_counts().sort_index().items()
        )
    }

    results = {
        "task": "P3_clustering",
        "features": feature_columns,
        "n_clusters": n_clusters,
        "stations": len(X),
        "silhouette_score": silhouette,
        "cluster_sizes": cluster_sizes,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"p3_kmeans_{n_clusters}_clusters.joblib"
    clusters_path = REPORTS_DIR / f"p3_station_clusters_{n_clusters}.csv"

    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
        },
        model_path,
    )

    output = df.copy()
    output["cluster"] = labels
    output.to_csv(clusters_path, index=False)

    log_mlflow_run(
        run_name=f"P3_kmeans_{n_clusters}_clusters",
        params={
            "task": "P3_clustering",
            "algorithm": "KMeans",
            "n_clusters": n_clusters,
            "random_state": random_state,
            "n_init": n_init,
            "features": feature_columns,
            "stations": len(X),
            **dataset_metadata(PROFILES_PATH),
        },
        metrics={
            "data_silhouette_score": silhouette,
            "data_stations": len(X),
        },
        tags={
            "problem": "P3",
            "question": "station_clustering",
            "model_type": "kmeans_baseline",
        },
        artifact_paths=[
            model_path,
            clusters_path,
        ],
    )

    logger.success("P3 KMeans with {} clusters completed.", n_clusters)
    return results


def main() -> None:
    """Train only the selected question/model and track it in MLflow."""
    parser = argparse.ArgumentParser(description="Train a selected MiBici model.")
    parser.add_argument("--question", choices=["p1", "p2", "p3"], default="p1")
    parser.add_argument("--model", choices=["baseline", "rf", "kmeans"], default="baseline")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=10)
    args = parser.parse_args()

    allowed = {"p1": {"baseline", "rf"}, "p2": {"baseline", "rf"}, "p3": {"kmeans"}}
    if args.model not in allowed[args.question]:
        parser.error(f"model={args.model} is not valid for question={args.question}; choose {sorted(allowed[args.question])}.")
    if args.n_estimators < 1 or args.max_depth < 1 or args.n_clusters < 2 or args.n_init < 1:
        parser.error("n_estimators, max_depth, n_init must be positive and n_clusters must be at least 2.")

    configure_mlflow()
    required_path = PROFILES_PATH if args.question == "p3" else HOURLY_PATH
    if not required_path.exists():
        raise FileNotFoundError(f"Required feature dataset not found: {required_path}")

    if args.question == "p1":
        result = train_p1(args.model, args.n_estimators, args.max_depth, args.random_state)
    elif args.question == "p2":
        result = train_p2(args.model, args.n_estimators, args.max_depth, args.class_weight, args.random_state)
    else:
        result = train_p3(args.n_clusters, args.random_state, args.n_init)

    logger.info("Training completed: question={}, model={}", args.question, args.model)
    logger.info("Result summary: {}", result)


if __name__ == "__main__":
    main()