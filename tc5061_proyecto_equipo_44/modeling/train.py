"""Train baseline models for P1, P2 and P3 and track them in MLflow."""

import json
import subprocess
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
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
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

RANDOM_STATE = 0
TEST_SIZE = 0.30
N_CLUSTERS = 3
TEAM_TAG = "equipo44"

HOURLY_PATH = PROCESSED_DATA_DIR / "hourly_features.parquet"
PROFILES_PATH = PROCESSED_DATA_DIR / "station_profiles.parquet"


def configure_mlflow() -> None:
    """Configure local MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
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

    with mlflow.start_run(run_name=run_name):
        # MLflow parameters must be scalar-compatible values.
        safe_params = {**params, "git_commit": commit}

        for key, value in safe_params.items():
            if isinstance(value, (list, tuple, dict)):
                value = json.dumps(
                    value,
                    ensure_ascii=False,
                    default=str,
                )

            mlflow.log_param(key, value)

        safe_tags = {
            **tags,
            "team": TEAM_TAG,
            "project": "MiBici",
            "git_commit": commit,
        }
        mlflow.set_tags(
            {key: str(value) for key, value in safe_tags.items()}
        )

        # P1/P2 metrics are test metrics. P3 uses data_ metrics.
        numeric_metrics = {}

        for key, value in metrics.items():
            if (
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                and np.isfinite(value)
            ):
                metric_name = (
                    key
                    if key.startswith(("test_", "data_"))
                    else f"test_{key}"
                )
                numeric_metrics[metric_name] = float(value)

        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

        # Log only files that exist.
        for artifact in artifact_paths or []:
            artifact_path = Path(artifact)

            if artifact_path.is_file():
                mlflow.log_artifact(str(artifact_path))
            else:
                logger.warning(
                    "Artifact not found; not logged: {}",
                    artifact_path,
                )

        logger.info("MLflow run logged: {}", run_name)


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


def train_p1() -> dict:
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

    # The baseline predicts the training-set mean for every test row.
    # Placeholder input is intentional: this is a reference baseline,
    # not the future feature-based model.
    mean_model = DummyRegressor(strategy="mean")
    mean_model.fit(np.zeros((len(y_train), 1)), y_train)

    predictions = mean_model.predict(
        np.zeros((len(y_test), 1))
    )

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "p1_mean_baseline.joblib"
    joblib.dump(mean_model, model_path)

    results = {
        "task": "P1_regression",
        "target": "departures",
        "baseline": "DummyRegressor(strategy=mean)",
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
        run_name="P1_mean_baseline",
        params={
            "task": "P1_regression",
            "target": "departures",
            "baseline": "DummyRegressor",
            "strategy": "mean",
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
            "model_type": "mean_baseline",
        },
        artifact_paths=[model_path],
    )

    logger.success("P1 mean baseline completed.")
    return results


def train_p2() -> dict | None:
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

        model = DummyClassifier(
            strategy="most_frequent",
            random_state=RANDOM_STATE,
        )
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
            model_filename = "p2_no_bikes_dummy_baseline.joblib"
        else:
            model_filename = "p2_no_docks_dummy_baseline.joblib"

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
            run_name=f"P2_{target}_majority_baseline",
            params={
                "task": "P2_classification",
                "target": target,
                "strategy": "most_frequent",
                "random_state": RANDOM_STATE,
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
                "model_type": "majority_class_baseline",
            },
            artifact_paths=[model_path],
        )

        results[target] = metrics
        logger.success(
            "P2 majority baseline completed for '{}'.",
            target,
        )

    return results or None


def train_p3() -> dict:
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

    if len(X) <= N_CLUSTERS:
        raise ValueError(
            f"P3 needs more than {N_CLUSTERS} station records."
        )

    if len(X.drop_duplicates()) < N_CLUSTERS:
        raise ValueError(
            "P3 has fewer distinct profiles than clusters."
        )

    model = make_pipeline(
        StandardScaler(),
        KMeans(
            n_clusters=N_CLUSTERS,
            random_state=RANDOM_STATE,
            n_init=10,
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
        "n_clusters": N_CLUSTERS,
        "stations": len(X),
        "silhouette_score": silhouette,
        "cluster_sizes": cluster_sizes,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "p3_kmeans_baseline.joblib"
    clusters_path = REPORTS_DIR / "p3_station_clusters.csv"

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
        run_name="P3_kmeans_3_clusters",
        params={
            "task": "P3_clustering",
            "algorithm": "KMeans",
            "n_clusters": N_CLUSTERS,
            "random_state": RANDOM_STATE,
            "n_init": 10,
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

    logger.success("P3 KMeans baseline completed.")
    return results


def main() -> None:
    """Run all baselines and track results in MLflow."""
    configure_mlflow()

    for path in [HOURLY_PATH, PROFILES_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required feature dataset not found: {path}"
            )

    results = {
        "P1": train_p1(),
        "P2": train_p2(),
        "P3": train_p3(),
    }

    logger.info("Baseline training summary:")

    for task, result in results.items():
        status = (
            "skipped (targets unavailable)"
            if result is None
            else "completed"
        )
        logger.info("{}: {}", task, status)


if __name__ == "__main__":
    main()