"""Train baseline models for P1, P2 and P3."""

import json
import warnings

import joblib
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

RANDOM_STATE = 0
TEST_SIZE = 0.30
N_CLUSTERS = 3

HOURLY_PATH = PROCESSED_DATA_DIR / "hourly_features.parquet"
PROFILES_PATH = PROCESSED_DATA_DIR / "station_profiles.parquet"

# P2 target columns are created by feature engineering.


def save_metrics(metrics: dict, filename: str) -> None:
    """Save experiment metrics as JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    path = REPORTS_DIR / filename
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False, default=float)

    logger.info("Metrics saved to {}", path)


def save_metrics(metrics: dict, filename: str) -> None:
    """Save experiment metrics as JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    path = REPORTS_DIR / filename
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False, default=float)

    logger.info("Metrics saved to {}", path)


def train_p1() -> dict:
    """Prepare P1 feature matrices and evaluate baseline references."""
    P1_FEATURES = [
        "Origen_Id",  # Identificador categórico de la estación
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

    required = {"Origen_Id", "hour", "departures"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"P1 is missing columns: {sorted(missing)}")

    # Validate that all planned features exist before preparing X_train/X_test.
    available_p1_features = [col for col in P1_FEATURES if col in df.columns]
    missing_p1_features = [col for col in P1_FEATURES if col not in df.columns]
    if missing_p1_features:
        raise ValueError(f"P1 is missing features: {missing_p1_features}")
    logger.info("P1 features available: {}", available_p1_features)

    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df["departures"] = pd.to_numeric(
        df["departures"], errors="coerce"
    )
    df = df.dropna(subset=["Origen_Id", "hour", "departures"])
    df = df.sort_values(["hour", "Origen_Id"]).reset_index(drop=True)

    # Split chronologically: never train on observations after the test period.
    unique_hours = np.sort(df["hour"].unique())
    if len(unique_hours) < 2:
        raise ValueError("P1 requires at least two distinct timestamps.")

    split_index = int(len(unique_hours) * (1 - TEST_SIZE))
    split_index = min(max(split_index, 1), len(unique_hours) - 1)
    split_time = unique_hours[split_index]

    train = df[df["hour"] < split_time].copy()
    test = df[df["hour"] >= split_time].copy()

    if train.empty or test.empty:
        raise ValueError("P1 chronological split produced an empty partition.")
    
    y_train = train["departures"].to_numpy()
    y_test = test["departures"].to_numpy()

    # Prepare the planned P1 feature matrices using the same chronological split.
    # These matrices are ready for a future predictive estimator; the baselines
    # below intentionally do not consume them.
    X_train = train[P1_FEATURES].copy()
    X_test = test[P1_FEATURES].copy()

    # Station ID is categorical, not a continuous numeric measurement.
    X_train["Origen_Id"] = X_train["Origen_Id"].astype("string")
    X_test["Origen_Id"] = X_test["Origen_Id"].astype("string")

    numeric_features = [col for col in P1_FEATURES if col != "Origen_Id"]
    for col in numeric_features:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    logger.info(
        "P1 feature matrices prepared: X_train={}, X_test={}, features={}",
        X_train.shape,
        X_test.shape,
        P1_FEATURES,
    )

    # Baseline 1: predict the training-set mean for every observation.
    # The DummyRegressor receives a constant placeholder, not P1_FEATURES,
    # because it is a reference baseline rather than the future feature model.
    mean_model = DummyRegressor(strategy="mean")
    mean_model.fit(np.zeros((len(train), 1)), y_train)
    mean_predictions = mean_model.predict(np.zeros((len(test), 1)))

    metrics = {
        "task": "P1_regression",
        "target": "departures",
        "split": "chronological",
        "test_size": TEST_SIZE,
        "split_time": str(pd.Timestamp(split_time)),
        "train_rows": len(train),
        "test_rows": len(test),
        "model_training_status": "not_trained_yet_baselines_only",
        "planned_model_features": P1_FEATURES,
        "available_planned_model_features": available_p1_features,
        "missing_planned_model_features": missing_p1_features,
        "feature_matrices_prepared": True,
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "station_id_treated_as": "categorical",
        "mean_baseline": {
            "mae": mean_absolute_error(y_test, mean_predictions),
            "rmse": np.sqrt(
                mean_squared_error(y_test, mean_predictions)
            ),
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(mean_model, MODELS_DIR / "p1_mean_baseline.joblib")

    save_metrics(metrics, "p1_baseline_metrics.json")
    logger.success("P1 baseline trained.")

    return metrics


def train_p2() -> dict | None:
    """
    Train majority-class baselines for both station risk indicators.

    The target labels are derived from the estimated inventory created in
    features.py. Predictors use temporal and lagged information only; the
    contemporaneous inventory and risk labels are deliberately excluded.
    """
    df = pd.read_parquet(HOURLY_PATH)

    risk_targets = ["risk_no_bikes", "risk_no_docks"]
    available_targets = [col for col in risk_targets if col in df.columns]

    if not available_targets:
        logger.warning(
            "P2 skipped: neither risk_no_bikes nor risk_no_docks exists in {}. "
            "Run feature engineering first to create the estimated inventory "
            "and risk labels.",
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
    feature_columns = [col for col in candidate_features if col in df.columns]

    if not feature_columns:
        raise ValueError("P2 has no eligible predictor columns.")

    df = df.sort_values(["hour", "Origen_Id"]).reset_index(drop=True)
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"]).copy()

    # Split by timestamp so records from the same hour cannot appear in both
    # train and test partitions.
    unique_hours = np.sort(df["hour"].unique())
    if len(unique_hours) < 2:
        raise ValueError("P2 requires at least two distinct timestamps.")

    split_index = int(len(unique_hours) * (1 - TEST_SIZE))
    split_index = min(max(split_index, 1), len(unique_hours) - 1)
    split_time = unique_hours[split_index]

    train_all = df[df["hour"] < split_time].copy()
    test_all = df[df["hour"] >= split_time].copy()

    results = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for target_column in available_targets:
        train = train_all.dropna(subset=[target_column]).copy()
        test = test_all.dropna(subset=[target_column]).copy()

        if train.empty or test.empty:
            logger.warning("P2 target '{}' skipped: empty train/test split.", target_column)
            continue

        if train[target_column].nunique() < 2:
            logger.warning(
                "P2 target '{}' skipped: training data contains fewer than two classes.",
                target_column,
            )
            continue

        X_train = train[feature_columns].apply(pd.to_numeric, errors="coerce")
        X_test = test[feature_columns].apply(pd.to_numeric, errors="coerce")
        y_train = train[target_column].astype(int)
        y_test = test[target_column].astype(int)

        # Fit imputations on training data only.
        medians = X_train.median().fillna(0)
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        model = DummyClassifier(
            strategy="most_frequent",
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        labels = sorted(pd.concat([y_train, y_test]).unique().tolist())
        metrics = {
            "task": "P2_classification",
            "target": target_column,
            "model_type": "majority_class_baseline",
            "features": feature_columns,
            "split": "chronological_by_timestamp",
            "split_time": str(pd.Timestamp(split_time)),
            "test_size": TEST_SIZE,
            "train_rows": len(train),
            "test_rows": len(test),
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "labels": [int(label) for label in labels],
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=labels
            ).tolist(),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
        }

        model_filename = (
            "p2_no_bikes_dummy_baseline.joblib"
            if target_column == "risk_no_bikes"
            else "p2_no_docks_dummy_baseline.joblib"
        )
        metrics_filename = (
            "p2_no_bikes_baseline_metrics.json"
            if target_column == "risk_no_bikes"
            else "p2_no_docks_baseline_metrics.json"
        )

        joblib.dump(
            {"model": model, "features": feature_columns, "medians": medians},
            MODELS_DIR / model_filename,
        )
        save_metrics(metrics, metrics_filename)
        results[target_column] = metrics
        logger.success("P2 baseline trained for '{}'.", target_column)

    return results or None


def train_p3() -> dict:
    """Cluster station profiles using scaled numeric usage features."""
    df = pd.read_parquet(PROFILES_PATH)

    excluded = {"Origen_Id", "id"}
    feature_columns = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in excluded
    ]

    if not feature_columns:
        raise ValueError("P3 has no numeric station-profile features.")

    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0)

    # KMeans requires enough distinct stations and feature profiles.
    if len(X) <= N_CLUSTERS:
        raise ValueError(
            f"P3 needs more than {N_CLUSTERS} station records."
        )
    if len(X.drop_duplicates()) < N_CLUSTERS:
        raise ValueError("P3 has fewer distinct profiles than clusters.")

    model = make_pipeline(
        StandardScaler(),
        KMeans(
            n_clusters=N_CLUSTERS,
            random_state=RANDOM_STATE,
            n_init=10,
        ),
    )
    labels = model.fit_predict(X)

    metrics = {
        "task": "P3_clustering",
        "features": feature_columns,
        "n_clusters": N_CLUSTERS,
        "stations": len(X),
        "silhouette_score": silhouette_score(
            model.named_steps["standardscaler"].transform(X),
            labels,
        ),
        "cluster_sizes": {
            str(int(cluster)): int(count)
            for cluster, count in
            pd.Series(labels).value_counts().sort_index().items()
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": feature_columns},
        MODELS_DIR / "p3_kmeans_baseline.joblib",
    )

    output = df.copy()
    output["cluster"] = labels
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        REPORTS_DIR / "p3_station_clusters.csv", index=False
    )

    save_metrics(metrics, "p3_baseline_metrics.json")
    logger.success("P3 baseline trained.")

    return metrics


def main() -> None:
    """Train all baselines that have the required data."""
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
    for task, metrics in results.items():
        logger.info(
            "{}: {}",
            task,
            "skipped (target unavailable)" if metrics is None
            else "completed",
        )


if __name__ == "__main__":
    main()