"""Train P2 and track in MLflow."""


from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tc5061_proyecto_equipo_44.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

from .data_utils import chronological_split, dataset_metadata
from .mlflow_utils import log_mlflow_run



RANDOM_STATE = 0
TEST_SIZE = 0.30
N_CLUSTERS = 3

HOURLY_PATH = PROCESSED_DATA_DIR / "hourly_features.parquet"
PROFILES_PATH = PROCESSED_DATA_DIR / "station_profiles.parquet"



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