"""Train model of the P1."""

import json
import os
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

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