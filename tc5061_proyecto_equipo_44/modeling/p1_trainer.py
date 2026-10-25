"""Train P1 regression models and track results in MLflow."""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tc5061_proyecto_equipo_44.config import (MODELS_DIR,PROCESSED_DATA_DIR,REPORTS_DIR,)

from .data_utils import chronological_split, dataset_metadata
from .mlflow_utils import log_mlflow_run


RANDOM_STATE = 0
TEST_SIZE = 0.30
HOURLY_PATH = PROCESSED_DATA_DIR / "hourly_features.parquet"


def train_p1(model_name: str = "baseline",n_estimators: int = 300,max_depth: int = 15,random_state: int = 0,) -> dict:
    """P1: estimate hourly station departures using a baseline or random forest."""
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
        raise ValueError(f"P1 is missing required columns: {sorted(missing_columns)}")

    missing_features = [column for column in p1_features if column not in df.columns]
    if missing_features:
        raise ValueError(f"P1 is missing planned features: {missing_features}")

    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df["departures"] = pd.to_numeric(df["departures"], errors="coerce")
    df = df.dropna(subset=["Origen_Id", "hour", "departures"])
    df = df.sort_values(["hour", "Origen_Id"]).reset_index(drop=True)

    train, test, split_time = chronological_split(df)

    # Register the exact training partition as an MLflow input dataset.
    training_dataset = mlflow.data.from_pandas(train,source=str(HOURLY_PATH),name="p1_training_dataset",)

    y_train = train["departures"].to_numpy()
    y_test = test["departures"].to_numpy()

    X_train = train[p1_features].copy()
    X_test = test[p1_features].copy()
    X_train["Origen_Id"] = X_train["Origen_Id"].astype("string")
    X_test["Origen_Id"] = X_test["Origen_Id"].astype("string")

    numeric_features = [feature for feature in p1_features if feature != "Origen_Id"]
    X_train_model = X_train[numeric_features].apply(pd.to_numeric, errors="coerce")
    X_test_model = X_test[numeric_features].apply(pd.to_numeric, errors="coerce")

    # Imputation statistics are learned only from the training partition.
    medians = X_train_model.median().fillna(0)
    X_train_model = X_train_model.fillna(medians)
    X_test_model = X_test_model.fillna(medians)

    if model_name == "baseline":
        model = DummyRegressor(strategy="mean")
        baseline_train_input = np.zeros((len(y_train), 1))
        model.fit(baseline_train_input, y_train)
        model_input_example = np.zeros((min(5, len(y_test)), 1))
        predictions = model.predict(np.zeros((len(y_test), 1)))
        model_type = "mean_baseline"
        model_filename = "p1_mean_baseline.joblib"
    elif model_name == "rf":
        model = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,random_state=random_state,n_jobs=-1,)
        model.fit(X_train_model, y_train)
        predictions = model.predict(X_test_model)
        model_input_example = X_test_model.head(5)
        model_type = "random_forest_regressor"
        model_filename = "p1_random_forest.joblib"
    else:
        raise ValueError("P1 model must be 'baseline' or 'rf'.")

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / model_filename
    joblib.dump(model, model_path)

    # Save plots as run artifacts.
    actual_vs_predicted_path = figures_dir / f"p1_{model_name}_actual_vs_predicted.png"
    residuals_path = figures_dir / f"p1_{model_name}_residuals.png"

    plot_limit = min(len(y_test), 5000)
    actual_sample = y_test[:plot_limit]
    predicted_sample = predictions[:plot_limit]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(actual_sample, predicted_sample, alpha=0.35, s=12)
    lower = float(min(np.min(actual_sample), np.min(predicted_sample)))
    upper = float(max(np.max(actual_sample), np.max(predicted_sample)))
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlabel("Valores reales")
    ax.set_ylabel("Valores predichos")
    ax.set_title(f"P1: reales vs. predichos ({model_name})")
    fig.tight_layout()
    fig.savefig(actual_vs_predicted_path, dpi=150)
    plt.close(fig)

    residuals = actual_sample - predicted_sample
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predicted_sample, residuals, alpha=0.35, s=12)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Valores predichos")
    ax.set_ylabel("Residuos (real - predicho)")
    ax.set_title(f"P1: residuos ({model_name})")
    fig.tight_layout()
    fig.savefig(residuals_path, dpi=150)
    plt.close(fig)

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

    config_path = REPORTS_DIR / f"p1_{model_name}_config.json"
    config_path.write_text(
        json.dumps(
            {
                "task": "P1_regression",
                "model": model_name,
                "n_estimators": n_estimators if model_name == "rf" else None,
                "max_depth": max_depth if model_name == "rf" else None,
                "random_state": random_state,
                "test_size": TEST_SIZE,
                "split": "chronological",
                "split_time": str(split_time),
                "features": numeric_features if model_name == "rf" else ["constant_zero"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log_mlflow_run(
        run_name=f"P1_{model_name}",
        params={
            "task": "P1_regression",
            "target": "departures",
            "model": model_name,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "test_size": TEST_SIZE,
            "split": "chronological",
            "split_time": str(split_time),
            "train_rows": len(train),
            "test_rows": len(test),
            "planned_features": p1_features,
            "model_features": numeric_features if model_name == "rf" else ["constant_zero"],
            "station_id_treated_as": "categorical",
            "data_start_time": str(df["hour"].min()),
            "data_end_time": str(df["hour"].max()),
            **dataset_metadata(HOURLY_PATH),
        },
        metrics={"mae": mae, "rmse": rmse},
        tags={"problem": "P1","question": "demand_prediction","model_type": model_type,},
        artifact_paths=[model_path],
        model=model,
        input_example=model_input_example,
        artifact_dirs={"plots": [actual_vs_predicted_path, residuals_path],"config": [config_path],},
        input_dataset=training_dataset,
    )

    logger.success("P1 {} completed.", model_name)
    return results
