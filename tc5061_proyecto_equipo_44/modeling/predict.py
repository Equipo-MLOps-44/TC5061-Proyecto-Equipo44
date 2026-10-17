import argparse

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from tc5061_proyecto_equipo_44.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)


def predict_p1() -> None:
    """Generate mean-baseline predictions for hourly feature records."""
    model_path = MODELS_DIR / "p1_mean_baseline.joblib"
    data_path = PROCESSED_DATA_DIR / "hourly_features.parquet"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Train P1 before predicting: {model_path}"
        )

    df = pd.read_parquet(data_path)
    if "departures" not in df.columns:
        raise ValueError("P1 requires the departures column.")

    df = df.copy()
    df["departures"] = pd.to_numeric(
        df["departures"], errors="coerce"
    )
    df = df.dropna(subset=["departures"])

    model = joblib.load(model_path)
    df["predicted_departures"] = model.predict(
        np.zeros((len(df), 1))
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "p1_baseline_predictions.csv"
    df.to_csv(output, index=False)
    logger.success("P1 predictions saved to {}", output)


def predict_p2() -> None:
    """
    Predict both station risk indicators using their trained baseline models.
    """
    data_path = PROCESSED_DATA_DIR / "hourly_features.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"P2 feature dataset not found: {data_path}. "
            "Run feature engineering first."
        )

    df = pd.read_parquet(data_path)

    risk_models = {
        "risk_no_bikes": {
            "model_file": "p2_no_bikes_dummy_baseline.joblib",
            "prediction_column": "predicted_risk_no_bikes",
        },
        "risk_no_docks": {
            "model_file": "p2_no_docks_dummy_baseline.joblib",
            "prediction_column": "predicted_risk_no_docks",
        },
    }

    available_models = {
        target: config
        for target, config in risk_models.items()
        if (MODELS_DIR / config["model_file"]).exists()
    }

    if not available_models:
        expected = ", ".join(
            config["model_file"] for config in risk_models.values()
        )
        raise FileNotFoundError(
            f"No P2 risk models found in {MODELS_DIR}. "
            f"Expected one or both of: {expected}. Train P2 first."
        )

    df = df.copy()
    for target, config in available_models.items():
        model_path = MODELS_DIR / config["model_file"]
        artifact = joblib.load(model_path)

        model = artifact["model"]
        feature_columns = artifact["features"]
        medians = artifact["medians"]

        missing = set(feature_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"P2 model for {target} is missing input columns: "
                f"{sorted(missing)}"
            )

        X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(medians)

        df[config["prediction_column"]] = model.predict(X)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "p2_baseline_predictions.csv"
    df.to_csv(output, index=False)
    logger.success("P2 risk predictions saved to {}", output)


def predict_p3() -> None:
    """Assign stations to their learned KMeans clusters."""
    model_path = MODELS_DIR / "p3_kmeans_baseline.joblib"
    data_path = PROCESSED_DATA_DIR / "station_profiles.parquet"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Train P3 before predicting: {model_path}"
        )

    df = pd.read_parquet(data_path)
    artifact = joblib.load(model_path)

    model = artifact["model"]
    feature_columns = artifact["features"]

    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"P3 input is missing columns: {sorted(missing)}"
        )

    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0)

    df = df.copy()
    df["cluster"] = model.predict(X)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "p3_station_clusters.csv"
    df.to_csv(output, index=False)
    logger.success("P3 clusters saved to {}", output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate predictions from baseline models."
    )
    parser.add_argument(
        "--task",
        choices=["p1", "p2", "p3", "all"],
        default="all",
        help="Business question to predict.",
    )
    args = parser.parse_args()

    tasks = {
        "p1": predict_p1,
        "p2": predict_p2,
        "p3": predict_p3,
    }

    selected = tasks if args.task == "all" else {args.task: tasks[args.task]}

    for name, predict_function in selected.items():
        try:
            predict_function()
        except FileNotFoundError as error:
            logger.warning("{}: {}", name.upper(), error)


if __name__ == "__main__":
    main()
