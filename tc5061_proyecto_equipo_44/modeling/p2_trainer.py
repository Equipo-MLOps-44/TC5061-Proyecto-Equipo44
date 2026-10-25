"""Train P2 and track in MLflow."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,f1_score,recall_score,)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from tc5061_proyecto_equipo_44.config import (MODELS_DIR,PROCESSED_DATA_DIR,REPORTS_DIR,)

from .data_utils import chronological_split, dataset_metadata
from .mlflow_utils import log_mlflow_run



RANDOM_STATE = 0
TEST_SIZE = 0.30
N_CLUSTERS = 3

HOURLY_PATH = PROCESSED_DATA_DIR / "hourly_features.parquet"
PROFILES_PATH = PROCESSED_DATA_DIR / "station_profiles.parquet"



def train_p2(model_name: str = "baseline",n_estimators: int = 300,max_depth: int = 15,class_weight: str = "balanced",random_state: int = 0,run_name_suffix: str | None = None,stage_of_work: str | None = None,) -> dict | None:
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

        X_train = train[feature_columns].apply(pd.to_numeric, errors="coerce")
        X_test = test[feature_columns].apply(pd.to_numeric, errors="coerce")

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

        labels = sorted(set(y_train.unique()).union(set(y_test.unique())))

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test,predictions,average="macro",zero_division=0,)
        macro_recall = recall_score(y_test,predictions,average="macro",zero_division=0,)

        metrics = {
            "task": "P2_classification",
            "target": target,
            "model_type": model_type,
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
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
            "classification_report": classification_report(y_test,predictions,labels=labels,output_dict=True,zero_division=0,),
        }

        target_slug = "no_bikes" if target == "risk_no_bikes" else "no_docks"
        suffix = f"_{run_name_suffix}" if run_name_suffix else ""
        model_filename = f"p2_{target_slug}_{model_name}{suffix}.joblib"

        model_path = MODELS_DIR / model_filename

        # Keep P2 evaluation artifacts separate for each target and model run.
        artifact_slug = f"p2_{target_slug}_{model_name}{suffix}"
        artifact_dir = REPORTS_DIR / "p2" / artifact_slug
        plots_dir = artifact_dir / "plots"
        config_dir = artifact_dir / "config"
        reports_dir = artifact_dir / "data"
        plots_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump({"model": model,"features": feature_columns,"medians": medians,},model_path,)

        # Save the confusion matrix as a PNG.
        cm = confusion_matrix(y_test, predictions, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(cm, interpolation="nearest")
        ax.figure.colorbar(image, ax=ax)
        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Predicted label",
            ylabel="True label",
            title=f"Confusion matrix: {target}",
        )
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        threshold = cm.max() / 2.0 if cm.size else 0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(
                    col, row, format(cm[row, col], "d"),
                    ha="center", va="center",
                    color="white" if cm[row, col] > threshold else "black",
                )
        fig.tight_layout()
        confusion_plot_path = plots_dir / "confusion_matrix.png"
        fig.savefig(confusion_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save a per-class report as JSON.
        report_path = reports_dir / "classification_report.json"
        report_path.write_text(
            json.dumps(metrics["classification_report"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Save reproducibility/configuration metadata as JSON.
        config_path = config_dir / "config.json"
        config_payload = {
            "task": "P2_classification",
            "target": target,
            "model": model_name,
            "model_type": model_type,
            "random_state": random_state,
            "test_size": TEST_SIZE,
            "split": "chronological_by_timestamp",
            "split_time": str(split_time),
            "features": feature_columns,
            "imputation_medians": {
                str(key): (None if pd.isna(value) else float(value))
                for key, value in medians.items()
            },
            "data_start_time": str(df["hour"].min()),
            "data_end_time": str(df["hour"].max()),
            "train_rows": len(train),
            "test_rows": len(test),
        }
        if model_name == "rf":
            config_payload.update({"n_estimators": n_estimators,"max_depth": max_depth,"class_weight": class_weight,})
        config_path.write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # Register the exact training partition as an MLflow dataset input.
        # The source points to the processed parquet used to build this split.
        training_dataset = mlflow.data.from_pandas(train,source=str(HOURLY_PATH),name=f"p2_{target}_training_dataset",)

        log_mlflow_run(
            run_name=f"P2_{target}_{model_name}{suffix}",
            params={
                "task": "P2_classification",
                "target": target,
                "model": model_name,
                **(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "class_weight": class_weight,
                        }
                        if model_name == "rf"
                        else {}
                ),
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
            metrics={"accuracy": accuracy,"macro_f1": macro_f1,"macro_recall": macro_recall,},
            tags={
                "problem": "P2",
                "question": "p2",
                "target": target,
                "model_type": model_type,
                "model_family": (
                    "random_forest" if model_name == "rf" else "dummy_classifier"
                ),
                "stage_of_work": stage_of_work
                or ("baseline" if model_name == "baseline" else "training"),
            },
            artifact_paths=[model_path],
            artifact_dirs={"plots": [confusion_plot_path],"data": [report_path],"config": [config_path],},
            model=model,
            input_example=X_test.head(5),
            input_dataset=training_dataset,
        )

        results[target] = metrics
        logger.success(
            "P2 {} completed for '{}'.",
            model_type,
            target,
        )

    return results or None

# 3 * 2 * 2 = 12 configurations; each configuration evaluates both P2 targets.
P2_RF_PARAM_GRID = {"max_depth": [5, 10, 20],"class_weight": ["balanced", "balanced_subsample"],}


def search_p2(n_estimators: int = 300,random_state: int = RANDOM_STATE,) -> dict:
    """Evaluate a grid of Random Forest configurations using nested MLflow runs."""
    configurations = list(ParameterGrid(P2_RF_PARAM_GRID))
    results = {}

    with mlflow.start_run(run_name="P2_RF_Hyperparameter_Search",nested=mlflow.active_run() is not None,):
        mlflow.set_tags({"problem": "P2","question": "p2","model_family": "random_forest","stage_of_work": "hyperparameter_search","team": "equipo44",})
        
        mlflow.log_params({"search_type": "parameter_grid","n_estimators": n_estimators,"number_of_configurations": len(configurations),"random_state": random_state,})

        for index, config in enumerate(configurations, start=1):
            config_name = (
                f"config_{index:02d}_n{n_estimators}"
                f"_d{config['max_depth']}_{config['class_weight']}"
            )

            results[config_name] = {
                "configuration": {
                    "n_estimators": n_estimators,
                    **config,
                },
                "results": train_p2(
                    model_name="rf",
                    n_estimators=n_estimators,
                    max_depth=config["max_depth"],
                    class_weight=config["class_weight"],
                    random_state=random_state,
                    run_name_suffix=config_name,
                    stage_of_work="hyperparameter_search",
                ),
            }

    logger.success(
        "P2 hyperparameter search completed: {} configurations.",
        len(configurations),
    )
    return results
