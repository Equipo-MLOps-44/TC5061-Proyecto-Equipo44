"""Train P3 and track in MLflow."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tc5061_proyecto_equipo_44.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

from .data_utils import dataset_metadata
from .mlflow_utils import log_mlflow_run

RANDOM_STATE = 0
TEST_SIZE = 0.30
N_CLUSTERS = 3

HOURLY_PATH = PROCESSED_DATA_DIR / "hourly_features.parquet"
PROFILES_PATH = PROCESSED_DATA_DIR / "station_profiles.parquet"


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