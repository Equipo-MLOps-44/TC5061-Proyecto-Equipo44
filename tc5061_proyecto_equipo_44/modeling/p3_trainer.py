"""Train P3 and track in MLflow."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import mlflow
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tc5061_proyecto_equipo_44.config import (MODELS_DIR,PROCESSED_DATA_DIR,REPORTS_DIR,)

from .data_utils import dataset_metadata
from .mlflow_utils import log_mlflow_run

RANDOM_STATE = 0
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

    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0)

    if len(X) <= n_clusters:
        raise ValueError(f"P3 needs more than {n_clusters} station records.")

    if len(X.drop_duplicates()) < n_clusters:
        raise ValueError("P3 has fewer distinct profiles than clusters.")

    model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=n_clusters,random_state=random_state,n_init=n_init,),
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
    plots_dir = REPORTS_DIR / "p3_plots"
    config_dir = REPORTS_DIR / "p3_config"
    plots_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"p3_kmeans_{n_clusters}_clusters.joblib"
    clusters_path = REPORTS_DIR / f"p3_station_clusters_{n_clusters}.csv"
    config_path = config_dir / f"p3_kmeans_{n_clusters}_config.json"
    distribution_plot_path = plots_dir / f"p3_cluster_distribution_{n_clusters}.png"
    silhouette_plot_path = plots_dir / f"p3_silhouette_{n_clusters}.png"

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

    # Persist the effective training configuration as a separate artifact.
    config = {
        "task": "P3_clustering",
        "algorithm": "KMeans",
        "n_clusters": n_clusters,
        "random_state": random_state,
        "n_init": n_init,
        "features": feature_columns,
        "stations": int(len(X)),
        "dataset_file": PROFILES_PATH.name,
        "preprocessing": ["replace_inf_with_nan", "median_imputation", "StandardScaler"],
        "silhouette_score": float(silhouette),
        "cluster_sizes": cluster_sizes,
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # Plot the number of stations assigned to each cluster.
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(int(i)) for i in counts.index], counts.values)
    ax.set_title("P3: distribución de estaciones por clúster")
    ax.set_xlabel("Clúster")
    ax.set_ylabel("Número de estaciones")
    fig.tight_layout()
    fig.savefig(distribution_plot_path, dpi=150)
    plt.close(fig)

    # Silhouette values by cluster; this is a diagnostic visualization.
    sample_values = silhouette_samples(scaled_X, labels)
    fig, ax = plt.subplots(figsize=(9, 6))
    y_lower = 10
    for cluster_id in sorted(np.unique(labels)):
        values = np.sort(sample_values[labels == cluster_id])
        y_upper = y_lower + len(values)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            values,
            alpha=0.7,
            label=f"Clúster {cluster_id}",
        )
        y_lower = y_upper + 10
    ax.axvline(float(silhouette), linestyle="--", label=f"Promedio = {silhouette:.3f}")
    ax.set_title("P3: análisis Silhouette por clúster")
    ax.set_xlabel("Coeficiente Silhouette")
    ax.set_ylabel("Estaciones agrupadas por clúster")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(silhouette_plot_path, dpi=150)
    plt.close(fig)

    training_dataset = mlflow.data.from_pandas(
        df,
        source=str(PROFILES_PATH),
        name="p3_station_profiles_training_dataset",
    )

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
        metrics={"data_silhouette_score": silhouette,"data_stations": len(X),},
        tags={"problem": "P3","question": "station_clustering","model_type": "kmeans_baseline",},
        artifact_dirs={"model_files": [model_path],"data": [clusters_path],"config": [config_path],"plots": [distribution_plot_path, silhouette_plot_path],},
        # Log the fitted pipeline as an MLflow model, including its signature
        # and a representative input example for reproducibility/serving.
        model=model,
        input_example=X.head(5),
        input_dataset=training_dataset,
    )
    return results
