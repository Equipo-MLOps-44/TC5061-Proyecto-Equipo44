
"""Validación de data drift para MiBici con DeepChecks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift


DATA_PATH = Path("data/processed/hourly_features.parquet")
REPORT_PATH = Path("reports/deepchecks/data_drift.html")

# Fecha de separación utilizada en P1
SPLIT_DATE = pd.Timestamp("2026-02-24")

RANDOM_STATE = 42
SAMPLE_SIZE = 10_000
SAMPLE_RATE = 0.01

# Variables utilizadas por el modelo P1
FEATURES = [
    "departures_lag_1h",
    "departures_lag_24h",
    "departures_lag_168h",
    "departures_rolling_mean_24h",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
]

CATEGORICAL = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
]


def sample_train_test(path: Path):
    """Obtener muestras reproducibles respetando el corte temporal."""

    rng = np.random.default_rng(RANDOM_STATE)

    train_samples = []
    test_samples = []

    parquet = pq.ParquetFile(path)

    for batch in parquet.iter_batches(
        batch_size=65_536,
        columns=["hour"] + FEATURES,
    ):
        frame = batch.to_pandas()

        # Muestreo aleatorio para no cargar millones de filas
        selected = frame.loc[
            rng.random(len(frame)) < SAMPLE_RATE
        ]

        train = selected.loc[
            selected["hour"] < SPLIT_DATE, FEATURES
        ]

        test = selected.loc[
            selected["hour"] >= SPLIT_DATE, FEATURES
        ]

        if not train.empty:
            train_samples.append(train)

        if not test.empty:
            test_samples.append(test)

    if not train_samples or not test_samples:
        raise ValueError(
            "No se obtuvieron muestras de ambos periodos."
        )

    train_df = pd.concat(train_samples, ignore_index=True)
    test_df = pd.concat(test_samples, ignore_index=True)

    # Mantener máximo 10,000 filas por conjunto
    train_df = train_df.sample(
        n=min(SAMPLE_SIZE, len(train_df)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    test_df = test_df.sample(
        n=min(SAMPLE_SIZE, len(test_df)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return train_df, test_df


def main():
    """Ejecutar DeepChecks y generar un reporte estático."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    train_df, test_df = sample_train_test(DATA_PATH)

    print(f"Registros entrenamiento: {len(train_df):,}")
    print(f"Registros prueba: {len(test_df):,}")
    print(f"Fecha de separación: {SPLIT_DATE.date()}")

    train_dataset = Dataset(
        train_df,
        cat_features=CATEGORICAL,
    )

    test_dataset = Dataset(
        test_df,
        cat_features=CATEGORICAL,
    )

    # Evaluación de cambios en las distribuciones
    check = FeatureDrift(
        n_samples=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    result = check.run(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    print("\nResultados de Data Drift:")

    rows = []

    for feature, information in result.value.items():
        score = information.get("Drift score")
        method = information.get("Method")

        print(f"{feature}: {score} ({method})")

        rows.append({
            "Variable": feature,
            "Drift score": score,
            "Método": method,
        })

    report = pd.DataFrame(rows)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <title>MiBici - Data Drift</title>
        <style>
            body {{
                font-family: Arial;
                max-width: 1000px;
                margin: 40px auto;
                padding: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                padding: 12px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background: #e8eef5;
            }}
        </style>
    </head>
    <body>
        <h1>MiBici - Data Drift</h1>
        <p>Entrenamiento: {len(train_df):,} registros</p>
        <p>Prueba: {len(test_df):,} registros</p>
        <p>Separación temporal: {SPLIT_DATE.date()}</p>
        <p>Semilla: {RANDOM_STATE}</p>
        {report.to_html(index=False, escape=True)}
    </body>
    </html>
    """

    REPORT_PATH.write_text(
        html,
        encoding="utf-8",
    )

    print(f"\nReporte generado: {REPORT_PATH}")


if __name__ == "__main__":
    main()

