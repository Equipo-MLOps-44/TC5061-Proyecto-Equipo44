
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from tc5061_proyecto_equipo_44.features import create_station_profiles


DATA_PATH = Path("data/processed/hourly_features.parquet")

COLUMNS = [
    "Origen_Id",
    "hour",
    "departures",
    "arrivals",
    "dpcapacity",
    "latitude",
    "longitude",
]

FEATURES = [
    "total_departures",
    "total_arrivals",
    "morning_departures",
    "weekend_departures",
    "avg_daily_departures",
    "morning_share",
    "weekend_share",
    "dpcapacity",
    "latitude",
    "longitude",
]


def load_profiles(start, end):
    # Leer solamente los datos del periodo seleccionado
    df = pd.read_parquet(
        DATA_PATH,
        columns=COLUMNS,
        filters=[
            ("hour", ">=", pd.Timestamp(start)),
            ("hour", "<", pd.Timestamp(end)),
        ],
    )

    df = df.dropna(subset=["Origen_Id", "hour"])

    if df.empty:
        raise ValueError(f"No hay datos entre {start} y {end}")

    # Reutilizar la funcion original del equipo
    profiles = create_station_profiles(df)

    # Recuperar capacidad y ubicacion de cada estacion
    station_info = (
        df.groupby("Origen_Id")[
            ["dpcapacity", "latitude", "longitude"]
        ]
        .median()
        .reset_index()
    )

    profiles = profiles.merge(
        station_info,
        on="Origen_Id",
        how="left",
        validate="one_to_one",
    )

    return profiles


# Periodos provisionales de igual duracion (92 dias)
reference = load_profiles("2025-07-01", "2025-10-01")
current = load_profiles("2026-07-01", "2026-10-01")

# Comparar las mismas estaciones en ambos periodos
reference_ids = set(reference["Origen_Id"])
current_ids = set(current["Origen_Id"])
common_ids = reference_ids & current_ids

print("Estaciones de referencia:", len(reference_ids))
print("Estaciones actuales:", len(current_ids))
print("Estaciones compartidas:", len(common_ids))
print("Estaciones nuevas:", len(current_ids - reference_ids))
print("Estaciones ausentes:", len(reference_ids - current_ids))

if len(common_ids) < 2:
    raise ValueError("No hay suficientes estaciones compartidas.")

reference = (
    reference[reference["Origen_Id"].isin(common_ids)]
    .sort_values("Origen_Id")[FEATURES]
    .reset_index(drop=True)
)

current = (
    current[current["Origen_Id"].isin(common_ids)]
    .sort_values("Origen_Id")[FEATURES]
    .reset_index(drop=True)
)

print("Generando reporte de Evidently para P3...")

report = Report([
    DataDriftPreset(),
    DataSummaryPreset(),
])

result = report.run(
    current_data=current,
    reference_data=reference,
)

output = Path("reports/evidently/p3_drift.html")
output.parent.mkdir(parents=True, exist_ok=True)

result.save_html(str(output))

print(f"Reporte generado: {output}")
