
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

DATA_PATH = Path("data/processed/hourly_features_challenger.parquet")

FEATURES = [
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


def load_period(start, end):
    data = pd.read_parquet(
        DATA_PATH,
        columns=["hour"] + FEATURES,
        filters=[
            ("hour", ">=", pd.Timestamp(start)),
            ("hour", "<", pd.Timestamp(end)),
        ],
    )

    if data.empty:
        raise ValueError(f"No hay datos entre {start} y {end}")

    # Muestra reproducible para controlar el tiempo de ejecución
    data = data.sample(
        n=min(10000, len(data)),
        random_state=42,
    )

    return data[FEATURES].reset_index(drop=True)


# Periodos propuestos
reference = load_period("2024-10-01", "2026-10-01")
current = load_period("2026-10-01", "2026-11-01")

print(f"Registros de referencia: {len(reference)}")
print(f"Registros actuales: {len(current)}")

report = Report([
    DataDriftPreset(),
    DataSummaryPreset(),
])

result = report.run(
    current_data=current,
    reference_data=reference,
)

output = Path("reports/evidently/p2_drift_v2.html")
output.parent.mkdir(parents=True, exist_ok=True)

result.save_html(str(output))

print(f"Reporte generado: {output}")
