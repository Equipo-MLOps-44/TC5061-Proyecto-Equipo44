
"""Validación de calidad de datos MiBici con DeepChecks."""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    ColumnsInfo,
    DataDuplicates,
    MixedDataTypes,
    MixedNulls,
    PercentOfNulls,
)


# Configuración
DATA_PATH = Path("data/processed/dataset.parquet")
REPORT_PATH = Path("reports/deepchecks/integridad_viajes.html")

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42

# Excluimos identificadores individuales del reporte.
COLUMNS = [
    "genero",
    "ano_de_nacimiento",
    "inicio_del_viaje",
    "fin_del_viaje",
    "origen_id",
    "destino_id",
]


def sample_parquet(path: Path) -> pd.DataFrame:
    """Tomar una muestra reproducible sin cargar todo el Parquet."""

    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows

    if total_rows == 0:
        raise ValueError("El dataset está vacío.")

    rng = np.random.default_rng(RANDOM_STATE)

    positions = np.sort(
        rng.choice(
            total_rows,
            size=min(SAMPLE_SIZE, total_rows),
            replace=False,
        )
    )

    samples = []
    offset = 0

    for batch in parquet.iter_batches(
        batch_size=65_536,
        columns=COLUMNS,
    ):
        end = offset + batch.num_rows

        start_index = np.searchsorted(positions, offset)
        end_index = np.searchsorted(positions, end)

        if end_index > start_index:
            local_positions = positions[start_index:end_index] - offset

            selected = batch.take(pa.array(local_positions))
            samples.append(selected.to_pandas())

        offset = end

    return pd.concat(samples, ignore_index=True)


def main() -> None:
    """Ejecutar las validaciones y generar reportes."""

    # 1. Verificar que exista el dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe: {DATA_PATH}")

    # 2. Obtener una muestra reproducible
    data = sample_parquet(DATA_PATH)

    print(f"Registros analizados: {len(data):,}")
    print(f"Semilla utilizada: {RANDOM_STATE}")

    # 3. Preparar los datos para DeepChecks
    dataset = Dataset(
        data,
        datetime_name="inicio_del_viaje",
    )

    # 4. Definir las validaciones de calidad
    suite = Suite(
        "MiBici - Integridad de viajes",
        ColumnsInfo(),
        PercentOfNulls(),
        MixedNulls(),
        MixedDataTypes(),
        DataDuplicates(),
    )

    # 5. Ejecutar DeepChecks
    result = suite.run(train_dataset=dataset)

    # 6. Crear carpeta de reportes
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 7. Guardar reporte original de DeepChecks
    result.save_as_html(
        str(REPORT_PATH),
        as_widget=False,
        requirejs=False,
        connected=False,
    )

    print(f"Reporte original: {REPORT_PATH}")

    # 8. Generar reporte alternativo sin JavaScript
    simple_path = REPORT_PATH.with_name(
        "integridad_viajes_simple.html"
    )

    sections = [
        "<!DOCTYPE html>",
        "<html lang='es'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>MiBici - Validación de datos</title>",
        "<style>",
        "body {font-family: Arial; max-width: 1000px; "
        "margin: 40px auto; padding: 20px;}",
        "h1 {color: #23395d;}",
        "h2 {margin-top: 30px; color: #34495e;}",
        "pre {background: #f3f4f6; padding: 16px; "
        "white-space: pre-wrap; overflow-wrap: anywhere;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>MiBici - Integridad de viajes</h1>",
        f"<p>Muestra: {len(data):,} registros</p>",
        f"<p>Semilla: {RANDOM_STATE}</p>",
    ]

    # 9. Incorporar los resultados de cada chequeo
    for check in result.results:
        name = escape(
            str(getattr(check, "header", type(check).__name__))
        )

        value = escape(
            str(
                getattr(
                    check,
                    "value",
                    "Sin resultados disponibles",
                )
            )
        )

        sections.append(f"<h2>{name}</h2>")
        sections.append(f"<pre>{value}</pre>")

    sections.append("</body></html>")

    # 10. Guardar reporte alternativo
    simple_path.write_text(
        "\n".join(sections),
        encoding="utf-8",
    )

    print(f"Reporte alternativo: {simple_path}")
    print("Validación finalizada correctamente.")


if __name__ == "__main__":
    main()
