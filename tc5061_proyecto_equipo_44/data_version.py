
"""Versionamiento reproducible de los datos de MiBici."""

import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def calculate_sha256(file_path: Path) -> str:
    """Calcula el hash SHA-256 sin cargar el archivo completo en memoria."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def count_csv_rows(file_path: Path) -> int:
    """Cuenta los registros de un CSV sin incluir el encabezado."""
    with file_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as file:
        reader = csv.reader(file)
        next(reader, None)
        return sum(1 for row in reader if row)


def create_data_manifest(
    source_files: list[Path],
    parquet_path: Path,
    manifest_path: Path = Path("references/data_manifest.json"),
) -> dict:
    """Genera un manifiesto con hashes y cantidades de registros."""

    if not source_files:
        raise ValueError("No se proporcionaron archivos CSV.")

    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    files = []

    for file_path in sorted(source_files, key=lambda path: path.name):
        file_path = Path(file_path)

        print(f"Verificando: {file_path.name}")

        files.append({
            "filename": file_path.name,
            "records": count_csv_rows(file_path),
            "sha256": calculate_sha256(file_path),
        })

    parquet_hash = calculate_sha256(parquet_path)
    parquet_rows = pq.read_metadata(parquet_path).num_rows

    version_content = {
        "source_files": files,
        "processed_sha256": parquet_hash,
    }

    version = hashlib.sha256(
        json.dumps(version_content, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    manifest = {
        "dataset": "MiBici",
        "version": version,
        "source_files": files,
        "processed_dataset": {
            "filename": parquet_path.name,
            "records": parquet_rows,
            "sha256": parquet_hash,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4, ensure_ascii=False)

    print(f"\nManifiesto guardado: {manifest_path}")
    print(f"Versión del dataset: {version}")
    print(f"Registros procesados: {parquet_rows:,}")

    return manifest


if __name__ == "__main__":
    source_dir = Path("data/external")
    parquet_path = Path("data/processed/dataset.parquet")

    monthly_files = sorted(
        source_dir.glob("datos_abiertos_*.csv")
    )[-24:]

    nomenclature_files = sorted(
        source_dir.glob("nomenclatura_*.csv")
    )

    if len(monthly_files) != 24:
        raise ValueError(
            f"Se esperaban 24 archivos mensuales; se encontraron {len(monthly_files)}."
        )

    if not nomenclature_files:
        raise FileNotFoundError("No se encontró la nomenclatura.")

    selected_files = monthly_files + [nomenclature_files[-1]]

    create_data_manifest(selected_files, parquet_path)
