import csv
from pathlib import Path

import pytest

from tc5061_proyecto_equipo_44.dataset import ingest_csv_files


def test_ingest_csv_files_combines_different_schemas(tmp_path: Path) -> None:
    source_dir = tmp_path / "external"
    source_dir.mkdir()
    (source_dir / "trips.csv").write_text(
        "trip_id,station_id\n1,10\n",
        encoding="utf-8",
    )
    (source_dir / "stations.csv").write_bytes(
        "station_id,name\n10,Caf\u00e9\n".encode("latin-1")
    )
    (source_dir / "download.csv.crdownload").write_text("incomplete")
    output_path = tmp_path / "processed" / "dataset.csv"

    result = ingest_csv_files(source_dir, output_path)

    assert result == output_path
    with output_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {"trip_id": "", "station_id": "10", "name": "Caf\u00e9"},
        {"trip_id": "1", "station_id": "10", "name": ""},
    ]


def test_ingest_csv_files_requires_csv_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        ingest_csv_files(tmp_path, tmp_path / "dataset.csv")
