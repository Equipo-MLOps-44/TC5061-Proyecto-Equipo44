from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.request import Request

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tc5061_proyecto_equipo_44.dataset import (
    download_recent_data,
    ingest_csv_files,
)


def test_ingest_csv_files_combines_different_schemas(tmp_path: Path) -> None:
    source_dir = tmp_path / "external"
    source_dir.mkdir()
    (source_dir / "trips.csv").write_text(
        "trip_id,station_id\n1,10\n",
        encoding="utf-8",
    )
    (source_dir / "stations.csv").write_bytes("station_id,name\n10,Caf\u00e9\n".encode("latin-1"))
    (source_dir / "download.csv.crdownload").write_text("incomplete")
    (source_dir / "nomenclatura_2026_09.csv").write_text(
        "station_id,name\n10,Central\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "processed" / "dataset.parquet"

    result = ingest_csv_files(source_dir, output_path)

    assert result == output_path
    rows = pq.read_table(output_path).to_pylist()
    assert rows == [
        {"trip_id": "", "station_id": "10", "name": "Caf\u00e9"},
        {"trip_id": "1", "station_id": "10", "name": ""},
    ]


def test_ingest_csv_files_requires_csv_files(tmp_path: Path) -> None:
    (tmp_path / "nomenclatura_2026_09.csv").write_text(
        "station_id,name\n10,Central\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        ingest_csv_files(tmp_path, tmp_path / "dataset.parquet")


def test_ingest_normalizes_columns_and_parses_dates(tmp_path: Path) -> None:
    source_dir = tmp_path / "external"
    source_dir.mkdir()
    (source_dir / "trips.csv").write_bytes(
        (
            "\ufeffViaje_Id,Año_de_nacimiento,Inicio_del_viaje,Fin_del_viaje,Genero\n"
            "123,1997,2026-09-01 00:00:59,2026-09-01 00:18:55,Cafe\u0301\n"
            "124,2001,2026-09-02 01:02:03,,M\n"
            "125,1978,01/10/2024 0:00,01/10/2024 0:21,F\n"
        ).encode("utf-8")
    )
    output_path = tmp_path / "processed" / "dataset.parquet"

    ingest_csv_files(source_dir, output_path)

    table = pq.read_table(output_path)
    assert table.schema.names == [
        "viaje_id",
        "ano_de_nacimiento",
        "inicio_del_viaje",
        "fin_del_viaje",
        "genero",
    ]
    assert table.schema.field("inicio_del_viaje").type == pa.timestamp("us")
    assert table.schema.field("fin_del_viaje").type == pa.timestamp("us")
    assert table.to_pylist() == [
        {
            "viaje_id": "123",
            "ano_de_nacimiento": "1997",
            "inicio_del_viaje": datetime(2026, 9, 1, 0, 0, 59),
            "fin_del_viaje": datetime(2026, 9, 1, 0, 18, 55),
            "genero": "Caf\u00e9",
        },
        {
            "viaje_id": "124",
            "ano_de_nacimiento": "2001",
            "inicio_del_viaje": datetime(2026, 9, 2, 1, 2, 3),
            "fin_del_viaje": None,
            "genero": "M",
        },
        {
            "viaje_id": "125",
            "ano_de_nacimiento": "1978",
            "inicio_del_viaje": datetime(2024, 10, 1, 0, 0),
            "fin_del_viaje": datetime(2024, 10, 1, 0, 21),
            "genero": "F",
        },
    ]


def test_download_recent_data_selects_last_24_months_and_nomenclature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_url = "https://mibici.example/es/datos-abiertos/"
    links = []
    year, month = 2024, 9
    while (year, month) <= (2026, 9):
        links.append(
            f'<a href="/site/assets/files/{year}{month:02d}/'
            f'datos_abiertos_{year}_{month:02d}.csv">CSV</a>'
        )
        month += 1
        if month == 13:
            year += 1
            month = 1
    links.append('<a href="/site/assets/files/1118/nomenclatura_2026_09.csv">' "Nomenclatura</a>")
    page_content = "".join(links).encode("utf-8")
    requested_urls: list[str] = []

    def fake_urlopen(request: Request, timeout: int) -> BytesIO:
        assert timeout == 30
        assert request.get_header("User-agent").startswith("Mozilla/5.0 (Windows NT")
        assert request.get_header("Accept-language") == "es-MX,es;q=0.9,en;q=0.8"
        url = request.full_url
        requested_urls.append(url)
        if url == page_url:
            return BytesIO(page_content)
        return BytesIO(b"trip_id,value\n1,test\n")

    monkeypatch.setattr(
        "tc5061_proyecto_equipo_44.dataset.urlopen",
        fake_urlopen,
    )

    downloaded = download_recent_data(tmp_path, page_url)

    monthly_names = sorted(
        path.name for path in downloaded if path.name.startswith("datos_abiertos_")
    )
    assert len(monthly_names) == 24
    assert monthly_names[0] == "datos_abiertos_2024_10.csv"
    assert monthly_names[-1] == "datos_abiertos_2026_09.csv"
    assert (tmp_path / "nomenclatura_2026_09.csv").is_file()
    assert len(requested_urls) == 26
