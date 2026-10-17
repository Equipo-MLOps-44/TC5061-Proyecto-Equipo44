"""@file dataset.py
@brief Download MiBici open data and ingest trip CSVs into Parquet.

The module discovers published CSV links from the official MiBici open-data
page, downloads the latest available 24-month window and the newest
nomenclature file, then combines the trip data using bounded memory.

Column names are converted to lowercase ASCII ``snake_case``. Text values are
normalized to Unicode NFC, and the trip start/end columns are stored as
``timestamp[us]`` in Parquet. Other columns remain strings to preserve
identifiers and their formatting.

Run this module from the project root with
``python -m tc5061_proyecto_equipo_44.dataset``.

@var DATA_PAGE_URL Official page used to discover published MiBici CSV links.
@var HTTP_HEADERS HTTP headers required by the MiBici website.
@var MONTHLY_FILENAME_RE Pattern used to identify monthly trip CSV files.
@var NOMENCLATURE_FILENAME_RE Pattern used to identify nomenclature CSV files.
@var PARQUET_BATCH_SIZE Maximum number of rows held for a Parquet write batch.
@var DATE_COLUMNS Normalized columns parsed and written as timestamps.
"""

import codecs
import csv
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Iterable
import unicodedata
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from tc5061_proyecto_equipo_44.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

DATA_PAGE_URL = "https://mibici.net/es/datos-abiertos/"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
}
MONTHLY_FILENAME_RE = re.compile(r"datos_abiertos_(\d{4})_(0[1-9]|1[0-2])(?:-[^.]+)?\.csv$")
NOMENCLATURE_FILENAME_RE = re.compile(r"nomenclatura_(\d{4})_(0[1-9]|1[0-2])(?:-[^.]+)?\.csv$")
PARQUET_BATCH_SIZE = 100_000
DATE_COLUMNS = {"inicio_del_viaje", "fin_del_viaje"}


class _DataLinksParser(HTMLParser):
    """@brief Extract monthly and nomenclature CSV links from MiBici HTML.

    Only links to CSV assets hosted under the MiBici site's assets directory
    are collected.
    """

    def __init__(self, page_url: str) -> None:
        """@brief Initialize the parser for a MiBici data page.

        @param page_url URL of the page containing CSV download links.
        """
        super().__init__()
        self._page_url = page_url
        self.monthly_links: dict[tuple[int, int], str] = {}
        self.nomenclature_links: dict[tuple[int, int], str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """@brief Store supported MiBici CSV links found in an HTML start tag.

        Links are keyed by their ``(year, month)`` pair.

        @param tag HTML element tag.
        @param attrs Attributes associated with the HTML element.
        """
        if tag != "a":
            return

        href = dict(attrs).get("href")
        if not href:
            return

        url = urljoin(self._page_url, href)
        parsed_url = urlsplit(url)
        if parsed_url.netloc != urlsplit(self._page_url).netloc or not parsed_url.path.startswith(
            "/site/assets/files/"
        ):
            return

        filename = PurePosixPath(parsed_url.path).name
        monthly_match = MONTHLY_FILENAME_RE.fullmatch(filename)
        nomenclature_match = NOMENCLATURE_FILENAME_RE.fullmatch(filename)

        if monthly_match:
            year, month = (int(part) for part in monthly_match.groups()[:2])
            self.monthly_links[(year, month)] = url
        elif nomenclature_match:
            year, month = (int(part) for part in nomenclature_match.groups()[:2])
            self.nomenclature_links[(year, month)] = url


def _download_file(url: str, destination_dir: Path) -> Path:
    """@brief Download one file atomically into a destination directory.

    The response is first streamed to a temporary file in the destination
    directory. The temporary file replaces the final path only after the
    transfer completes.

    @param url URL of the file to download.
    @param destination_dir Directory where the downloaded file is stored.
    @return Path to the completed local file.
    @throws OSError If the destination cannot be written or renamed.
    @throws urllib.error.URLError If the network request fails.
    """
    filename = PurePosixPath(urlsplit(url).path).name
    destination = destination_dir / filename
    temporary_path: Path | None = None
    request = Request(url, headers=HTTP_HEADERS)

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination_dir,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with urlopen(request, timeout=30) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return destination


def download_recent_data(
    destination_dir: Path = EXTERNAL_DATA_DIR,
    page_url: str = DATA_PAGE_URL,
) -> list[Path]:
    """@brief Download recent MiBici trip data and its newest nomenclature CSV.

    The monthly files are selected from the latest available month backwards
    over a 24-month window. The most recent nomenclature link on the page is
    downloaded separately from the trip data.

    @param destination_dir Directory where all CSV downloads are stored.
    @param page_url MiBici page used to discover download links.
    @return Paths to the downloaded monthly trip files and nomenclature file.
    @throws ValueError If the page contains no monthly or no nomenclature CSV links.
    @throws urllib.error.URLError If the page or any file cannot be fetched.
    @throws OSError If the destination directory or downloaded files cannot be written.
    """
    request = Request(page_url, headers=HTTP_HEADERS)
    with urlopen(request, timeout=30) as response:
        page = response.read().decode("utf-8", errors="replace")

    parser = _DataLinksParser(page_url)
    parser.feed(page)
    if not parser.monthly_links:
        raise ValueError(f"No monthly MiBici CSV links found on {page_url}")
    if not parser.nomenclature_links:
        raise ValueError(f"No MiBici nomenclature CSV link found on {page_url}")

    latest_month = max(parser.monthly_links)
    latest_ordinal = latest_month[0] * 12 + latest_month[1] - 1
    first_month_ordinal = latest_ordinal - 23
    recent_months = sorted(
        month
        for month in parser.monthly_links
        if month[0] * 12 + month[1] - 1 >= first_month_ordinal
    )
    latest_nomenclature = max(parser.nomenclature_links)

    destination_dir.mkdir(parents=True, exist_ok=True)
    urls = [parser.monthly_links[month] for month in recent_months]
    urls.append(parser.nomenclature_links[latest_nomenclature])
    downloaded_paths = [_download_file(url, destination_dir) for url in urls]

    logger.success(
        "Downloaded {} monthly MiBici CSV files and nomenclature {}_{} into {}",
        len(recent_months),
        latest_nomenclature[0],
        f"{latest_nomenclature[1]:02d}",
        destination_dir,
    )
    return downloaded_paths


def _get_csv_encoding(csv_path: Path) -> str:
    """@brief Detect the text encoding used by a CSV file.

    UTF-8 (with or without a byte-order mark) is preferred. If the sampled
    bytes are not valid UTF-8, Windows-1252 is used for legacy MiBici exports.

    @param csv_path CSV file whose encoding is inspected.
    @return Encoding name accepted by ``Path.open``.
    @throws OSError If the file cannot be opened or read.
    """
    with csv_path.open("rb") as file:
        sample = file.read(65536)

    try:
        codecs.getincrementaldecoder("utf-8-sig")().decode(sample, final=False)
    except UnicodeDecodeError:
        return "cp1252"
    return "utf-8-sig"


def _normalize_column_name(column_name: str) -> str:
    """@brief Convert a source column name into lowercase ASCII ``snake_case``.

    Diacritics are removed and runs of non-alphanumeric characters become a
    single underscore.

    @param column_name Original column name from the CSV header.
    @return Normalized column name.
    @throws ValueError If normalization leaves no ASCII letters or digits.
    """
    decomposed = unicodedata.normalize("NFKD", column_name.strip())
    ascii_name = decomposed.encode("ascii", errors="ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_name).strip("_").lower()
    if not normalized:
        raise ValueError(f"Column name contains no ASCII letters or digits: {column_name!r}")
    return normalized


def _normalize_text(value: str) -> str:
    """@brief Normalize a decoded text field to Unicode NFC.

    @param value Decoded CSV field value.
    @return NFC-normalized Unicode string.
    """
    return unicodedata.normalize("NFC", value)


def _parse_date(value: str, csv_path: Path, row_number: int, column_name: str) -> datetime | None:
    """@brief Parse a MiBici date value into a Python datetime.

    ISO timestamps and ``dd/mm/yyyy HH:MM[:SS]`` values are accepted. A blank
    field is represented as ``None`` and therefore as a null Parquet value.

    @param value Date string read from the CSV.
    @param csv_path Source file containing the value.
    @param row_number One-based CSV row number, including its header row.
    @param column_name Normalized name of the date column.
    @return Parsed datetime, or ``None`` for a blank value.
    @throws ValueError If the non-empty value does not match a supported format.
    """
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        formats = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M")
        for date_format in formats:
            try:
                return datetime.strptime(value, date_format)
            except ValueError:
                continue
    raise ValueError(
        f"Invalid datetime in {csv_path} at row {row_number}, "
        f"column {column_name!r}: {value!r}"
    )


def ingest_csv_files(
    source_dir: Path = EXTERNAL_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "dataset.parquet",
    *,
    csv_paths: Iterable[Path] | None = None,
) -> Path:
    """@brief Normalize and combine trip CSV files into a Parquet dataset.

    The union of all CSV headers defines the output schema. Column names are
    normalized to lowercase ASCII ``snake_case``; decoded text values are
    normalized to Unicode NFC. ``inicio_del_viaje`` and ``fin_del_viaje`` are
    written as ``timestamp[us]``; remaining columns are written as strings.
    Rows are processed in bounded batches, and the final output replaces the
    target only after a complete successful write. MiBici nomenclature files
    are excluded from the trip dataset.

    @param source_dir Directory containing the source CSV files.
    @param output_path Destination path for the Parquet dataset.
    @param csv_paths Optional explicit iterable of CSV files to ingest. When
        omitted, eligible CSVs are discovered directly in ``source_dir``.
    @return Path to the completed Parquet dataset.
    @throws FileNotFoundError If the source directory does not exist or no
        eligible trip CSV files are found.
    @throws ValueError If a CSV has no header, contains duplicate normalized
        column names, or contains an invalid date value.
    @throws OSError If a source file cannot be read or the Parquet output cannot
        be written.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"CSV source directory does not exist: {source_dir}")

    if csv_paths is None:
        selected_paths = sorted(
            path
            for path in source_dir.glob("*.csv")
            if not NOMENCLATURE_FILENAME_RE.fullmatch(path.name)
        )
    else:
        selected_paths = sorted(
            path
            for path in csv_paths
            if path.suffix.lower() == ".csv" and not NOMENCLATURE_FILENAME_RE.fullmatch(path.name)
        )

    if not selected_paths:
        raise FileNotFoundError(f"No CSV files found in: {source_dir}")

    encodings = {csv_path: _get_csv_encoding(csv_path) for csv_path in selected_paths}
    fieldnames: list[str] = []
    column_mappings: dict[Path, dict[str, str]] = {}
    for csv_path in selected_paths:
        with csv_path.open("r", encoding=encodings[csv_path], newline="") as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError(f"CSV file has no header: {csv_path}") from error
            if not header:
                raise ValueError(f"CSV file has an empty header: {csv_path}")
            normalized_header = [_normalize_column_name(fieldname) for fieldname in header]
            if len(normalized_header) != len(set(normalized_header)):
                raise ValueError(
                    f"CSV file has duplicate column names after normalization: {csv_path}"
                )
            column_mappings[csv_path] = dict(zip(header, normalized_header))
            for fieldname in normalized_header:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)

    schema = pa.schema(
        [
            (fieldname, pa.timestamp("us") if fieldname in DATE_COLUMNS else pa.string())
            for fieldname in fieldnames
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_output_path = Path(temporary_file.name)
    row_count = 0

    try:
        with pq.ParquetWriter(temporary_output_path, schema=schema) as writer:
            for csv_path in selected_paths:
                with csv_path.open(
                    "r",
                    encoding=encodings[csv_path],
                    newline="",
                ) as input_file:
                    reader = csv.DictReader(input_file)
                    batch: list[dict[str, str | datetime | None]] = []
                    mapping = column_mappings[csv_path]
                    for row_number, row in enumerate(reader, start=2):
                        normalized_row: dict[str, str | datetime | None] = {
                            fieldname: None if fieldname in DATE_COLUMNS else ""
                            for fieldname in fieldnames
                        }
                        for original_name, normalized_name in mapping.items():
                            value = row.get(original_name) or ""
                            if normalized_name in DATE_COLUMNS:
                                normalized_row[normalized_name] = _parse_date(
                                    value,
                                    csv_path,
                                    row_number,
                                    normalized_name,
                                )
                            else:
                                normalized_row[normalized_name] = _normalize_text(value)
                        batch.append(normalized_row)
                        if len(batch) == PARQUET_BATCH_SIZE:
                            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                            row_count += len(batch)
                            batch.clear()

                    if batch:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        row_count += len(batch)
        temporary_output_path.replace(output_path)
    except Exception:
        temporary_output_path.unlink(missing_ok=True)
        raise

    logger.success(
        "Ingested {} rows from {} CSV files into {}",
        row_count,
        len(selected_paths),
        output_path,
    )
    return output_path



@app.command()
def main(
    source_dir: Path = EXTERNAL_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "dataset.parquet",
) -> None:
    """Run the complete MiBici download and ingestion workflow.

    @param source_dir Directory for downloaded CSVs and CSV ingestion.
    @param output_path Destination Parquet path for the combined trip dataset.
    """

    downloaded_paths = download_recent_data(source_dir)

    monthly_paths = [
        path
        for path in downloaded_paths
        if MONTHLY_FILENAME_RE.fullmatch(path.name)
    ]

    ingest_csv_files(
        source_dir,
        output_path,
        csv_paths=monthly_paths,
    )

    # Generar manifiesto para versionar los datos
    from .data_version import create_data_manifest

    create_data_manifest(
        source_files=downloaded_paths,
        parquet_path=output_path,
    )


if __name__ == "__main__":
    app()
