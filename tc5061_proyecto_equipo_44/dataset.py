import codecs
import csv
from pathlib import Path

from loguru import logger
import typer

from tc5061_proyecto_equipo_44.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def _get_csv_encoding(csv_path: Path) -> str:
    with csv_path.open("rb") as file:
        sample = file.read(65536)

    try:
        codecs.getincrementaldecoder("utf-8-sig")().decode(sample, final=False)
    except UnicodeDecodeError:
        return "latin-1"
    return "utf-8-sig"


def ingest_csv_files(
    source_dir: Path = EXTERNAL_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
) -> Path:
    """Combine every CSV in source_dir into one dataset, aligning differing columns."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"CSV source directory does not exist: {source_dir}")

    csv_paths = sorted(source_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {source_dir}")

    encodings = {csv_path: _get_csv_encoding(csv_path) for csv_path in csv_paths}
    fieldnames: list[str] = []
    for csv_path in csv_paths:
        with csv_path.open("r", encoding=encodings[csv_path], newline="") as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError(f"CSV file has no header: {csv_path}") from error
            if not header:
                raise ValueError(f"CSV file has an empty header: {csv_path}")
            for fieldname in header:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for csv_path in csv_paths:
            with csv_path.open("r", encoding=encodings[csv_path], newline="") as input_file:
                reader = csv.DictReader(input_file)
                for row in reader:
                    writer.writerow(
                        {
                            fieldname: row.get(fieldname) or ""
                            for fieldname in fieldnames
                        }
                    )
                    row_count += 1

    logger.success(
        "Ingested {} rows from {} CSV files into {}",
        row_count,
        len(csv_paths),
        output_path,
    )
    return output_path


@app.command()
def main(
    source_dir: Path = EXTERNAL_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
) -> None:
    ingest_csv_files(source_dir, output_path)


if __name__ == "__main__":
    app()
