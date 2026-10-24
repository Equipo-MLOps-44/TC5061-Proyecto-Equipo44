import pandas as pd
import numpy as np
from pathlib import Path

TEST_SIZE = 0.30

def dataset_metadata(path: Path) -> dict:
    """Collect basic metadata about the input dataset."""
    stat = path.stat()

    return {
        "dataset_file": path.name,
        "dataset_size_bytes": stat.st_size,
        "dataset_modified_time": pd.Timestamp(
            stat.st_mtime, unit="s"
        ).isoformat(),
    }

def chronological_split(
    df: pd.DataFrame,
    timestamp_column: str = "hour",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split records chronologically, keeping timestamps in one partition."""
    unique_times = np.sort(df[timestamp_column].dropna().unique())

    if len(unique_times) < 2:
        raise ValueError(
            "At least two distinct timestamps are required."
        )

    split_index = int(len(unique_times) * (1 - TEST_SIZE))
    split_index = min(max(split_index, 1), len(unique_times) - 1)
    split_time = pd.Timestamp(unique_times[split_index])

    train = df[df[timestamp_column] < split_time].copy()
    test = df[df[timestamp_column] >= split_time].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Chronological split produced an empty partition."
        )

    return train, test, split_time