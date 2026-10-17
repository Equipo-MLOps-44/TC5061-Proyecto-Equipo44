from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from tc5061_proyecto_equipo_44.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR
import pyarrow.parquet as pq
app = typer.Typer()


def aggregate_hourly_trips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of trips starting at each station per hour.

    Useful for P1 demand prediction and as a base
    for other feature engineering tasks.
    """

    # 1. Select the necessary columns
    trips = df[["Origen_Id", "Inicio_del_viaje"]].copy()

    # 2. Convert station IDs to numbers
    trips["Origen_Id"] = pd.to_numeric(
        trips["Origen_Id"], errors="coerce"
    )

    # 3. Convert dates to datetime
    trips["Inicio_del_viaje"] = pd.to_datetime(
        trips["Inicio_del_viaje"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    # 4. Remove invalid records
    trips = trips.dropna(
        subset=["Origen_Id", "Inicio_del_viaje"]
    )

    # 5. Group timestamps by hour
    trips["hour"] = trips["Inicio_del_viaje"].dt.floor("h")

    # 6. Count departures per station and hour
    hourly = (
        trips.groupby(["Origen_Id", "hour"])
        .size()
        .reset_index(name="departures")
    )

    hourly["Origen_Id"] = hourly["Origen_Id"].astype(int)

    return hourly.sort_values(
        ["Origen_Id", "hour"]
    ).reset_index(drop=True)


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from hourly trip data."""

    features = df.copy()

    # Convert hour column to datetime
    features["hour"] = pd.to_datetime(features["hour"])

    # Extract time-related features
    features["hour_of_day"] = features["hour"].dt.hour
    features["day_of_week"] = features["hour"].dt.dayofweek
    features["month"] = features["hour"].dt.month

    # Identify weekends (Saturday=5, Sunday=6)
    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype(int)

    return features

def complete_hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing hours with zero trips for each station."""

    results = []

    for station_id, group in df.groupby("Origen_Id"):

        # Organize trips chronologically
        hourly = group.set_index("hour")["departures"].sort_index()

        # Create every hour between the first and last record
        all_hours = pd.date_range(
            start=hourly.index.min(),
            end=hourly.index.max(),
            freq="h"
        )

        # Add hours with no recorded trips
        hourly = hourly.reindex(all_hours, fill_value=0)

        results.append(pd.DataFrame({
            "Origen_Id": station_id,
            "hour": all_hours,
            "departures": hourly.to_numpy()
        }))

    if not results:
        return pd.DataFrame(
            columns=["Origen_Id", "hour", "departures"]
        )

    return pd.concat(results, ignore_index=True)


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create historical demand features."""

    features = df.sort_values(
        ["Origen_Id", "hour"]
    ).copy()

    # Separate historical demand by station
    grouped = features.groupby("Origen_Id")["departures"]

    # Previous hour, previous day and previous week
    for lag in [1, 24, 168]:
        features[f"departures_lag_{lag}h"] = grouped.shift(lag)

    return features


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical rolling averages for each station.

    Uses only previous observations to prevent data leakage.
    Useful for demand prediction (P1) and risk classification (P2).
    """

    features = df.sort_values(
        ["Origen_Id", "hour"]
    ).copy()

    # Calculate the average demand of the previous 24 hours
    features["departures_rolling_mean_24h"] = (
        features.groupby("Origen_Id")["departures"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=24,
                min_periods=1
            ).mean()
        )
    )

    return features


def add_station_features(
    df: pd.DataFrame,
    stations: pd.DataFrame
) -> pd.DataFrame:
    """
    Add station capacity and geographic information.

    Match trip origin stations with station nomenclature.
    Useful for P1, P2, and P3.
    """

    features = df.copy()
    station_info = stations[
        ["id", "dpcapacity", "latitude", "longitude"]
    ].copy()

    # Convert station IDs to numeric values
    features["Origen_Id"] = pd.to_numeric(
        features["Origen_Id"], errors="coerce"
    )

    station_info["id"] = pd.to_numeric(
        station_info["id"], errors="coerce"
    )

    # Remove invalid and duplicate station IDs
    station_info = station_info.dropna(subset=["id"])
    station_info = station_info.drop_duplicates(subset=["id"])

    # Match trips with station information
    features = features.merge(
        station_info,
        left_on="Origen_Id",
        right_on="id",
        how="left",
        validate="many_to_one"
    )

    # Remove duplicated ID column
    features = features.drop(columns=["id"])

    return features


def create_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate hourly arrivals, departures, and net bicycle flow.

    Useful for estimating station imbalance risk (P2).
    Net flow = arrivals - departures.
    """

    # Count departures using our existing function
    departures = aggregate_hourly_trips(df)

    # Select destination information
    arrivals = df[["Destino_Id", "Fin_del_viaje"]].copy()

    # Convert station IDs and dates
    arrivals["Destino_Id"] = pd.to_numeric(
        arrivals["Destino_Id"], errors="coerce"
    )

    arrivals["Fin_del_viaje"] = pd.to_datetime(
        arrivals["Fin_del_viaje"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    # Remove invalid records
    arrivals = arrivals.dropna(
        subset=["Destino_Id", "Fin_del_viaje"]
    )

    # Group arrivals by hour
    arrivals["hour"] = arrivals["Fin_del_viaje"].dt.floor("h")

    arrivals = (
        arrivals.groupby(["Destino_Id", "hour"])
        .size()
        .reset_index(name="arrivals")
        .rename(columns={"Destino_Id": "Origen_Id"})
    )

    # Combine arrivals and departures by station and hour
    flow = departures.merge(
        arrivals,
        on=["Origen_Id", "hour"],
        how="outer"
    )

    # Replace missing counts with zero
    flow["departures"] = flow["departures"].fillna(0).astype(int)
    flow["arrivals"] = flow["arrivals"].fillna(0).astype(int)

    # Calculate net bicycle flow
    flow["net_flow"] = flow["arrivals"] - flow["departures"]

    flow["Origen_Id"] = flow["Origen_Id"].astype(int)

    return flow.sort_values(
        ["Origen_Id", "hour"]
    ).reset_index(drop=True)


def estimate_station_inventory(
    df: pd.DataFrame,
    initial_occupancy: float = 0.75,
    risk_threshold: float = 0.20,
) -> pd.DataFrame:
    """Estimate hourly bike inventory and derive station-risk labels.

    The first observed hour for each station starts at 75% of capacity.
    Inventory is updated as previous inventory - departures + arrivals,
    and constrained to [0, capacity]. Risk labels are proxy estimates, not
    observed availability, because actual station inventory is unavailable.
    """
    if not 0 <= initial_occupancy <= 1:
        raise ValueError("initial_occupancy must be between 0 and 1")
    if not 0 <= risk_threshold <= 1:
        raise ValueError("risk_threshold must be between 0 and 1")

    result = df.sort_values(["Origen_Id", "hour"]).copy()
    result["dpcapacity"] = pd.to_numeric(
        result["dpcapacity"], errors="coerce"
    )
    result["departures"] = pd.to_numeric(
        result["departures"], errors="coerce"
    ).fillna(0)
    result["arrivals"] = pd.to_numeric(
        result["arrivals"], errors="coerce"
    ).fillna(0)

    result["estimated_bikes_available"] = float("nan")

    for _, indices in result.groupby("Origen_Id", sort=False).groups.items():
        station = result.loc[indices].sort_values("hour")
        capacities = station["dpcapacity"].dropna()
        if capacities.empty or capacities.iloc[0] <= 0:
            continue

        capacity = float(capacities.iloc[0])
        bikes = round(capacity * initial_occupancy)
        estimates = []

        for row in station.itertuples():
            # Update from the observed flows for this hour.
            bikes = bikes - float(row.departures) + float(row.arrivals)
            bikes = min(max(bikes, 0.0), capacity)
            estimates.append(bikes)

        result.loc[station.index, "estimated_bikes_available"] = estimates

    result["estimated_free_docks"] = (
        result["dpcapacity"] - result["estimated_bikes_available"]
    )
    result["risk_no_bikes"] = (
        result["estimated_bikes_available"]
        <= result["dpcapacity"] * risk_threshold
    ).astype("Int64")
    result["risk_no_docks"] = (
        result["estimated_free_docks"]
        <= result["dpcapacity"] * risk_threshold
    ).astype("Int64")

    # Keep risk labels missing when station capacity/inventory is unknown.
    invalid = result["dpcapacity"].isna() | (result["dpcapacity"] <= 0)
    result.loc[invalid, ["risk_no_bikes", "risk_no_docks"]] = pd.NA

    return result


def create_station_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create station usage profiles for clustering (P3).

    Uses historical arrivals, departures and usage patterns.
    """

    profiles = df[
        ["Origen_Id", "hour", "departures", "arrivals"]
    ].copy()

    # Convert hour to datetime
    profiles["hour"] = pd.to_datetime(profiles["hour"])

    # Identify morning hours and weekends
    profiles["morning_departures"] = profiles["departures"].where(
        profiles["hour"].dt.hour.between(7, 9), 0
    )

    profiles["weekend_departures"] = profiles["departures"].where(
        profiles["hour"].dt.dayofweek >= 5, 0
    )

    # Aggregate historical activity by station
    station_profiles = (
        profiles.groupby("Origen_Id")
        .agg(
            total_departures=("departures", "sum"),
            total_arrivals=("arrivals", "sum"),
            morning_departures=("morning_departures", "sum"),
            weekend_departures=("weekend_departures", "sum")
        )
        .reset_index()
    )

    # Average demand per day in the observed period
    num_days = (
        profiles["hour"].max().normalize()
        - profiles["hour"].min().normalize()
    ).days + 1

    station_profiles["avg_daily_departures"] = (
        station_profiles["total_departures"] / num_days
    )

    # Calculate proportions of morning and weekend usage
    total = station_profiles["total_departures"].replace(0, 1)

    station_profiles["morning_share"] = (
        station_profiles["morning_departures"] / total
    )

    station_profiles["weekend_share"] = (
        station_profiles["weekend_departures"] / total
    )

    return station_profiles



def build_feature_datasets_from_flow(
    flow: pd.DataFrame,
    stations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build P1, P2 and P3 features from hourly aggregated flows.

    Avoids processing individual trips again.
    """

    # Complete hours without recorded departures
    hourly = complete_hourly_series(
        flow[["Origen_Id", "hour", "departures"]]
    )

    # Add arrivals for each station and hour
    hourly = hourly.merge(
        flow[["Origen_Id", "hour", "arrivals"]],
        on=["Origen_Id", "hour"],
        how="left",
        validate="one_to_one"
    )

    hourly["arrivals"] = (
        hourly["arrivals"].fillna(0).astype(int)
    )

    hourly["net_flow"] = (
        hourly["arrivals"] - hourly["departures"]
    )

    # Temporal and historical features
    hourly = create_time_features(hourly)
    hourly = create_lag_features(hourly)
    hourly = create_rolling_features(hourly)

    # Historical flow features for P2
    grouped = hourly.groupby("Origen_Id")

    hourly["arrivals_lag_1h"] = (
        grouped["arrivals"].shift(1)
    )

    hourly["net_flow_lag_1h"] = (
        grouped["net_flow"].shift(1)
    )

    # Station information
    hourly = add_station_features(hourly, stations)

    # P2: estimate bicycle inventory and derive both risk indicators.
    hourly = estimate_station_inventory(
        hourly, initial_occupancy=0.75, risk_threshold=0.20
    )

    # Historical station profiles for P3
    profiles = create_station_profiles(hourly)
    profiles = add_station_features(profiles, stations)

    return hourly, profiles


def build_feature_datasets(
    trips: pd.DataFrame,
    stations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features directly from individual trips."""

    flow = create_flow_features(trips)

    return build_feature_datasets_from_flow(
        flow, stations
    )



from collections import defaultdict


def aggregate_flow_chunks(chunks) -> pd.DataFrame:
    """
    Aggregate station arrivals and departures across data chunks.

    Keeps hourly counts instead of storing all individual trips.
    Useful for processing large MiBici datasets.
    """

    counts = defaultdict(lambda: [0, 0])

    for chunk in chunks:
        flow = create_flow_features(chunk)

        for row in flow.itertuples(index=False):
            key = (int(row.Origen_Id), row.hour)

            counts[key][0] += int(row.departures)
            counts[key][1] += int(row.arrivals)

    records = [
        (station, hour, values[0], values[1])
        for (station, hour), values in counts.items()
    ]

    result = pd.DataFrame(
        records,
        columns=["Origen_Id", "hour", "departures", "arrivals"]
    )

    if result.empty:
        result["net_flow"] = pd.Series(dtype="int64")
        return result

    result["net_flow"] = (
        result["arrivals"] - result["departures"]
    )

    return result.sort_values(
        ["Origen_Id", "hour"]
    ).reset_index(drop=True)
def find_nomenclature_file(directory: Path) -> Path:
    """Find the most recent MiBici station nomenclature CSV."""

    files = sorted(directory.glob("nomenclatura_????_??.csv"))

    if not files:
        raise FileNotFoundError(
            f"No nomenclature CSV found in {directory}"
        )

    return files[-1]


def read_trip_batches(
    path: Path,
    batch_size: int = 100_000
):
    """
    Read MiBici Parquet data in batches.

    Only loads the columns needed for feature engineering,
    reducing memory usage.
    """

    parquet_file = pq.ParquetFile(path)

    columns = [
        "origen_id",
        "destino_id",
        "inicio_del_viaje",
        "fin_del_viaje"
    ]

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=columns
    ):
        df = batch.to_pandas()

        # Match feature engineering column names
        df = df.rename(columns={
            "origen_id": "Origen_Id",
            "destino_id": "Destino_Id",
            "inicio_del_viaje": "Inicio_del_viaje",
            "fin_del_viaje": "Fin_del_viaje"
        })

        yield df

def read_dataset(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet dataset."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")

    raise ValueError(f"Unsupported file format: {path.suffix}")



@app.command()
def main(
    trips_path: Path = PROCESSED_DATA_DIR / "dataset.parquet",
    stations_path: Path | None = None,
    hourly_output: Path = PROCESSED_DATA_DIR / "hourly_features.parquet",
    profiles_output: Path = PROCESSED_DATA_DIR / "station_profiles.parquet",
) -> None:
    """Generate MiBici features for P1, P2 and P3."""

    logger.info("Starting Feature Engineering...")

    # Find station nomenclature
    if stations_path is None:
        stations_path = find_nomenclature_file(EXTERNAL_DATA_DIR)

    stations = read_dataset(stations_path)

    # Process large Parquet datasets in batches
    if trips_path.suffix.lower() == ".parquet":

        logger.info("Reading trips in batches...")

        batches = read_trip_batches(
            trips_path,
            batch_size=100_000
        )

        # Aggregate arrivals and departures
        flow = aggregate_flow_chunks(batches)

        logger.info("Building features from hourly flows...")

        hourly, profiles = build_feature_datasets_from_flow(
            flow, stations
        )

    # Maintain CSV support for small datasets and tests
    elif trips_path.suffix.lower() == ".csv":

        trips = read_dataset(trips_path)

        trips = trips.rename(columns={
            "origen_id": "Origen_Id",
            "destino_id": "Destino_Id",
            "inicio_del_viaje": "Inicio_del_viaje",
            "fin_del_viaje": "Fin_del_viaje"
        })

        hourly, profiles = build_feature_datasets(
            trips, stations
        )

    else:
        raise ValueError(
            f"Unsupported trip format: {trips_path.suffix}"
        )

    # Create output directories
    hourly_output.parent.mkdir(parents=True, exist_ok=True)
    profiles_output.parent.mkdir(parents=True, exist_ok=True)

    # Save datasets
    hourly.to_parquet(hourly_output, index=False)
    profiles.to_parquet(profiles_output, index=False)

    logger.success(
        "Generated {} hourly records",
        len(hourly)
    )

    logger.success(
        "Generated {} station profiles",
        len(profiles)
    )


if __name__ == "__main__":
    app()
