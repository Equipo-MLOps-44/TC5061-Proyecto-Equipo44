
import pandas as pd

from tc5061_proyecto_equipo_44.features import aggregate_hourly_trips


def test_aggregate_hourly_trips():

    # Create sample trip data
    df = pd.DataFrame({
        "Origen_Id": [2, 2, 2, 3],
        "Inicio_del_viaje": [
            "2025-01-01 08:10:00",
            "2025-01-01 08:25:00",
            "2025-01-01 09:15:00",
            "2025-01-01 08:40:00"
        ]
    })

    # Run feature engineering function
    result = aggregate_hourly_trips(df)

    # Verify the results
    assert len(result) == 3
    assert result["departures"].sum() == 4

    station_2 = result[result["Origen_Id"] == 2]

    assert station_2["departures"].tolist() == [2, 1]

from tc5061_proyecto_equipo_44.features import create_time_features


def test_create_time_features():
    df = pd.DataFrame({
        "Origen_Id": [2],
        "hour": ["2025-01-04 08:00:00"],
        "departures": [10]
    })

    result = create_time_features(df)

    assert result["hour_of_day"].iloc[0] == 8
    assert result["day_of_week"].iloc[0] == 5
    assert result["month"].iloc[0] == 1
    assert result["is_weekend"].iloc[0] == 1


from tc5061_proyecto_equipo_44.features import (
    complete_hourly_series,
    create_lag_features,
)


def test_create_lag_features():

    df = pd.DataFrame({
        "Origen_Id": [2, 2],
        "hour": [
            "2025-01-01 08:00:00",
            "2025-01-01 10:00:00"
        ],
        "departures": [5, 7]
    })

    df["hour"] = pd.to_datetime(df["hour"])

    # Complete the missing 9:00 hour
    complete = complete_hourly_series(df)

    # Generate historical features
    result = create_lag_features(complete)

    assert result["departures"].tolist() == [5, 0, 7]

    # At 10:00, the previous hour had zero trips
    assert result["departures_lag_1h"].iloc[2] == 0

    # The first hour has no historical information
    assert pd.isna(result["departures_lag_1h"].iloc[0])


from tc5061_proyecto_equipo_44.features import create_rolling_features


def test_create_rolling_features():

    df = pd.DataFrame({
        "Origen_Id": [2, 2, 2, 3],
        "hour": pd.to_datetime([
            "2025-01-01 08:00:00",
            "2025-01-01 09:00:00",
            "2025-01-01 10:00:00",
            "2025-01-01 08:00:00"
        ]),
        "departures": [10, 20, 30, 100]
    })

    result = create_rolling_features(df)

    station_2 = result[result["Origen_Id"] == 2]
    station_3 = result[result["Origen_Id"] == 3]

    # First observation has no historical data
    assert pd.isna(
        station_2["departures_rolling_mean_24h"].iloc[0]
    )

    # At 9:00, only the 8:00 value is available
    assert station_2["departures_rolling_mean_24h"].iloc[1] == 10

    # At 10:00, average of 10 and 20
    assert station_2["departures_rolling_mean_24h"].iloc[2] == 15

    # Station 3 must not use station 2's history
    assert pd.isna(
        station_3["departures_rolling_mean_24h"].iloc[0]
    )

from tc5061_proyecto_equipo_44.features import add_station_features


def test_add_station_features():

    # Sample hourly trip data
    trips = pd.DataFrame({
        "Origen_Id": [2, 3],
        "departures": [10, 20]
    })

    # Sample station nomenclature
    stations = pd.DataFrame({
        "id": [2, 3],
        "dpcapacity": [15, 19],
        "latitude": [20.666378, 20.667690],
        "longitude": [-103.348820, -103.368252]
    })

    result = add_station_features(trips, stations)

    assert result["dpcapacity"].tolist() == [15, 19]
    assert result["latitude"].iloc[0] == 20.666378
    assert result["longitude"].iloc[1] == -103.368252
    assert len(result) == 2

from tc5061_proyecto_equipo_44.features import create_flow_features


def test_create_flow_features():

    df = pd.DataFrame({
        "Origen_Id": [2, 2, 3],
        "Destino_Id": [3, 3, 2],
        "Inicio_del_viaje": [
            "2025-01-01 08:10:00",
            "2025-01-01 08:15:00",
            "2025-01-01 09:05:00"
        ],
        "Fin_del_viaje": [
            "2025-01-01 08:20:00",
            "2025-01-01 08:30:00",
            "2025-01-01 09:20:00"
        ]
    })

    result = create_flow_features(df)

    station_2 = result[
        (result["Origen_Id"] == 2) &
        (result["hour"].dt.hour == 8)
    ]

    assert station_2["departures"].iloc[0] == 2
    assert station_2["arrivals"].iloc[0] == 0
    assert station_2["net_flow"].iloc[0] == -2

    station_3 = result[
        (result["Origen_Id"] == 3) &
        (result["hour"].dt.hour == 8)
    ]

    assert station_3["net_flow"].iloc[0] == 2

from tc5061_proyecto_equipo_44.features import create_station_profiles


def test_create_station_profiles():

    df = pd.DataFrame({
        "Origen_Id": [2, 2, 3],
        "hour": [
            "2025-01-04 08:00:00",
            "2025-01-04 18:00:00",
            "2025-01-05 12:00:00"
        ],
        "departures": [10, 5, 20],
        "arrivals": [5, 8, 15]
    })

    result = create_station_profiles(df)

    station_2 = result[result["Origen_Id"] == 2].iloc[0]
    station_3 = result[result["Origen_Id"] == 3].iloc[0]

    assert station_2["total_departures"] == 15
    assert station_2["total_arrivals"] == 13

    assert station_2["morning_share"] == 10 / 15
    assert station_2["weekend_share"] == 1

    assert station_3["total_departures"] == 20
    assert station_3["morning_share"] == 0

from tc5061_proyecto_equipo_44.features import build_feature_datasets


def test_build_feature_datasets():

    trips = pd.DataFrame({
        "Origen_Id": [2, 2],
        "Destino_Id": [3, 3],
        "Inicio_del_viaje": [
            "2025-01-01 08:10:00",
            "2025-01-01 09:10:00"
        ],
        "Fin_del_viaje": [
            "2025-01-01 08:20:00",
            "2025-01-01 09:20:00"
        ]
    })

    stations = pd.DataFrame({
        "id": [2, 3],
        "dpcapacity": [15, 19],
        "latitude": [20.666378, 20.667690],
        "longitude": [-103.348820, -103.368252]
    })

    hourly, profiles = build_feature_datasets(
        trips, stations
    )

    assert len(hourly) == 4
    assert len(profiles) == 2

    assert "departures_lag_1h" in hourly.columns
    assert "net_flow_lag_1h" in hourly.columns
    assert "dpcapacity" in hourly.columns
    assert "morning_share" in profiles.columns

from tc5061_proyecto_equipo_44.features import main


def test_generate_parquet_files(tmp_path):

    # Create sample trip data
    trips = pd.DataFrame({
        "Origen_Id": [2, 2, 3],
        "Destino_Id": [3, 3, 2],
        "Inicio_del_viaje": [
            "2025-01-01 08:10:00",
            "2025-01-01 09:10:00",
            "2025-01-01 09:30:00"
        ],
        "Fin_del_viaje": [
            "2025-01-01 08:25:00",
            "2025-01-01 09:25:00",
            "2025-01-01 09:45:00"
        ]
    })

    # Create sample station information
    stations = pd.DataFrame({
        "id": [2, 3],
        "dpcapacity": [15, 19],
        "latitude": [20.666378, 20.667690],
        "longitude": [-103.348820, -103.368252]
    })

    # Temporary paths for testing
    trips_path = tmp_path / "trips.csv"
    stations_path = tmp_path / "stations.csv"
    hourly_path = tmp_path / "hourly_features.parquet"
    profiles_path = tmp_path / "station_profiles.parquet"

    # Save sample input data
    trips.to_csv(trips_path, index=False)
    stations.to_csv(stations_path, index=False)

    # Execute the complete feature pipeline
    main(
        trips_path=trips_path,
        stations_path=stations_path,
        hourly_output=hourly_path,
        profiles_output=profiles_path
    )

    # Verify that both Parquet files were created
    assert hourly_path.exists()
    assert profiles_path.exists()

    # Read generated files
    hourly = pd.read_parquet(hourly_path)
    profiles = pd.read_parquet(profiles_path)

    # Verify their contents
    assert not hourly.empty
    assert len(profiles) == 2
    assert "departures_lag_1h" in hourly.columns
    assert "net_flow_lag_1h" in hourly.columns
    assert "dpcapacity" in hourly.columns
    assert "morning_share" in profiles.columns

def test_no_future_data_leakage():

    df = pd.DataFrame({
        "Origen_Id": [2, 2, 2],
        "hour": pd.to_datetime([
            "2025-01-01 08:00:00",
            "2025-01-01 09:00:00",
            "2025-01-01 10:00:00"
        ]),
        "departures": [5, 7, 10]
    })

    # Generate historical features
    original = create_lag_features(
        create_rolling_features(df)
    )

    # Change the future value at 10:00
    modified_df = df.copy()
    modified_df.loc[2, "departures"] = 999

    modified = create_lag_features(
        create_rolling_features(modified_df)
    )

    # Historical features at 10:00 must remain unchanged
    assert (
        original["departures_lag_1h"].iloc[2]
        == modified["departures_lag_1h"].iloc[2]
    )

    assert (
        original["departures_rolling_mean_24h"].iloc[2]
        == modified["departures_rolling_mean_24h"].iloc[2]
    )

from tc5061_proyecto_equipo_44.features import aggregate_flow_chunks


def test_aggregate_flow_chunks():

    chunk1 = pd.DataFrame({
        "Origen_Id": [2],
        "Destino_Id": [3],
        "Inicio_del_viaje": ["2025-01-01 08:10:00"],
        "Fin_del_viaje": ["2025-01-01 08:20:00"]
    })

    chunk2 = pd.DataFrame({
        "Origen_Id": [2],
        "Destino_Id": [3],
        "Inicio_del_viaje": ["2025-01-01 08:35:00"],
        "Fin_del_viaje": ["2025-01-01 08:50:00"]
    })

    result = aggregate_flow_chunks([chunk1, chunk2])

    station_2 = result[result["Origen_Id"] == 2].iloc[0]
    station_3 = result[result["Origen_Id"] == 3].iloc[0]

    assert station_2["departures"] == 2
    assert station_2["net_flow"] == -2

    assert station_3["arrivals"] == 2
    assert station_3["net_flow"] == 2

from tc5061_proyecto_equipo_44.features import find_nomenclature_file


def test_find_nomenclature_file(tmp_path):

    # Create example nomenclature files
    (tmp_path / "nomenclatura_2025_01.csv").touch()
    (tmp_path / "nomenclatura_2026_09.csv").touch()

    # Find the most recent file
    result = find_nomenclature_file(tmp_path)

    assert result.name == "nomenclatura_2026_09.csv"


from tc5061_proyecto_equipo_44.features import read_trip_batches


def test_read_trip_batches(tmp_path):

    trips = pd.DataFrame({
        "origen_id": ["2", "2", "3"],
        "destino_id": ["3", "3", "2"],
        "inicio_del_viaje": pd.to_datetime([
            "2025-01-01 08:10:00",
            "2025-01-01 08:20:00",
            "2025-01-01 09:10:00"
        ]),
        "fin_del_viaje": pd.to_datetime([
            "2025-01-01 08:25:00",
            "2025-01-01 08:35:00",
            "2025-01-01 09:25:00"
        ])
    })

    path = tmp_path / "dataset.parquet"
    trips.to_parquet(path, index=False)

    batches = list(read_trip_batches(path, batch_size=2))

    # Three records should produce two batches
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1

    # Verify normalized column names
    assert "Origen_Id" in batches[0].columns
    assert "Inicio_del_viaje" in batches[0].columns

def test_parquet_batch_aggregation(tmp_path):
    """Verify hourly aggregation across Parquet batches."""

    trips = pd.DataFrame({
        "origen_id": ["2", "2", "3", "2"],
        "destino_id": ["3", "3", "2", "3"],
        "inicio_del_viaje": pd.to_datetime([
            "2025-01-01 08:10:00",
            "2025-01-01 08:20:00",
            "2025-01-01 09:10:00",
            "2025-01-01 08:40:00"
        ]),
        "fin_del_viaje": pd.to_datetime([
            "2025-01-01 08:25:00",
            "2025-01-01 08:35:00",
            "2025-01-01 09:25:00",
            "2025-01-01 08:55:00"
        ])
    })

    # Save sample data as Parquet
    path = tmp_path / "dataset.parquet"
    trips.to_parquet(path, index=False)

    # Read in batches of 2 records
    batches = read_trip_batches(path, batch_size=2)

    # Aggregate all batches
    result = aggregate_flow_chunks(batches)

    station_2 = result[
        (result["Origen_Id"] == 2) &
        (result["hour"] == pd.Timestamp("2025-01-01 08:00"))
    ].iloc[0]

    station_3 = result[
        (result["Origen_Id"] == 3) &
        (result["hour"] == pd.Timestamp("2025-01-01 08:00"))
    ].iloc[0]

    assert station_2["departures"] == 3
    assert station_2["arrivals"] == 0
    assert station_2["net_flow"] == -3

    assert station_3["arrivals"] == 3
    assert station_3["net_flow"] == 3

from tc5061_proyecto_equipo_44.features import (
    build_feature_datasets_from_flow,
)


def test_features_from_aggregated_flow():

    trips = pd.DataFrame({
        "Origen_Id": [2, 2],
        "Destino_Id": [3, 3],
        "Inicio_del_viaje": [
            "2025-01-01 08:10:00",
            "2025-01-01 09:10:00"
        ],
        "Fin_del_viaje": [
            "2025-01-01 08:20:00",
            "2025-01-01 09:20:00"
        ]
    })

    stations = pd.DataFrame({
        "id": [2, 3],
        "dpcapacity": [15, 19],
        "latitude": [20.66, 20.67],
        "longitude": [-103.34, -103.36]
    })

    # Original method
    expected_hourly, expected_profiles = (
        build_feature_datasets(trips, stations)
    )

    # New method using aggregated flows
    flow = create_flow_features(trips)

    actual_hourly, actual_profiles = (
        build_feature_datasets_from_flow(flow, stations)
    )

    # Both methods must produce identical results
    pd.testing.assert_frame_equal(
        expected_hourly, actual_hourly
    )

    pd.testing.assert_frame_equal(
        expected_profiles, actual_profiles
    )

from tc5061_proyecto_equipo_44.features import main


def test_main_end_to_end_parquet(tmp_path):
    """Test the complete feature engineering pipeline."""

    # Simulate the ingestion output
    trips = pd.DataFrame({
        "origen_id": [2, 2, 3],
        "destino_id": [3, 3, 2],
        "inicio_del_viaje": pd.to_datetime([
            "2025-01-01 08:10:00",
            "2025-01-01 08:20:00",
            "2025-01-01 09:10:00"
        ]),
        "fin_del_viaje": pd.to_datetime([
            "2025-01-01 08:25:00",
            "2025-01-01 08:35:00",
            "2025-01-01 09:25:00"
        ])
    })

    trips_path = tmp_path / "dataset.parquet"
    trips.to_parquet(trips_path, index=False)

    # Simulate station nomenclature
    stations = pd.DataFrame({
        "id": [2, 3],
        "dpcapacity": [15, 19],
        "latitude": [20.66, 20.67],
        "longitude": [-103.34, -103.36]
    })

    stations_path = tmp_path / "nomenclatura_2025_01.csv"
    stations.to_csv(stations_path, index=False)

    hourly_path = tmp_path / "hourly_features.parquet"
    profiles_path = tmp_path / "station_profiles.parquet"

    # Execute complete pipeline
    main(
        trips_path=trips_path,
        stations_path=stations_path,
        hourly_output=hourly_path,
        profiles_output=profiles_path
    )

    # Verify output files
    assert hourly_path.exists()
    assert profiles_path.exists()

    hourly = pd.read_parquet(hourly_path)
    profiles = pd.read_parquet(profiles_path)

    # Verify that trips were counted correctly
    assert hourly["departures"].sum() == 3
    assert hourly["arrivals"].sum() == 3

    # Verify that features were generated
    assert "net_flow" in hourly.columns
    assert len(profiles) > 0
