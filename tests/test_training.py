"""Tests for the P2 training pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo_44.modeling import p1_trainer
from tc5061_proyecto_equipo_44.modeling import p2_trainer


def _hourly_p1_training_data(n_rows: int = 40) -> pd.DataFrame:
    """Create deterministic, time-ordered data for P1 tests."""
    hours = pd.date_range("2025-01-01", periods=n_rows, freq="h")
    return pd.DataFrame(
        {
            "hour": hours,
            "Origen_Id": np.tile([2, 3, 4, 5], n_rows // 4),
            "departures": (np.arange(n_rows) % 9).astype(float),
            "departures_lag_1h": np.arange(n_rows, dtype=float),
            "departures_lag_24h": np.arange(n_rows, dtype=float) % 7,
            "departures_lag_168h": np.arange(n_rows, dtype=float) % 5,
            "departures_rolling_mean_24h": np.arange(n_rows, dtype=float) / 2,
            "hour_of_day": hours.hour,
            "day_of_week": hours.dayofweek,
            "month": hours.month,
            "is_weekend": (hours.dayofweek >= 5).astype(int),
        }
    )


@pytest.fixture
def isolated_p1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Redirect P1 data and artifacts to temporary paths and disable MLflow."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    models_dir.mkdir()
    reports_dir.mkdir()

    hourly_path = processed_dir / "hourly_features.parquet"
    manifest_path = tmp_path / "data_manifest.json"
    manifest_path.write_text('{"version": "test"}', encoding="utf-8")

    monkeypatch.setattr(p1_trainer, "HOURLY_PATH", hourly_path)
    monkeypatch.setattr(p1_trainer, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(p1_trainer, "MODELS_DIR", models_dir)
    monkeypatch.setattr(p1_trainer, "REPORTS_DIR", reports_dir)

    df = _hourly_p1_training_data()
    monkeypatch.setattr(p1_trainer.pd, "read_parquet", lambda _path: df.copy())

    monkeypatch.setattr(p1_trainer, "log_mlflow_run", lambda **_kwargs: None)
    monkeypatch.setattr(
        p1_trainer.mlflow.data,
        "from_pandas",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(p1_trainer, "dataset_metadata", lambda _path: {})

    return df, models_dir, reports_dir


def test_train_p1_returns_metrics_and_artifacts(isolated_p1):
    """A valid P1 baseline run should return metrics and save its artifacts."""
    df, models_dir, reports_dir = isolated_p1

    results = p1_trainer.train_p1(model_name="baseline", random_state=0)

    assert results["task"] == "P1_regression"
    assert results["target"] == "departures"
    assert results["model"] == "baseline"
    assert results["split"] == "chronological"
    assert results["train_rows"] + results["test_rows"] == len(df)
    assert results["X_train_shape"] == [results["train_rows"], 9]
    assert results["X_test_shape"] == [results["test_rows"], 9]
    assert results["mae"] >= 0.0
    assert results["rmse"] >= 0.0

    assert (models_dir / "p1_mean_baseline.joblib").is_file()
    assert (reports_dir / "p1_baseline_config.json").is_file()
    assert (reports_dir / "figures" / "p1_baseline_actual_vs_predicted.png").is_file()
    assert (reports_dir / "figures" / "p1_baseline_residuals.png").is_file()


def test_train_p1_rejects_unknown_model(isolated_p1):
    """Unsupported P1 model names should raise ValueError."""
    with pytest.raises(ValueError, match="P1 model must be 'baseline', 'rf', or 'search'"):
        p1_trainer.train_p1(model_name="invalid")


def _hourly_training_data(n_rows: int = 40,include_targets: bool = True,) -> pd.DataFrame:
    """Create deterministic, time-ordered data for P2 tests."""
    hours = pd.date_range("2025-01-01", periods=n_rows, freq="h")
    data = {
        "hour": hours,
        "Origen_Id": np.tile([2, 3, 4, 5], n_rows // 4),
        "hour_of_day": hours.hour,
        "day_of_week": hours.dayofweek,
        "month": hours.month,
        "is_weekend": (hours.dayofweek >= 5).astype(int),
        "departures_lag_1h": np.arange(n_rows, dtype=float),
        "departures_lag_24h": np.arange(n_rows, dtype=float) % 7,
        "dpcapacity": np.tile([15, 18, 20, 22], n_rows // 4),
    }
    if include_targets:
        # Both classes occur throughout the time series, including train/test.
        data["risk_no_bikes"] = (np.arange(n_rows) % 2).astype(int)
        data["risk_no_docks"] = ((np.arange(n_rows) // 2) % 2).astype(int)
    return pd.DataFrame(data)


@pytest.fixture
def isolated_p2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Redirect data and artifacts to temporary paths and disable MLflow logging."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    models_dir.mkdir()
    reports_dir.mkdir()

    monkeypatch.setattr(p2_trainer,"HOURLY_PATH",processed_dir / "hourly_features.parquet",)
    monkeypatch.setattr(p2_trainer, "MODELS_DIR", models_dir)
    monkeypatch.setattr(p2_trainer, "REPORTS_DIR", reports_dir)

    df = _hourly_training_data()
    monkeypatch.setattr(p2_trainer.pd, "read_parquet", lambda _path: df.copy())

    # Prevent real MLflow runs and dataset logging.
    monkeypatch.setattr(p2_trainer, "log_mlflow_run", lambda **_kwargs: None)
    monkeypatch.setattr(p2_trainer.mlflow.data,"from_pandas",lambda *_args, **_kwargs: object(),)
    monkeypatch.setattr(p2_trainer, "dataset_metadata", lambda _path: {})

    return df, models_dir, reports_dir


def test_train_p2_returns_metrics_for_both_targets(isolated_p2):
    """A valid training run should return metrics for both risk targets."""
    _df, models_dir, reports_dir = isolated_p2

    results = p2_trainer.train_p2(model_name="baseline",random_state=0,)

    assert results is not None
    assert set(results) == {"risk_no_bikes", "risk_no_docks"}

    for target_metrics in results.values():
        assert 0.0 <= target_metrics["accuracy"] <= 1.0
        assert 0.0 <= target_metrics["macro_f1"] <= 1.0
        assert 0.0 <= target_metrics["macro_recall"] <= 1.0
        assert target_metrics["train_rows"] > 0
        assert target_metrics["test_rows"] > 0

    assert (models_dir / "p2_no_bikes_baseline.joblib").is_file()
    assert (models_dir / "p2_no_docks_baseline.joblib").is_file()
    assert list((reports_dir / "p2").rglob("config.json"))


def test_train_p2_is_reproducible(isolated_p2):
    """Repeated runs with the same data and random_state should match."""
    first = p2_trainer.train_p2(model_name="rf", n_estimators=20, random_state=0)
    second = p2_trainer.train_p2(model_name="rf", n_estimators=20, random_state=0)

    assert first is not None
    assert second is not None
    assert first.keys() == second.keys()

    for target in first:
        for metric in ("accuracy", "macro_f1", "macro_recall", "confusion_matrix"):
            assert first[target][metric] == second[target][metric]


def test_train_p2_returns_none_when_targets_are_missing(monkeypatch: pytest.MonkeyPatch):
    """P2 should be skipped when neither risk target exists."""
    df = _hourly_training_data(include_targets=False)
    monkeypatch.setattr(p2_trainer.pd, "read_parquet", lambda _path: df.copy())

    assert p2_trainer.train_p2() is None


def test_train_p2_rejects_unknown_model(isolated_p2):
    """Unsupported model names should raise ValueError."""
    with pytest.raises(ValueError, match="P2 model must be 'baseline' or 'rf'"):
        p2_trainer.train_p2(model_name="invalid")
