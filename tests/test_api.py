import pandas as pd
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api import main


P2_PAYLOAD = {
    "hour_of_day": 14,
    "day_of_week": 0,
    "month": 8,
    "is_weekend": 0,
    "departures_lag_1h": 2.0,
    "departures_lag_24h": 3.0,
    "departures_lag_168h": 4.0,
    "departures_rolling_mean_24h": 3.0,
    "arrivals_lag_1h": 1.0,
    "net_flow_lag_1h": -1.0,
    "dpcapacity": 20.0,
}


class FixedClassifier:
    def __init__(self, prediction: int):
        self.prediction = prediction

    def predict(self, features: pd.DataFrame) -> list[int]:
        assert not features.empty
        return [self.prediction]


def test_p2_predicts_both_risk_targets(monkeypatch: pytest.MonkeyPatch):
    features = ["hour_of_day", "net_flow_lag_1h"]
    artifacts = {
        "features": features,
        "medians": pd.Series({"hour_of_day": 12.0, "net_flow_lag_1h": 0.0}),
    }
    monkeypatch.setattr(
        main,
        "p2_models",
        {
            "risk_no_bikes": {
                **artifacts,
                "model": FixedClassifier(1),
            },
            "risk_no_docks": {
                **artifacts,
                "model": FixedClassifier(0),
            },
        },
    )

    result = main.predict_p2(main.PredictionInputP2(**P2_PAYLOAD))

    assert result.predicted_risk_no_bikes == 1
    assert result.predicted_risk_no_docks == 0


def test_p2_returns_service_unavailable_without_model_artifacts(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(main, "p2_models", {})

    with pytest.raises(HTTPException) as error:
        main.predict_p2(main.PredictionInputP2(**P2_PAYLOAD))

    assert error.value.status_code == 503
    assert "p2_no_bikes_baseline.joblib" in error.value.detail


def test_p2_input_validates_hour_range():
    with pytest.raises(ValidationError):
        main.PredictionInputP2(**{**P2_PAYLOAD, "hour_of_day": 24})
