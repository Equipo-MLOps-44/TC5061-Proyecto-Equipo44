
from pathlib import Path
from typing import Literal
import joblib

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI(
    title="MiBici API - Equipo 44",
    version="1.0.0"
)

# Cargar el modelo descargado desde el Model Registry
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "serving"
model = mlflow.pyfunc.load_model(str(MODEL_PATH))

# Carga de modelo para P3
p3_data = joblib.load("models/serving/p3_kmeans_5_clusters.joblib")
model_p3 = p3_data["model"]
features_p3 = p3_data["features"]


# Datos que recibirá nuestra API
class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    departures_lag_1h: float = Field(ge=0, allow_inf_nan=False)
    departures_lag_24h: float = Field(ge=0, allow_inf_nan=False)
    departures_lag_168h: float = Field(ge=0, allow_inf_nan=False)
    departures_rolling_mean_24h: float = Field(ge=0, allow_inf_nan=False)
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    is_weekend: Literal[0, 1]

class PredictionInputP3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_departures: int = Field(ge=0, allow_inf_nan=False)
    total_arrivals: int = Field(ge=0, allow_inf_nan=False)
    morning_departures: int = Field(ge=0, allow_inf_nan=False)
    weekend_departures: int = Field(ge=0, allow_inf_nan=False)
    dpcapacity: float = Field(ge=0, allow_inf_nan=False)
    avg_daily_departures: float = Field(ge=0, allow_inf_nan=False)
    morning_share: float = Field(ge=0, allow_inf_nan=False)
    weekend_share: float = Field(ge=0, allow_inf_nan=False)
    latitude: float = Field(ge = 0, allow_inf_nan=False)
    longitude: float = Field(le = 0, allow_inf_nan=False)


# Datos que devolverá nuestra API
class PredictionOutput(BaseModel):
    predicted_departures: float

# Datos para P3
class PredictionOutputP3(BaseModel):
    predicted_cluster: int


@app.get("/health")
def health():
    return {"status": "ok", "model": "mibici-p1-rf", "version": "1"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    entrada = pd.DataFrame([data.model_dump()])
    resultado = model.predict(entrada)

    return PredictionOutput(
        predicted_departures=float(resultado[0])
    )

@app.post("/predict/p3", response_model=PredictionOutputP3)
def predict_p3(data: PredictionInputP3):
    input = pd.DataFrame([data.model_dump()])

    input = input[features_p3]

    result = model_p3.predict(input)

    return PredictionOutputP3(
        predicted_cluster=int(result[0])
    )