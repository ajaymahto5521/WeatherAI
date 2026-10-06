"""Next-hour temperature forecast for New Delhi, served from the joblib artifact
written by training/train_forecaster.py."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WeatherConditions:
    """Weather at hour T plus recent history; the model predicts T+1.

    Units: temperatures in °C, humidity/cloud_cover in %, wind_speed in km/h,
    precipitation in mm (past hour), pressure in hPa, radiation in W/m²,
    evapotranspiration in mm/h. The lag/rolling values must be supplied by the
    caller (hour-ago temperatures, trailing means, etc.).
    """
    temperature: float
    humidity: float
    wind_speed: float
    precipitation: float
    cloud_cover: float
    pressure: float
    radiation: float
    evapotranspiration: float
    timestamp: datetime

    temp_lag_1h: float
    temp_lag_2h: float
    temp_lag_3h: float
    temp_lag_6h: float
    temp_lag_24h: float

    temp_rmean_3h: float
    temp_rmean_6h: float
    temp_rmean_12h: float
    temp_rstd_3h: float
    temp_trend_3h: float
    rad_rmean_3h: float

    hum_lag_1h: float
    wind_lag_1h: float
    rad_lag_1h: float
    pres_lag_1h: float


@dataclass
class ForecastResult:
    predicted_temperature: float  # °C
    prediction_hour: int          # 0-23
    prediction_datetime: str      # ISO 8601
    model_type: str
    location: str
    confidence_note: str


class ForecastingModel:
    """Loads the saved model on first use and turns WeatherConditions into a forecast."""

    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            from app.config import settings
            model_path = settings.forecasting_model_path

        self._path = Path(model_path)
        self._model = None
        self._metadata: dict = {}
        self._feature_cols: list[str] = []

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Forecasting model not found at {self._path}. "
                "Run: python training/train_forecaster.py"
            )

        logger.info("Loading forecasting model from %s", self._path)
        artifact = joblib.load(self._path)
        self._model = artifact["model"]
        self._metadata = artifact.get("metadata", {})
        self._feature_cols = artifact["feature_cols"]

    def _build_feature_vector(self, cond: WeatherConditions) -> np.ndarray:
        """Build a (1, n_features) array in the column order saved at training time.

        Names and formulas here must match engineer_features() in
        training/train_forecaster.py, otherwise predictions are silently wrong.
        """
        ts = cond.timestamp
        doy = ts.timetuple().tm_yday

        feature_map = {
            "temp_lag_1h":   cond.temp_lag_1h,
            "temp_lag_2h":   cond.temp_lag_2h,
            "temp_lag_3h":   cond.temp_lag_3h,
            "temp_lag_6h":   cond.temp_lag_6h,
            "temp_lag_24h":  cond.temp_lag_24h,

            "hum_lag_1h":    cond.hum_lag_1h,
            "wind_lag_1h":   cond.wind_lag_1h,
            "rad_lag_1h":    cond.rad_lag_1h,
            "pres_lag_1h":   cond.pres_lag_1h,

            "temp_rmean_3h":  cond.temp_rmean_3h,
            "temp_rmean_6h":  cond.temp_rmean_6h,
            "temp_rmean_12h": cond.temp_rmean_12h,
            "temp_rstd_3h":   cond.temp_rstd_3h,
            "temp_trend_3h":  cond.temp_trend_3h,
            "rad_rmean_3h":   cond.rad_rmean_3h,

            "hour_sin":  np.sin(2 * np.pi * ts.hour / 24),
            "hour_cos":  np.cos(2 * np.pi * ts.hour / 24),
            "month_sin": np.sin(2 * np.pi * ts.month / 12),
            "month_cos": np.cos(2 * np.pi * ts.month / 12),
            "doy_sin":   np.sin(2 * np.pi * doy / 365),
            "doy_cos":   np.cos(2 * np.pi * doy / 365),

            "humidity":    cond.humidity,
            "wind_speed":  cond.wind_speed,
            "precip":      cond.precipitation,
            "cloud_cover": cond.cloud_cover,
            "pressure":    cond.pressure,
            "radiation":   cond.radiation,
            "evapotrans":  cond.evapotranspiration,

            "is_daytime": float(6 <= ts.hour <= 20),
        }

        try:
            vector = [feature_map[c] for c in self._feature_cols]
        except KeyError as e:
            raise ValueError(
                f"Feature {e} is in the training list but missing here. "
                "Compare train_forecaster.py with forecasting.py."
            ) from e
        return np.array(vector, dtype=np.float64).reshape(1, -1)

    def predict(self, conditions: WeatherConditions) -> ForecastResult:
        if self._model is None:
            self._load()

        X = self._build_feature_vector(conditions)
        # clip to a sane range in case the model extrapolates
        predicted_temp = float(np.clip(self._model.predict(X)[0], -10.0, 55.0))
        next_hour = conditions.timestamp + timedelta(hours=1)

        return ForecastResult(
            predicted_temperature=round(predicted_temp, 1),
            prediction_hour=next_hour.hour,
            prediction_datetime=next_hour.isoformat(),
            model_type=self._metadata.get("model_type", "RidgeRegression"),
            location=self._metadata.get("location", "New Delhi, India"),
            confidence_note=(
                "This model was trained on 2 years of historical data for New Delhi. "
                "Test RMSE is approximately 0.86°C. "
                "Predictions are for educational purposes only."
            ),
        )
