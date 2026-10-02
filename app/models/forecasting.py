"""
app/models/forecasting.py
==========================
Inference wrapper for the trained temperature forecasting model.

This module is the ONLY place the rest of the application touches the
forecasting model. It provides a clean, validated interface that hides
all the internal details (feature names, normalisation, model type).

TRAINING vs INFERENCE separation:
  Training:  training/train_forecaster.py  → runs once, saves forecaster.joblib
  Inference: this file                     → runs per request, loads from .joblib

The model predicts: next-hour temperature in °C for New Delhi.

Usage:
    from app.models.forecasting import ForecastingModel

    model = ForecastingModel()                    # loads model from disk
    result = model.predict(current_conditions)    # returns a dict
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import — joblib is only needed when the model is loaded
# This keeps import time fast if the model isn't used
try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    logger.warning("joblib not installed — ForecastingModel will not work")


# =============================================================================
# Input schema (dataclass — no Pydantic dependency in the models layer)
# =============================================================================

@dataclass
class WeatherConditions:
    """
    Current weather conditions required to make a 1-hour-ahead forecast.

    All values are for the CURRENT hour (time T).
    The model predicts temperature at T+1.

    Units:
        temperatures     — degrees Celsius
        humidity         — percentage (0–100)
        wind_speed       — km/h
        precipitation    — mm in the past hour
        cloud_cover      — percentage (0–100)
        pressure         — hPa (hectopascals)
        radiation        — W/m²
        evapotranspiration — mm/hour
        timestamp        — current datetime (used for hour/month cyclic features)

    Lag values (temp_lag_*):
        These are PAST temperatures that you must supply.
        temp_lag_1h  = temperature 1 hour ago
        temp_lag_2h  = temperature 2 hours ago
        temp_lag_3h  = temperature 3 hours ago
        temp_lag_6h  = temperature 6 hours ago
        temp_lag_24h = temperature 24 hours ago (same hour yesterday)

    In a real deployment these would come from a short historical buffer
    maintained in memory or a small database. For the demo they can be
    approximated as current_temp - small_delta.
    """
    # Current conditions
    temperature: float
    humidity: float
    wind_speed: float
    precipitation: float
    cloud_cover: float
    pressure: float
    radiation: float
    evapotranspiration: float
    timestamp: datetime

    # Lag temperatures (past hours)
    temp_lag_1h: float
    temp_lag_2h: float
    temp_lag_3h: float
    temp_lag_6h: float
    temp_lag_24h: float

    # Rolling statistics (pre-computed from recent history)
    temp_rmean_3h: float
    temp_rmean_6h: float
    temp_rmean_12h: float
    temp_rstd_3h: float
    temp_trend_3h: float
    rad_rmean_3h: float

    # Lag of other variables
    hum_lag_1h: float
    wind_lag_1h: float
    rad_lag_1h: float
    pres_lag_1h: float


@dataclass
class ForecastResult:
    """Output of a single forecast call."""
    predicted_temperature: float     # °C, next hour
    prediction_hour: int             # hour of day for the prediction (0–23)
    prediction_datetime: str         # ISO 8601 string
    model_type: str                  # which model produced this
    location: str                    # location this model was trained for
    confidence_note: str             # honest note about model limitations


# =============================================================================
# Model class
# =============================================================================

class ForecastingModel:
    """
    Thin wrapper around the trained forecasting model artifact.

    Responsibilities:
    - Load the model artifact from disk (once, lazily on first predict)
    - Build the feature vector from WeatherConditions in the exact same
      order and format as the training script
    - Call model.predict() and wrap the result in ForecastResult
    - Log prediction inputs and outputs for observability

    Thread safety:
    - The model is loaded once during initialisation.
    - predict() is stateless — safe to call from multiple threads.
    """

    def __init__(self, model_path: str | Path | None = None):
        """
        Parameters
        ----------
        model_path : path to the .joblib artifact.
                     If None, reads from app.config.settings.forecasting_model_path
        """
        if model_path is None:
            from app.config import settings
            model_path = settings.forecasting_model_path

        self._path     = Path(model_path)
        self._artifact = None     # loaded lazily on first call
        self._model    = None
        self._metadata = None
        self._feature_cols: list[str] = []

    def _load(self) -> None:
        """Load the joblib artifact from disk. Called once on first predict."""
        if not _JOBLIB_AVAILABLE:
            raise RuntimeError("joblib is not installed. Run: pip install joblib")

        if not self._path.exists():
            raise FileNotFoundError(
                f"Forecasting model not found at {self._path}. "
                "Run: python training/train_forecaster.py"
            )

        logger.info("Loading forecasting model from %s", self._path)
        self._artifact     = joblib.load(self._path)
        self._model        = self._artifact["model"]
        self._metadata     = self._artifact.get("metadata", {})
        self._feature_cols = self._artifact.get("feature_cols", [])

        logger.info(
            "Forecasting model loaded: %s | trained %s → %s | test RMSE: %.4f°C",
            self._metadata.get("model_type", "unknown"),
            self._metadata.get("train_start", "?"),
            self._metadata.get("train_end", "?"),
            self._metadata.get("evaluation", {}).get("ridge", {}).get("rmse", 0),
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _build_feature_vector(self, cond: WeatherConditions) -> np.ndarray:
        """
        Convert WeatherConditions into a numpy array matching the training
        feature order exactly.

        This is the most critical method — the feature ORDER and VALUES must
        exactly match what train_forecaster.py produced. If there's any mismatch,
        predictions will be silently wrong.

        The feature order is stored in self._feature_cols (from the artifact).
        We build a dict and then extract values in that order.
        """
        ts = cond.timestamp

        # Build feature dict matching engineering in train_forecaster.py
        feature_map = {
            # Lag temperatures
            "temp_lag_1h":   cond.temp_lag_1h,
            "temp_lag_2h":   cond.temp_lag_2h,
            "temp_lag_3h":   cond.temp_lag_3h,
            "temp_lag_6h":   cond.temp_lag_6h,
            "temp_lag_24h":  cond.temp_lag_24h,

            # Lag other variables
            "hum_lag_1h":    cond.hum_lag_1h,
            "wind_lag_1h":   cond.wind_lag_1h,
            "rad_lag_1h":    cond.rad_lag_1h,
            "pres_lag_1h":   cond.pres_lag_1h,

            # Rolling statistics
            "temp_rmean_3h":  cond.temp_rmean_3h,
            "temp_rmean_6h":  cond.temp_rmean_6h,
            "temp_rmean_12h": cond.temp_rmean_12h,
            "temp_rstd_3h":   cond.temp_rstd_3h,
            "temp_trend_3h":  cond.temp_trend_3h,
            "rad_rmean_3h":   cond.rad_rmean_3h,

            # Cyclic time encodings
            "hour_sin":  np.sin(2 * np.pi * ts.hour  / 24),
            "hour_cos":  np.cos(2 * np.pi * ts.hour  / 24),
            "month_sin": np.sin(2 * np.pi * ts.month / 12),
            "month_cos": np.cos(2 * np.pi * ts.month / 12),
            "doy_sin":   np.sin(2 * np.pi * ts.timetuple().tm_yday / 365),
            "doy_cos":   np.cos(2 * np.pi * ts.timetuple().tm_yday / 365),

            # Raw weather features
            "humidity":   cond.humidity,
            "wind_speed": cond.wind_speed,
            "precip":     cond.precipitation,
            "cloud_cover":cond.cloud_cover,
            "pressure":   cond.pressure,
            "radiation":  cond.radiation,
            "evapotrans": cond.evapotranspiration,

            # Calendar flag
            "is_daytime": float(6 <= ts.hour <= 20),
        }

        # Extract values in the exact column order used during training
        if self._feature_cols:
            try:
                vector = np.array([feature_map[c] for c in self._feature_cols],
                                   dtype=np.float64)
            except KeyError as e:
                missing = str(e)
                raise ValueError(
                    f"Feature {missing} is in the training feature list but "
                    "not in the inference feature map. Check for mismatch between "
                    "train_forecaster.py and forecasting.py."
                ) from e
        else:
            # Fallback: use dict insertion order (Python 3.7+)
            vector = np.array(list(feature_map.values()), dtype=np.float64)

        return vector.reshape(1, -1)   # shape: (1, n_features)

    def predict(self, conditions: WeatherConditions) -> ForecastResult:
        """
        Predict next-hour temperature given current weather conditions.

        Parameters
        ----------
        conditions : WeatherConditions
            Current and recent weather values.

        Returns
        -------
        ForecastResult
            Prediction with metadata.
        """
        # Lazy load on first call
        if not self.is_loaded:
            self._load()

        X = self._build_feature_vector(conditions)

        # Call the model — this is the same .predict() used by any numpy model
        raw_prediction = self._model.predict(X)
        predicted_temp = float(raw_prediction[0])

        # Clip to a physically plausible range for New Delhi
        # (protects against extrapolation errors)
        predicted_temp = float(np.clip(predicted_temp, -10.0, 55.0))

        from datetime import timedelta
        next_hour = conditions.timestamp + timedelta(hours=1)

        result = ForecastResult(
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

        logger.debug(
            "Forecast: %.1f°C at %s (input temp: %.1f°C)",
            result.predicted_temperature,
            result.prediction_datetime,
            conditions.temperature,
        )

        return result

    def get_metadata(self) -> dict:
        """Return training metadata (useful for the /ask response)."""
        if not self.is_loaded:
            self._load()
        return dict(self._metadata)


# =============================================================================
# Module-level singleton — loaded once per process
# =============================================================================

# We don't instantiate at import time because:
# 1. The model file may not exist yet (before training)
# 2. We want to control WHEN the disk read happens (at startup, not import)
# The singleton is created in app/main.py lifespan and stored on app.state

def create_forecasting_model(model_path: str | Path | None = None) -> ForecastingModel:
    """
    Factory function. Call this in the FastAPI lifespan to pre-load the model.

    Example (in main.py lifespan):
        app.state.forecasting_model = create_forecasting_model()
        # This triggers the disk load immediately at startup.
        # Subsequent requests use the already-loaded model.
    """
    model = ForecastingModel(model_path=model_path)
    model._load()   # eager load — fail fast if model file is missing
    return model
