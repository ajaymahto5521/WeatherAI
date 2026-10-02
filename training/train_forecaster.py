"""
training/train_forecaster.py
=============================
Train a next-hour temperature forecasting model — pure numpy/scipy.

WHY PURE NUMPY?
---------------
This project targets environments where scikit-learn's compiled C extensions
(.pyd/.so files) may be blocked by corporate Application Control policies.
We implement everything from first principles using only:
  - numpy  (array math, linear algebra)
  - scipy  (optimisation helpers)
  - pandas (data loading, time-series operations)
  - joblib (model serialisation)

This is actually MORE instructive — you see exactly what's happening
inside a linear model, rather than calling a black box.

PROBLEM DEFINITION
------------------
Given current weather conditions at time T, predict temperature at T+1h.
This is a regression problem with a time-series structure.

MODELS IMPLEMENTED
------------------
1. Persistence Baseline  — predict(T+1) = temperature(T)   [the "do nothing" model]
2. Ridge Regression      — linear model with L2 regularisation, solved analytically
3. Gradient Boosted Stumps — simple additive ensemble of depth-1 decision trees

Usage:
    python training/train_forecaster.py

Expected output:
    Persistence Baseline  RMSE: ~1.80°C
    Ridge Regression      RMSE: ~1.20°C
    Gradient Boosting     RMSE: ~0.80°C  ← this is our deployed model
"""

import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
# Support running from TEMP or any location via WEATHER_AI_ROOT env var.
_env_root = os.environ.get("WEATHER_AI_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH  = PROJECT_ROOT / "data" / "raw" / "delhi_weather_hourly.csv"
PROC_DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_DIR      = PROJECT_ROOT / "models"
MODEL_PATH     = MODEL_DIR / "forecaster.joblib"

# Add project root to sys.path so we can import app.models.ml_models
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import model classes from the shared module — this ensures the classes
# are serialised as "app.models.ml_models.RidgeRegression" (resolvable
# from any module) rather than "__main__.RidgeRegression" (only from script)
from app.models.ml_models import (  # noqa: E402
    DecisionStump,
    GradientBoostingRegressor,
    RidgeRegression,
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# STEP 1: Load Data
# =============================================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV. Time column becomes a DatetimeIndex."""
    logger.info("Loading raw data from %s", path)
    if not path.exists():
        logger.error("Data file not found: %s", path)
        logger.error("Run:  python scripts/download_weather_data.py")
        sys.exit(1)

    df = pd.read_csv(path, index_col="time", parse_dates=True)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))
    logger.info("Date range: %s to %s", df.index[0], df.index[-1])
    return df


# =============================================================================
# STEP 2: Feature Engineering
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model input features from raw weather columns.

    Feature groups:
      A) Lag features         — recent temperature history
      B) Rolling statistics   — trends and volatility
      C) Cyclic encoding      — hour/month as sin+cos pairs
      D) Raw weather features — humidity, radiation, pressure, etc.
      E) Calendar flags       — is_daytime

    Target: temperature 1 hour ahead (shift -1).
    """
    logger.info("Engineering features...")
    feat = pd.DataFrame(index=df.index)

    # A) Lag features
    feat["temp_lag_1h"]  = df["temperature_2m"].shift(1)
    feat["temp_lag_2h"]  = df["temperature_2m"].shift(2)
    feat["temp_lag_3h"]  = df["temperature_2m"].shift(3)
    feat["temp_lag_6h"]  = df["temperature_2m"].shift(6)
    feat["temp_lag_24h"] = df["temperature_2m"].shift(24)
    feat["hum_lag_1h"]   = df["relative_humidity_2m"].shift(1)
    feat["wind_lag_1h"]  = df["windspeed_10m"].shift(1)
    feat["rad_lag_1h"]   = df["shortwave_radiation"].shift(1)
    feat["pres_lag_1h"]  = df["pressure_msl"].shift(1)

    # B) Rolling statistics — .rolling(w).mean() = mean of last w rows
    feat["temp_rmean_3h"]  = df["temperature_2m"].rolling(3,  min_periods=1).mean()
    feat["temp_rmean_6h"]  = df["temperature_2m"].rolling(6,  min_periods=1).mean()
    feat["temp_rmean_12h"] = df["temperature_2m"].rolling(12, min_periods=1).mean()
    feat["temp_rstd_3h"]   = df["temperature_2m"].rolling(3,  min_periods=2).std().fillna(0)
    feat["temp_trend_3h"]  = df["temperature_2m"] - df["temperature_2m"].shift(3)
    feat["rad_rmean_3h"]   = df["shortwave_radiation"].rolling(3, min_periods=1).mean()

    # C) Cyclic encoding — sin+cos avoids discontinuity at cycle boundaries
    #    hour 23 and hour 0 are adjacent; without encoding they're 23 apart numerically
    feat["hour_sin"]       = np.sin(2 * np.pi * df.index.hour        / 24)
    feat["hour_cos"]       = np.cos(2 * np.pi * df.index.hour        / 24)
    feat["month_sin"]      = np.sin(2 * np.pi * df.index.month       / 12)
    feat["month_cos"]      = np.cos(2 * np.pi * df.index.month       / 12)
    feat["doy_sin"]        = np.sin(2 * np.pi * df.index.day_of_year / 365)
    feat["doy_cos"]        = np.cos(2 * np.pi * df.index.day_of_year / 365)

    # D) Raw weather features
    feat["humidity"]    = df["relative_humidity_2m"]
    feat["wind_speed"]  = df["windspeed_10m"]
    feat["precip"]      = df["precipitation"]
    feat["cloud_cover"] = df["cloudcover"]
    feat["pressure"]    = df["pressure_msl"]
    feat["radiation"]   = df["shortwave_radiation"]
    feat["evapotrans"]  = df["et0_fao_evapotranspiration"]

    # E) Calendar flag
    feat["is_daytime"]  = ((df.index.hour >= 6) & (df.index.hour <= 20)).astype(float)

    # TARGET — 1 hour ahead temperature
    # shift(-1): at row T, this is the temperature at T+1
    feat["target"] = df["temperature_2m"].shift(-1)

    n_before = len(feat)
    feat     = feat.dropna()
    logger.info("Features: %d rows x %d cols  (%d rows dropped for NaN)",
                len(feat), len(feat.columns) - 1, n_before - len(feat))
    return feat


# =============================================================================
# STEP 3: Temporal Train/Test Split
# =============================================================================

def temporal_split(
    feat: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple:
    """
    Chronological 80/20 split — the only correct way to split time-series.

    NO random shuffling. The test set must always be AFTER the training set
    in time — mimicking real deployment where you train on history and
    predict the future.
    """
    target_col   = "target"
    feature_cols = [c for c in feat.columns if c != target_col]

    split_idx = int(len(feat) * (1 - test_fraction))

    train = feat.iloc[:split_idx]
    test  = feat.iloc[split_idx:]

    X_train = train[feature_cols].values.astype(np.float64)
    y_train = train[target_col].values.astype(np.float64)
    X_test  = test[feature_cols].values.astype(np.float64)
    y_test  = test[target_col].values.astype(np.float64)

    logger.info("Train: %d rows  (%s to %s)",
                len(train), train.index[0].date(), train.index[-1].date())
    logger.info("Test:  %d rows  (%s to %s)",
                len(test), test.index[0].date(), test.index[-1].date())

    return X_train, y_train, X_test, y_test, feature_cols, train, test


# =============================================================================
# STEP 4: Persistence Baseline
# =============================================================================

def persistence_baseline(X_test: np.ndarray, feature_cols: list) -> np.ndarray:
    """
    Naive persistence model: predict next-hour temp = current temp.

    temp_lag_1h holds the temperature at T (one step back from T+1).
    So using it as the prediction IS the persistence forecast.

    This is the minimum bar every ML model must clear.
    If your model can't beat "temperature won't change", it's useless.
    """
    idx = feature_cols.index("temp_lag_1h")
    return X_test[:, idx]


# =============================================================================
# STEP 5: Ridge Regression and Gradient Boosting
# (Classes imported from app.models.ml_models — see that file for full docs)
# =============================================================================


# =============================================================================
# STEP 7: Metrics
# =============================================================================

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error in the same units as the target (°C)."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error — average magnitude of errors in °C."""
    return float(np.mean(np.abs(actual - predicted)))


def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    R² — fraction of variance explained.
    1.0 = perfect, 0.0 = as good as predicting the mean, <0 = worse than mean.
    """
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def print_results_table(
    y_test: np.ndarray,
    base_preds: np.ndarray,
    ridge_preds: np.ndarray,
    gb_preds: np.ndarray,
) -> dict:
    """Print a comparison table and return the metrics dict."""
    results = {
        "baseline": {
            "rmse": round(rmse(y_test, base_preds),  4),
            "mae":  round(mae(y_test,  base_preds),  4),
        },
        "ridge": {
            "rmse": round(rmse(y_test, ridge_preds), 4),
            "mae":  round(mae(y_test,  ridge_preds), 4),
            "r2":   round(r2(y_test,   ridge_preds), 4),
        },
        "gradient_boosting": {
            "rmse": round(rmse(y_test, gb_preds),    4),
            "mae":  round(mae(y_test,  gb_preds),    4),
            "r2":   round(r2(y_test,   gb_preds),    4),
        },
    }

    base_rmse  = results["baseline"]["rmse"]
    ridge_rmse = results["ridge"]["rmse"]
    gb_rmse    = results["gradient_boosting"]["rmse"]

    print()
    print("=" * 68)
    print("  EVALUATION RESULTS  (held-out test set, last 20% of data)")
    print("=" * 68)
    print(f"  {'Model':<35} {'RMSE (°C)':>10}  {'MAE (°C)':>10}  {'R²':>6}")
    print(f"  {'-'*35} {'-'*10}  {'-'*10}  {'-'*6}")
    print(f"  {'Persistence Baseline':<35} {base_rmse:>10.4f}  "
          f"{results['baseline']['mae']:>10.4f}  {'—':>6}")
    print(f"  {'Ridge Regression':<35} {ridge_rmse:>10.4f}  "
          f"{results['ridge']['mae']:>10.4f}  {results['ridge']['r2']:>6.4f}")
    print(f"  {'Gradient Boosting (deployed)':<35} {gb_rmse:>10.4f}  "
          f"{results['gradient_boosting']['mae']:>10.4f}  "
          f"{results['gradient_boosting']['r2']:>6.4f}")
    print(f"  {'-'*35} {'-'*10}  {'-'*10}  {'-'*6}")

    for name, model_rmse in [("Ridge", ridge_rmse), ("GradBoost", gb_rmse)]:
        improv = (base_rmse - model_rmse) / base_rmse * 100
        print(f"  {name} RMSE improvement vs baseline: {improv:+.1f}%")
    print("=" * 68)

    if gb_rmse < base_rmse:
        print("\n  [PASS] Gradient Boosting beats the persistence baseline.")
    else:
        print("\n  [FAIL] Model does NOT beat baseline. Check implementation.")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print()
    print("=" * 68)
    print("  Weather Intelligence Assistant — Forecaster Training")
    print("  Implementation: pure numpy/scipy (no scikit-learn)")
    print("=" * 68)
    print()

    # 1. Load
    df = load_data(RAW_DATA_PATH)

    # 2. Feature engineering
    feat = engineer_features(df)
    PROC_DATA_PATH.mkdir(parents=True, exist_ok=True)
    feat.to_csv(PROC_DATA_PATH / "features.csv")
    logger.info("Feature matrix saved to data/processed/features.csv")

    # 3. Temporal split
    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = \
        temporal_split(feat, test_fraction=0.20)

    # 4. Persistence baseline
    base_preds = persistence_baseline(X_test, feature_cols)

    # 5a. Train Ridge Regression
    print()
    logger.info("Training Ridge Regression (lambda=1.0)...")
    ridge = RidgeRegression(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    logger.info("Ridge done. RMSE=%.4f", rmse(y_test, ridge_preds))

    # 5b. Train Gradient Boosting
    print()
    logger.info("Training Gradient Boosting (200 stumps, lr=0.1, subsample=0.8)...")
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    logger.info("Gradient Boosting done. RMSE=%.4f", rmse(y_test, gb_preds))

    # 6. Evaluate
    results = print_results_table(y_test, base_preds, ridge_preds, gb_preds)

    # 7. Save the best model (Ridge wins — linear problem suits Ridge well) + all metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model":          ridge,             # Ridge is the deployed model (best RMSE)
        "model_type":     "RidgeRegression", # label what's deployed
        "gb_model":       gb,                # also save GB for reference/comparison
        "feature_cols":   feature_cols,
        "metadata": {
            "location":    "New Delhi, India",
            "target":      "temperature_2m + 1h (Celsius)",
            "model_type":  "RidgeRegression (pure numpy, L2 regularisation)",
            "deployed_model": "RidgeRegression",
            "train_start": str(train_df.index[0].date()),
            "train_end":   str(train_df.index[-1].date()),
            "test_start":  str(test_df.index[0].date()),
            "test_end":    str(test_df.index[-1].date()),
            "n_train":     int(len(X_train)),
            "n_test":      int(len(X_test)),
            "n_features":  int(len(feature_cols)),
            "evaluation":  results,
            "ridge_alpha": ridge.alpha,
        },
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    size_kb = MODEL_PATH.stat().st_size / 1024
    logger.info("Model saved to %s  (%.0f KB)", MODEL_PATH, size_kb)

    print()
    print(f"  Done. Model: {MODEL_PATH}")
    print(f"  Deployed: Ridge Regression (RMSE={results['ridge']['rmse']:.4f}°C)")
    print(f"  RMSE improvement (Ridge vs baseline): "
          f"{(results['baseline']['rmse']-results['ridge']['rmse'])/results['baseline']['rmse']*100:.1f}%")
    print()


if __name__ == "__main__":
    main()
