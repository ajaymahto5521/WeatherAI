"""Train the next-hour temperature forecaster (numpy models, no scikit-learn).

Compares a persistence baseline, ridge regression and gradient-boosted stumps
on a chronological 80/20 split, then saves ridge (the best performer) plus the
boosted model to models/forecaster.joblib.

Usage:  python training/train_forecaster.py
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH  = PROJECT_ROOT / "data" / "raw" / "delhi_weather_hourly.csv"
PROC_DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_DIR      = PROJECT_ROOT / "models"
MODEL_PATH     = MODEL_DIR / "forecaster.joblib"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import tracking  # noqa: E402

# Must come from the package so the pickle references app.models.ml_models,
# not __main__.
from app.models.ml_models import (  # noqa: E402
    GradientBoostingRegressor,
    RidgeRegression,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    logger.info("Loading raw data from %s", path)
    if not path.exists():
        logger.error("Data file not found: %s", path)
        logger.error("Run:  python scripts/download_weather_data.py")
        sys.exit(1)

    df = pd.read_csv(path, index_col="time", parse_dates=True)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))
    logger.info("Date range: %s to %s", df.index[0], df.index[-1])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features at hour T with the temperature at T+1 as `target`.

    Column names must stay in sync with ForecastingModel._build_feature_vector.
    """
    logger.info("Engineering features...")
    feat = pd.DataFrame(index=df.index)

    # lags
    feat["temp_lag_1h"]  = df["temperature_2m"].shift(1)
    feat["temp_lag_2h"]  = df["temperature_2m"].shift(2)
    feat["temp_lag_3h"]  = df["temperature_2m"].shift(3)
    feat["temp_lag_6h"]  = df["temperature_2m"].shift(6)
    feat["temp_lag_24h"] = df["temperature_2m"].shift(24)
    feat["hum_lag_1h"]   = df["relative_humidity_2m"].shift(1)
    feat["wind_lag_1h"]  = df["windspeed_10m"].shift(1)
    feat["rad_lag_1h"]   = df["shortwave_radiation"].shift(1)
    feat["pres_lag_1h"]  = df["pressure_msl"].shift(1)

    # rolling stats
    feat["temp_rmean_3h"]  = df["temperature_2m"].rolling(3,  min_periods=1).mean()
    feat["temp_rmean_6h"]  = df["temperature_2m"].rolling(6,  min_periods=1).mean()
    feat["temp_rmean_12h"] = df["temperature_2m"].rolling(12, min_periods=1).mean()
    feat["temp_rstd_3h"]   = df["temperature_2m"].rolling(3,  min_periods=2).std().fillna(0)
    feat["temp_trend_3h"]  = df["temperature_2m"] - df["temperature_2m"].shift(3)
    feat["rad_rmean_3h"]   = df["shortwave_radiation"].rolling(3, min_periods=1).mean()

    # cyclic time encodings (so hour 23 and hour 0 are neighbours)
    feat["hour_sin"]  = np.sin(2 * np.pi * df.index.hour        / 24)
    feat["hour_cos"]  = np.cos(2 * np.pi * df.index.hour        / 24)
    feat["month_sin"] = np.sin(2 * np.pi * df.index.month       / 12)
    feat["month_cos"] = np.cos(2 * np.pi * df.index.month       / 12)
    feat["doy_sin"]   = np.sin(2 * np.pi * df.index.day_of_year / 365)
    feat["doy_cos"]   = np.cos(2 * np.pi * df.index.day_of_year / 365)

    # raw conditions
    feat["humidity"]    = df["relative_humidity_2m"]
    feat["wind_speed"]  = df["windspeed_10m"]
    feat["precip"]      = df["precipitation"]
    feat["cloud_cover"] = df["cloudcover"]
    feat["pressure"]    = df["pressure_msl"]
    feat["radiation"]   = df["shortwave_radiation"]
    feat["evapotrans"]  = df["et0_fao_evapotranspiration"]

    feat["is_daytime"] = ((df.index.hour >= 6) & (df.index.hour <= 20)).astype(float)

    feat["target"] = df["temperature_2m"].shift(-1)

    n_before = len(feat)
    feat = feat.dropna()
    logger.info("Features: %d rows x %d cols  (%d rows dropped for NaN)",
                len(feat), len(feat.columns) - 1, n_before - len(feat))
    return feat


def temporal_split(feat: pd.DataFrame, test_fraction: float = 0.20) -> tuple:
    """Chronological split; the test set is the most recent slice."""
    feature_cols = [c for c in feat.columns if c != "target"]

    split_idx = int(len(feat) * (1 - test_fraction))
    train = feat.iloc[:split_idx]
    test = feat.iloc[split_idx:]

    X_train = train[feature_cols].values.astype(np.float64)
    y_train = train["target"].values.astype(np.float64)
    X_test = test[feature_cols].values.astype(np.float64)
    y_test = test["target"].values.astype(np.float64)

    logger.info("Train: %d rows  (%s to %s)",
                len(train), train.index[0].date(), train.index[-1].date())
    logger.info("Test:  %d rows  (%s to %s)",
                len(test), test.index[0].date(), test.index[-1].date())

    return X_train, y_train, X_test, y_test, feature_cols, train, test


def persistence_baseline(X_test: np.ndarray, feature_cols: list) -> np.ndarray:
    """Predict that the temperature stays put (temp_lag_1h is the temp at T)."""
    return X_test[:, feature_cols.index("temp_lag_1h")]


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def print_results_table(y_test, base_preds, ridge_preds, gb_preds) -> dict:
    """Print the comparison table and return the metrics dict."""
    results = {
        "baseline": {"rmse": rmse(y_test, base_preds), "mae": mae(y_test, base_preds)},
        "ridge": {"rmse": rmse(y_test, ridge_preds), "mae": mae(y_test, ridge_preds),
                  "r2": r2(y_test, ridge_preds)},
        "gradient_boosting": {"rmse": rmse(y_test, gb_preds), "mae": mae(y_test, gb_preds),
                              "r2": r2(y_test, gb_preds)},
    }
    results = {k: {m: round(v, 4) for m, v in d.items()} for k, d in results.items()}
    base_rmse = results["baseline"]["rmse"]

    rows = [
        ("Persistence Baseline", "baseline"),
        ("Ridge Regression (deployed)", "ridge"),
        ("Gradient Boosting", "gradient_boosting"),
    ]
    print(f"\n  {'Model':<30} {'RMSE (°C)':>10} {'MAE (°C)':>10} {'R²':>8} {'vs baseline':>12}")
    for name, key in rows:
        m = results[key]
        r2_text = f"{m['r2']:.4f}" if "r2" in m else "—"
        gain = (base_rmse - m["rmse"]) / base_rmse * 100
        print(f"  {name:<30} {m['rmse']:>10.4f} {m['mae']:>10.4f} {r2_text:>8} {gain:>+11.1f}%")

    verdict = "beats" if results["ridge"]["rmse"] < base_rmse else "does NOT beat"
    print(f"\n  Ridge {verdict} the persistence baseline.")
    return results


def main() -> None:
    with tracking.start_run("forecaster", "ridge-vs-gb"):
        _train()


def _train() -> None:
    df = load_data(RAW_DATA_PATH)

    feat = engineer_features(df)
    PROC_DATA_PATH.mkdir(parents=True, exist_ok=True)
    feat.to_csv(PROC_DATA_PATH / "features.csv")
    logger.info("Feature matrix saved to data/processed/features.csv")

    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = \
        temporal_split(feat, test_fraction=0.20)

    base_preds = persistence_baseline(X_test, feature_cols)

    logger.info("Training Ridge Regression (alpha=1.0)...")
    ridge = RidgeRegression(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    logger.info("Ridge done. RMSE=%.4f", rmse(y_test, ridge_preds))

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

    results = print_results_table(y_test, base_preds, ridge_preds, gb_preds)

    tracking.log_params({
        "ridge_alpha": ridge.alpha, "gb_n_estimators": 200, "gb_learning_rate": 0.1,
        "gb_subsample": 0.8, "test_fraction": 0.20, "n_train": len(X_train),
        "n_test": len(X_test), "n_features": len(feature_cols),
    })
    tracking.log_metrics({f"{model}_{metric}": value
                          for model, d in results.items() for metric, value in d.items()})

    # Ridge is what gets served; the boosted model is kept for comparison.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model":          ridge,
        "model_type":     "RidgeRegression",
        "gb_model":       gb,
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
    tracking.log_artifact(MODEL_PATH)
    tracking.register_model("weatherai-forecaster", MODEL_PATH)
    logger.info("Model saved to %s  (%.0f KB)", MODEL_PATH, size_kb)

    print(f"\n  Done. Deployed Ridge (RMSE={results['ridge']['rmse']:.4f}°C): {MODEL_PATH}\n")


if __name__ == "__main__":
    main()
