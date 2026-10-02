"""
scripts/verify_forecaster.py
Verify the trained forecasting model end-to-end.
Run from project root via the runner pattern (copy to TEMP).
"""
import sys
import os

ROOT = os.environ.get("WEATHER_AI_ROOT",
       os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import joblib
from pathlib import Path
from datetime import datetime

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [OK]   {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1

# --- Test 1: file exists ---
print("--- Model Artifact ---")
model_path = Path(ROOT) / "models" / "forecaster.joblib"
check("forecaster.joblib exists", model_path.exists(), str(model_path))

if not model_path.exists():
    print("  Run: python training/train_forecaster.py first")
    sys.exit(1)

# --- Test 2: artifact structure ---
artifact = joblib.load(model_path)
check("artifact is a dict",           isinstance(artifact, dict))
check("artifact has 'model' key",     "model"        in artifact)
check("artifact has 'feature_cols'",  "feature_cols" in artifact)
check("artifact has 'metadata'",      "metadata"     in artifact)

meta = artifact.get("metadata", {})
fc   = artifact.get("feature_cols", [])
check("feature_cols non-empty",       len(fc) > 0, f"got {len(fc)}")
check("metadata has evaluation",      "evaluation" in meta)

print(f"\n  Model type   : {meta.get('model_type', 'unknown')}")
print(f"  Train period : {meta.get('train_start')} -> {meta.get('train_end')}")
print(f"  Test period  : {meta.get('test_start')} -> {meta.get('test_end')}")
ridge_eval = meta.get("evaluation", {}).get("ridge", {})
print(f"  Test RMSE    : {ridge_eval.get('rmse', '?')} C")
print(f"  Features     : {len(fc)}")
print()

# --- Test 3: model predict method ---
print("--- Model Object ---")
model_obj = artifact["model"]
check("model has predict()",  hasattr(model_obj, "predict"))
check("model has weights",    hasattr(model_obj, "weights") and model_obj.weights is not None)
print()

# --- Test 4: inference wrapper ---
print("--- Inference Wrapper ---")
from app.models.forecasting import ForecastingModel, WeatherConditions

fm = ForecastingModel(model_path=model_path)

cond = WeatherConditions(
    temperature=38.5,   humidity=35.0,   wind_speed=8.0,
    precipitation=0.0,  cloud_cover=20.0, pressure=998.0,
    radiation=650.0,    evapotranspiration=0.45,
    timestamp=datetime(2026, 6, 15, 15, 0, 0),
    temp_lag_1h=38.0,  temp_lag_2h=37.2, temp_lag_3h=36.0,
    temp_lag_6h=32.0,  temp_lag_24h=37.8,
    temp_rmean_3h=37.2, temp_rmean_6h=35.5, temp_rmean_12h=32.0,
    temp_rstd_3h=1.0,  temp_trend_3h=2.5, rad_rmean_3h=620.0,
    hum_lag_1h=36.0,   wind_lag_1h=7.5,   rad_lag_1h=630.0,
    pres_lag_1h=998.5,
)

result = fm.predict(cond)
check("predict() returns result",        result is not None)
check("predicted_temperature is float",  isinstance(result.predicted_temperature, float))
check("prediction in range (-10..55 C)",
      -10 <= result.predicted_temperature <= 55,
      f"got {result.predicted_temperature}")
delta = abs(result.predicted_temperature - cond.temperature)
check("delta from input < 5 C",  delta < 5.0, f"delta={delta:.2f}")

print(f"\n  Input temp    : {cond.temperature} C")
print(f"  Predicted temp: {result.predicted_temperature} C (next hour)")
print(f"  Prediction at : {result.prediction_datetime}")
print(f"  Model type    : {result.model_type}")
print()

# --- Test 5: metrics ---
print("--- Metric Check ---")
eval_data  = meta.get("evaluation", {})
ridge_rmse = eval_data.get("ridge", {}).get("rmse")
base_rmse  = eval_data.get("baseline", {}).get("rmse")

check("evaluation metrics present",   ridge_rmse is not None and base_rmse is not None)
if ridge_rmse and base_rmse:
    check("Ridge RMSE < Baseline RMSE", ridge_rmse < base_rmse,
          f"ridge={ridge_rmse}, baseline={base_rmse}")
    impr = (base_rmse - ridge_rmse) / base_rmse * 100
    print(f"\n  Baseline RMSE : {base_rmse:.4f} C")
    print(f"  Ridge RMSE    : {ridge_rmse:.4f} C")
    print(f"  Improvement   : {impr:.1f}%")

# --- Summary ---
print()
print("=" * 55)
total = passed + failed
print(f"Results: {passed}/{total} checks passed")
if failed == 0:
    print("PASS - Milestone 1 forecaster verification complete.")
else:
    print(f"FAIL - {failed} check(s) failed.")
    sys.exit(1)
