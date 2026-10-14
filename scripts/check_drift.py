"""Report input drift: logged forecast inputs vs. the training feature matrix.

Usage:  python scripts/check_drift.py [--threshold 1.0]
Needs data/processed/features.csv (from training) and logs/forecast_predictions.jsonl.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.drift import drift_report, load_prediction_log  # noqa: E402
from app.monitoring import PREDICTION_LOG  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=1.0, help="mean shift in training stds")
    args = parser.parse_args()

    features = ROOT / "data" / "processed" / "features.csv"
    for path in (features, PREDICTION_LOG):
        if not path.exists():
            print(f"Missing {path}")
            return 2

    report = drift_report(pd.read_csv(features), load_prediction_log(PREDICTION_LOG), args.threshold)
    print(report.round(3).to_string(index=False))
    drifted = report[report["drifted"]]["feature"].tolist()
    print(f"\nDrifted features: {', '.join(drifted) if drifted else 'none'}")
    return 1 if drifted else 0


if __name__ == "__main__":
    sys.exit(main())
