"""Compare logged forecast inputs against the training feature distribution."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Cyclic time encodings and the daytime flag are fixed by the calendar, not weather.
SKIP = {"hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos", "is_daytime"}


def load_prediction_log(path: Path) -> pd.DataFrame:
    rows = [json.loads(line)["features"] for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)


def drift_report(train: pd.DataFrame, live: pd.DataFrame, threshold: float = 1.0,
                 min_rows: int = 30) -> pd.DataFrame:
    """Per-feature mean shift in training-std units; `drifted` when |shift| > threshold.

    Live data covers a short window, so seasonal features legitimately shift a little;
    compare against the same season's training rows before acting on small values.
    """
    if len(live) < min_rows:
        raise ValueError(f"Need at least {min_rows} logged predictions, have {len(live)}.")
    rows = []
    for col in live.columns:
        if col in SKIP or col not in train.columns:
            continue
        std = train[col].std()
        shift = (live[col].mean() - train[col].mean()) / std if std > 0 else 0.0
        rows.append({"feature": col, "train_mean": train[col].mean(), "live_mean": live[col].mean(),
                     "shift_std": float(shift), "drifted": bool(abs(shift) > threshold)})
    return pd.DataFrame(rows).sort_values("shift_std", key=np.abs, ascending=False, ignore_index=True)
