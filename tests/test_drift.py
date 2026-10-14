import numpy as np
import pandas as pd
import pytest

from app.drift import drift_report


def _frames(live_shift):
    rng = np.random.RandomState(0)
    train = pd.DataFrame({"humidity": rng.normal(50, 10, 1000), "hour_sin": rng.uniform(-1, 1, 1000)})
    live = pd.DataFrame({"humidity": rng.normal(50 + live_shift, 10, 50), "hour_sin": np.full(50, 0.9)})
    return train, live


def test_no_drift_when_distribution_matches():
    report = drift_report(*_frames(0))
    assert not report["drifted"].any()
    assert "hour_sin" not in report["feature"].values  # calendar features skipped


def test_flags_shifted_feature():
    report = drift_report(*_frames(25))
    assert report.loc[0, "feature"] == "humidity" and report.loc[0, "drifted"]


def test_too_few_rows_raises():
    train, live = _frames(0)
    with pytest.raises(ValueError):
        drift_report(train, live.head(5))
