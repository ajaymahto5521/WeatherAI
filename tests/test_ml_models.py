"""Tests for the trained models. Skipped when artifacts haven't been trained locally."""
import pytest

from tests.conftest import ROOT

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def clf():
    from app.models.severity import SeverityClassifier
    return SeverityClassifier(ROOT / "models" / "severity_classifier")


@pytest.mark.skipif(not (ROOT / "models" / "severity_classifier" / "training_meta.json").exists(),
                    reason="severity classifier not trained")
class TestSeverity:
    def test_calm_is_low(self, clf):
        assert clf.classify("Clear skies and a gentle breeze. No precautions needed.").label == "LOW"

    def test_catastrophe_is_critical(self, clf):
        text = "Evacuate immediately: catastrophic cyclone landfall with winds 180 km/h."
        assert clf.classify(text).label == "CRITICAL"

    def test_probabilities_sum_to_one(self, clf):
        r = clf.classify("Light drizzle in the evening.")
        assert sum(r.probabilities.values()) == pytest.approx(1, abs=0.01)
        assert r.confidence == max(r.probabilities.values())


def test_missing_severity_model_raises(tmp_path):
    from app.models.severity import SeverityClassifier
    with pytest.raises(FileNotFoundError):
        SeverityClassifier(tmp_path).classify("x")


def test_missing_forecaster_raises(tmp_path):
    from app.models.forecasting import ForecastingModel
    with pytest.raises(FileNotFoundError):
        ForecastingModel(tmp_path / "none.joblib")._load()
