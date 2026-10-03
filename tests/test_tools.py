import pytest
import requests

from app.agents import tools
from tests.conftest import ROOT


def test_build_conditions_matches_training_features(hourly):
    c = tools.build_conditions(hourly, 72)
    assert c.temperature == pytest.approx(32.2)
    assert c.temp_lag_1h == pytest.approx(32.1)
    assert c.temp_lag_24h == pytest.approx(29.8)
    assert c.temp_trend_3h == pytest.approx(0.3)
    assert c.temp_rmean_3h == pytest.approx(32.1)
    assert c.temp_rstd_3h == pytest.approx(0.1)


def test_build_conditions_needs_history(hourly):
    with pytest.raises(ValueError):
        tools.build_conditions(hourly, 10)


def test_build_conditions_rejects_missing_values(hourly):
    hourly["temperature_2m"][70] = None
    with pytest.raises(ValueError):
        tools.build_conditions(hourly, 72)


def test_live_weather_needs_key(monkeypatch):
    monkeypatch.setattr(tools.settings, "openweather_api_key", "")
    assert "not configured" in tools.get_live_weather("Pune").text


def test_live_weather_parses_and_builds_severity_text(stub_network):
    r = tools.get_live_weather("Pune")
    assert "Pune" in r.text
    assert "Winds 7 km/h" in r.data["severity_text"]  # 2 m/s -> 7 km/h


def test_live_weather_unknown_city(monkeypatch, stub_network):
    resp = requests.Response()
    resp.status_code = 404

    def raise404(url, params):
        raise requests.HTTPError(response=resp)

    monkeypatch.setattr(tools, "_http_get_json", raise404)
    assert "not found" in tools.get_live_weather("Nowhere").text


def test_live_weather_network_error(monkeypatch, stub_network):
    def boom(url, params):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(tools, "_http_get_json", boom)
    assert "failed" in tools.get_live_weather("Pune").text


def test_forecast_degrades_when_current_hour_missing(stub_network, hourly):
    hourly["time"] = ["2001" + t[4:] for t in hourly["time"]]
    assert "unavailable" in tools.forecast_temperature().text


def test_forecast_degrades_on_network_error(monkeypatch):
    def boom(url, params):
        raise requests.Timeout("slow")

    monkeypatch.setattr(tools, "_http_get_json", boom)
    assert "unavailable" in tools.forecast_temperature().text


@pytest.mark.slow
def test_forecast_with_real_model(stub_network, now):
    if not (ROOT / "models" / "forecaster.joblib").exists():
        pytest.skip("forecaster not trained")
    r = tools.forecast_temperature(now=now)
    assert -10 <= r.data["predicted_temperature"] <= 55


def test_rag_tool_returns_sources(retriever):
    r = tools.make_rag_tool(retriever)("tornado warning")
    assert r.data["sources"] and "[1]" in r.text
