"""Verify agent + tools with a scripted LLM and stubbed network. Run from project root."""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from app.agents import tools
from app.agents.agent import WeatherAgent
from app.rag.ingestion import ingest
from app.rag.retrieval import Retriever

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  [OK]   {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1


def fake_hourly(now):
    times = [(now - timedelta(hours=71 - k)).isoformat(timespec="minutes") for k in range(72 + 1)]
    n = len(times)
    return {
        "time": times,
        "temperature_2m": [25 + 0.1 * k for k in range(n)],
        "relative_humidity_2m": [50.0] * n, "windspeed_10m": [8.0] * n,
        "precipitation": [0.0] * n, "cloudcover": [20.0] * n,
        "pressure_msl": [1008.0] * n, "shortwave_radiation": [300.0] * n,
        "et0_fao_evapotranspiration": [0.2] * n,
    }


now = datetime(2026, 7, 1, 14, 0)
hourly = fake_hourly(now)

print("--- build_conditions ---")
c = tools.build_conditions(hourly, 71)
check("lags/rolling match hand calc",
      abs(c.temperature - 32.1) < 1e-9 and abs(c.temp_lag_24h - 29.7) < 1e-9
      and abs(c.temp_trend_3h - 0.3) < 1e-9 and abs(c.temp_rmean_3h - 32.0) < 1e-9, str(c))
try:
    tools.build_conditions(hourly, 10)
    check("short history rejected", False)
except ValueError:
    check("short history rejected", True)

print("--- tools (network stubbed) ---")
tools._http_get_json = lambda url, params: (
    {"hourly": hourly} if "open-meteo" in url else
    {"name": "Mumbai", "main": {"temp": 31, "feels_like": 36, "humidity": 70, "pressure": 1006},
     "weather": [{"description": "haze"}], "wind": {"speed": 3}}
)
r = tools.forecast_temperature(now=now)
if Path("models/forecaster.joblib").exists():
    check("forecast returns prediction", "predicted_temperature" in r.data, r.text)
else:
    check("forecast degrades without model file", "unavailable" in r.text, r.text)
tools.settings.openweather_api_key = ""
check("live weather without key degrades", "not configured" in tools.get_live_weather("Mumbai").text)
tools.settings.openweather_api_key = "k"
check("live weather parses", "Mumbai" in tools.get_live_weather("Mumbai").text)

print("--- agent loop ---")
with tempfile.TemporaryDirectory() as tmp:
    ingest(index_dir=Path(tmp), backend="local", embedding_backend="tfidf")
    registry = tools.build_registry(Retriever(index_dir=Path(tmp), backend="local"))

    script = iter([
        {"content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {
                "name": "search_weather_knowledge", "arguments": json.dumps({"query": "tornado warning"})}},
            {"id": "2", "type": "function", "function": {
                "name": "get_live_weather", "arguments": json.dumps({"city": "Mumbai"})}},
            {"id": "3", "type": "function", "function": {"name": "nope", "arguments": "{}"}},
        ]},
        {"content": "Final answer [1].", "tool_calls": None},
    ])
    seen = []

    def chat(messages, schemas):
        seen.append(messages[-1])
        return next(script)

    resp = WeatherAgent(chat, registry).answer("tornado warning and Mumbai weather?")
    check("final answer", resp.answer == "Final answer [1].")
    check("tools recorded", resp.tools_used == ["search_weather_knowledge", "get_live_weather", "nope"],
          str(resp.tools_used))
    check("rag sources surfaced", len(resp.sources) >= 1)
    check("unknown tool reported to LLM", "Unknown tool" in seen[-1]["content"]
          or any("Unknown tool" in m.get("content", "") for m in [seen[-1]]))

    # runaway tool loop is capped
    loop = {"content": None, "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "forecast_temperature", "arguments": "{}"}}]}
    calls = []

    def looping(messages, schemas):
        calls.append(len(schemas))
        return loop if schemas else {"content": "forced", "tool_calls": None}

    resp = WeatherAgent(looping, registry).answer("q")
    check("step cap forces final answer", resp.answer == "forced" and calls[-1] == 0, str(calls))

    # bad JSON args don't crash
    bad = iter([{"content": None, "tool_calls": [
        {"id": "b", "type": "function", "function": {"name": "get_live_weather", "arguments": "{not json"}}]},
        {"content": "ok", "tool_calls": None}])
    resp = WeatherAgent(lambda m, s: next(bad), registry).answer("q")
    check("bad tool args handled", resp.answer == "ok")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
