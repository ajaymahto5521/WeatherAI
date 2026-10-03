"""End-to-end check: POST /ask -> agent -> tools -> severity. Scripted LLM, stubbed network."""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fastapi.testclient import TestClient

from app.agents import tools
from app.agents.agent import WeatherAgent
from app.main import app
from app.models.severity import SeverityClassifier
from app.rag.ingestion import ingest
from app.rag.retrieval import Retriever
from app.services import answer_service

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  [OK]   {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1


def call(id_, name, args=None):
    return {"id": id_, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


now = datetime.now(tools.IST).replace(minute=0, second=0, microsecond=0, tzinfo=None)
times = [(now - timedelta(hours=71 - k)).isoformat(timespec="minutes") for k in range(73)]
hourly = {"time": times, "temperature_2m": [25 + 0.1 * k for k in range(73)],
          "relative_humidity_2m": [50.0] * 73, "windspeed_10m": [8.0] * 73,
          "precipitation": [0.0] * 73, "cloudcover": [20.0] * 73, "pressure_msl": [1008.0] * 73,
          "shortwave_radiation": [300.0] * 73, "et0_fao_evapotranspiration": [0.2] * 73}

CYCLONE = {"name": "Puri", "main": {"temp": 29, "feels_like": 34, "humidity": 95, "pressure": 960},
           "weather": [{"description": "extremely severe cyclone making landfall"}],
           "wind": {"speed": 45}, "rain": {"1h": 120}}
CALM = {"name": "Pune", "main": {"temp": 26, "feels_like": 27, "humidity": 50, "pressure": 1010},
        "weather": [{"description": "clear skies with a light breeze"}], "wind": {"speed": 2}}
live = {"d": CALM}
tools._http_get_json = lambda url, params: {"hourly": hourly} if "open-meteo" in url else live["d"]
tools.settings.openweather_api_key = "k"

with tempfile.TemporaryDirectory() as tmp:
    ingest(index_dir=Path(tmp), backend="local", embedding_backend="tfidf")
    registry = tools.build_registry(Retriever(index_dir=Path(tmp), backend="local"))
    clf = SeverityClassifier()

    def agent_with(script):
        it = iter(script)
        return WeatherAgent(lambda m, s: next(it), registry, clf)

    client = TestClient(app)

    # 1. weather + forecast question
    answer_service._service = agent_with([
        {"content": None, "tool_calls": [call("1", "get_live_weather", {"city": "Pune"}),
                                         call("2", "forecast_temperature")]},
        {"content": "It is calm in Pune.", "tool_calls": None}])
    live["d"] = CALM
    b = client.post("/ask", json={"question": "Weather in Pune and next-hour Delhi forecast?"}).json()
    check("tools_used", b["tools_used"] == ["get_live_weather", "forecast_temperature"], str(b))
    check("severity LOW for calm conditions", b["severity"] == "LOW", str(b["severity"]))
    check("model_prediction present", b["model_prediction"] and
          b["model_prediction"]["predicted_temperature"] is not None, str(b["model_prediction"]))

    # 2. dangerous conditions
    answer_service._service = agent_with([
        {"content": None, "tool_calls": [call("1", "get_live_weather", {"city": "Puri"})]},
        {"content": "Dangerous.", "tool_calls": None}])
    live["d"] = CYCLONE
    b = client.post("/ask", json={"question": "Weather in Puri?"}).json()
    check("severity CRITICAL/SEVERE for cyclone", b["severity"] in ("CRITICAL", "SEVERE"), str(b["severity"]))

    # 3. definitional question: RAG only, no severity label
    answer_service._service = agent_with([
        {"content": None, "tool_calls": [call("1", "search_weather_knowledge",
                                              {"query": "tornado warning meaning"})]},
        {"content": "A warning means it is happening [1].", "tool_calls": None}])
    b = client.post("/ask", json={"question": "What does a tornado warning mean?"}).json()
    check("RAG question has sources, no severity", b["sources"] and b["severity"] is None, str(b))

    # 4. classifier failure doesn't break the request
    class Broken:
        def classify(self, t):
            raise RuntimeError("model gone")
    it = iter([{"content": None, "tool_calls": [call("1", "get_live_weather", {"city": "Pune"})]},
               {"content": "ok", "tool_calls": None}])
    answer_service._service = WeatherAgent(lambda m, s: next(it), registry, Broken())
    live["d"] = CALM
    r = client.post("/ask", json={"question": "Pune weather?"})
    check("severity failure -> 200 with null severity", r.status_code == 200 and r.json()["severity"] is None)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
