"""Shared fixtures. Network is always stubbed; the RAG index is built in a tmp dir."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.agents import tools
from app.rag.ingestion import ingest
from app.rag.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("rag_index")
    ingest(index_dir=d, backend="local", embedding_backend="tfidf")
    return d


@pytest.fixture(scope="session")
def retriever(index_dir):
    return Retriever(index_dir=index_dir, backend="local")


@pytest.fixture
def now():
    return datetime.now(tools.IST).replace(minute=0, second=0, microsecond=0, tzinfo=None)


@pytest.fixture
def hourly(now):
    n = 73
    times = [(now - timedelta(hours=72 - k)).isoformat(timespec="minutes") for k in range(n)]
    return {
        "time": times,
        "temperature_2m": [25 + 0.1 * k for k in range(n)],
        "relative_humidity_2m": [50.0] * n, "windspeed_10m": [8.0] * n,
        "precipitation": [0.0] * n, "cloudcover": [20.0] * n,
        "pressure_msl": [1008.0] * n, "shortwave_radiation": [300.0] * n,
        "et0_fao_evapotranspiration": [0.2] * n,
    }


CALM = {"name": "Pune", "main": {"temp": 26, "feels_like": 27, "humidity": 50, "pressure": 1010},
        "weather": [{"description": "clear skies with a light breeze"}], "wind": {"speed": 2}}


@pytest.fixture
def stub_network(monkeypatch, hourly):
    """Route tool HTTP calls to canned payloads; returns a dict to swap the OWM reply."""
    state = {"owm": CALM}
    monkeypatch.setattr(
        tools, "_http_get_json",
        lambda url, params: {"hourly": hourly} if "open-meteo" in url else state["owm"],
    )
    monkeypatch.setattr(tools.settings, "openweather_api_key", "test-key")
    return state


def tool_call(id_, name, args=None, raw=None):
    return {"id": id_, "type": "function",
            "function": {"name": name, "arguments": raw if raw is not None else json.dumps(args or {})}}
