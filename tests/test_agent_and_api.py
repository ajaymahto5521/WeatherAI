import pytest
from fastapi.testclient import TestClient

from app.agents import tools
from app.agents.agent import MAX_STEPS, WeatherAgent
from app.agents.graph import NO_ANSWER, WeatherGraph
from app.main import app
from app.schemas.api import AskRequest
from app.services import answer_service
from tests.conftest import tool_call


class FakeClassifier:
    def __init__(self, label="SEVERE"):
        self.label, self.seen = label, []

    def classify(self, text):
        self.seen.append(text)
        return type("R", (), {"label": self.label})()


def scripted(script):
    it = iter(script)
    return lambda messages, schemas: next(it)


@pytest.fixture
def registry(retriever, stub_network):
    return tools.build_registry(retriever)


def test_agent_runs_tools_and_collects_outputs(registry):
    clf = FakeClassifier()
    chat = scripted([
        {"content": None, "tool_calls": [
            tool_call("1", "search_weather_knowledge", {"query": "tornado warning"}),
            tool_call("2", "get_live_weather", {"city": "Pune"}),
            tool_call("3", "nope")]},
        {"content": "Done [1].", "tool_calls": None}])
    r = WeatherAgent(chat, registry, clf).answer("q")
    assert r.answer == "Done [1]."
    assert r.tools_used == ["search_weather_knowledge", "get_live_weather", "nope"]
    assert r.sources and r.severity == "SEVERE" and len(clf.seen) == 1


def test_agent_reports_unknown_tool_to_llm(registry):
    captured = []
    replies = iter([{"content": None, "tool_calls": [tool_call("1", "nope")]},
                    {"content": "ok", "tool_calls": None}])

    def chat(messages, schemas):
        captured.append(list(messages))
        return next(replies)

    WeatherAgent(chat, registry).answer("q")
    assert "Unknown tool" in captured[-1][-1]["content"]


def test_agent_survives_bad_json_args(registry):
    chat = scripted([{"content": None, "tool_calls": [tool_call("1", "get_live_weather", raw="{bad")]},
                     {"content": "ok", "tool_calls": None}])
    assert WeatherAgent(chat, registry).answer("q").answer == "ok"


def test_agent_step_cap_forces_final_answer(registry):
    calls = []

    def chat(messages, schemas):
        calls.append(len(schemas))
        if schemas:
            return {"content": None, "tool_calls": [tool_call("x", "get_live_weather", {"city": "Pune"})]}
        return {"content": "forced", "tool_calls": None}

    r = WeatherAgent(chat, registry).answer("q")
    assert r.answer == "forced" and len(calls) == MAX_STEPS + 1 and calls[-1] == 0


def test_no_severity_for_rag_only_answers(registry):
    clf = FakeClassifier()
    chat = scripted([{"content": None, "tool_calls": [tool_call("1", "search_weather_knowledge", {"query": "x y"})]},
                     {"content": "a", "tool_calls": None}])
    assert WeatherAgent(chat, registry, clf).answer("q").severity is None and clf.seen == []


def test_classifier_failure_does_not_break_answer(registry):
    class Broken:
        def classify(self, t):
            raise RuntimeError("gone")

    chat = scripted([{"content": None, "tool_calls": [tool_call("1", "get_live_weather", {"city": "Pune"})]},
                     {"content": "ok", "tool_calls": None}])
    r = WeatherAgent(chat, registry, Broken()).answer("q")
    assert r.answer == "ok" and r.severity is None


def test_empty_llm_answer_gets_placeholder(registry):
    r = WeatherAgent(scripted([{"content": "  ", "tool_calls": None}]), registry).answer("q")
    assert r.answer == "I wasn't able to produce an answer."


def test_no_answer_message_does_not_include_model_scope_details(retriever):
    graph = WeatherGraph(retriever)

    response = graph.no_answer_node({"question": "Define weather?"})["answer"]

    assert response == NO_ANSWER
    assert "next-hour temperature for New Delhi" not in response


# ---------------------------------------------------------------- HTTP layer

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(answer_service, "_service", None)
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ask_uses_service(client, registry, monkeypatch):
    chat = scripted([{"content": "hi", "tool_calls": None}])
    monkeypatch.setattr(answer_service, "_service", WeatherAgent(chat, registry))
    r = client.post("/ask", json={"question": "hello"})
    assert r.status_code == 200 and r.json()["answer"] == "hi"


@pytest.mark.parametrize("body", [{"question": "   "}, {"question": ""}, {}, {"question": "x" * 2001}])
def test_ask_validation(client, body):
    assert client.post("/ask", json=body).status_code == 422


def test_ask_503_when_index_missing(client, monkeypatch):
    def missing():
        raise FileNotFoundError("run scripts/ingest_documents.py first")

    monkeypatch.setattr("app.api.routes.get_answer_service", missing)
    r = client.post("/ask", json={"question": "hello"})
    assert r.status_code == 503 and "ingest" in r.json()["detail"]


def test_question_is_stripped():
    assert AskRequest(question="  hi  ").question == "hi"
