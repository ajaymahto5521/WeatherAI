"""The LangGraph pipeline calls the forecaster tool only when the question needs it."""

import pytest

from app.agents.graph import NO_ANSWER, WeatherGraph, ml_can_answer
from app.agents.tools import ToolResult

PREDICTION = {
    "predicted_temperature": 31.4, "unit": "celsius", "model_name": "ridge",
    "target_time": "2026-10-10T15:00:00", "location": "New Delhi",
}


class FakeForecast:
    """Stands in for forecast_temperature and records each call."""

    def __init__(self, result=None):
        self.calls = 0
        self.result = result or ToolResult("Next-hour forecast for New Delhi: 31.4 C", PREDICTION)

    def __call__(self):
        self.calls += 1
        return self.result


@pytest.mark.parametrize("question", [
    "What will the temperature be in the next hour?",
    "Forecast the temperature for Delhi",
    "Will it be hot in the coming hour?",
])
def test_forecast_questions_call_the_tool(retriever, question):
    forecast = FakeForecast()
    resp = WeatherGraph(retriever, llm=None, forecast=forecast).answer(question)
    assert forecast.calls == 1
    assert resp.answered_by == "ml_forecaster"
    assert resp.tools_used == ["ml_forecaster"]
    assert resp.model_prediction.predicted_temperature == 31.4


@pytest.mark.parametrize("question", [
    "What does a tornado warning mean?",           # no forecast intent
    "What will the temperature be in Mumbai?",     # model only covers Delhi
    "At what wind speed must tower crane operations stop?",
])
def test_other_questions_do_not_call_the_tool(retriever, question):
    forecast = FakeForecast()
    resp = WeatherGraph(retriever, llm=None, forecast=forecast).answer(question)
    assert forecast.calls == 0
    assert "ml_forecaster" not in resp.tools_used
    assert resp.model_prediction is None


def test_document_question_is_answered_from_rag_with_sources(retriever):
    resp = WeatherGraph(retriever, llm=None, forecast=FakeForecast()).answer(
        "What water depth triggers site evacuation?")
    assert resp.answered_by == "rag_retriever"
    assert resp.sources


def test_forecaster_failure_does_not_crash_and_falls_back_to_rag(retriever):
    broken = FakeForecast(ToolResult("The forecast service is unavailable.", {}))
    resp = WeatherGraph(retriever, llm=None, forecast=broken).answer(
        "Predict the temperature in the next hour")
    assert resp.answer
    assert resp.model_prediction is None
    assert resp.tools_used[0] == "ml_forecaster"  # it was tried first


def test_unrelated_question_gets_no_answer(retriever):
    resp = WeatherGraph(retriever, llm=None, forecast=FakeForecast()).answer(
        "zxqv blorp wibble")
    assert resp.answer == NO_ANSWER
    assert resp.sources == []


def test_ml_routing_rule():
    assert ml_can_answer("forecast the temperature in the next hour")
    assert not ml_can_answer("forecast the temperature in Mumbai")
    assert not ml_can_answer("what is a heat wave")
