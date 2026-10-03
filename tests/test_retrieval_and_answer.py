import pytest

from app.rag.retrieval import Retriever
from app.services.answer_service import NO_CONTEXT_ANSWER, AnswerService

Q = "What is the difference between a tornado watch and a tornado warning?"


@pytest.mark.parametrize("question,doc", [
    (Q, "weather_alerts_definitions"),
    ("When does the southwest monsoon arrive in India?", "india_weather_guide"),
    ("What is relative humidity and dew point?", "meteorological_terms"),
])
def test_retrieval_finds_right_document(retriever, question, doc):
    assert any(doc in h.chunk.document_name for h in retriever.retrieve(question))


def test_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Retriever(index_dir=tmp_path, backend="local")


def test_llm_answer_with_deduped_sources(retriever):
    seen = {}

    def llm(system, user):
        seen["user"] = user
        return "Answer [1]."

    r = AnswerService(retriever, llm).answer(Q)
    assert r.answer == "Answer [1]." and r.tools_used == ["rag_retriever"]
    assert "[1]" in seen["user"] and Q in seen["user"]
    keys = [(s.document, s.section) for s in r.sources]
    assert keys and len(keys) == len(set(keys))


def test_fallback_without_llm(retriever):
    r = AnswerService(retriever, None).answer(Q)
    assert "LLM unavailable" in r.answer and r.sources


def test_fallback_when_llm_raises(retriever):
    def boom(s, u):
        raise RuntimeError("down")

    assert "LLM unavailable" in AnswerService(retriever, boom).answer(Q).answer


def test_gibberish_gets_no_context_answer(retriever):
    r = AnswerService(retriever, lambda s, u: "hallucination").answer("zzzqqq xxyyzz")
    assert r.answer == NO_CONTEXT_ANSWER and not r.sources
