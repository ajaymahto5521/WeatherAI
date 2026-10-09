"""LangGraph pipeline that answers a question from two sources, in order:

    START -> ml -> rag -> no_answer -> END
              \\     \\
               +-----+--> END   (as soon as one source produced an answer)

LangGraph runs the flow (state, nodes, conditional edges).
LangChain provides the pieces inside the nodes (prompt | chat model | parser).
"""

from __future__ import annotations

import logging
import re
from typing import Callable, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from app.agents.tools import forecast_temperature
from app.config import settings
from app.schemas.api import AskResponse, ModelPrediction, SourceDocument

logger = logging.getLogger(__name__)

MIN_SCORE = 0.05        # RAG chunks scoring below this are treated as unrelated
NOT_FOUND = "NOT_FOUND"  # the LLM replies with this when the context can't answer
NO_ANSWER = (
    "I couldn't find an answer to that in the available weather references or forecasting model. "
    "Please try a related weather question."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a weather assistant. Answer the question using ONLY the context below. "
     "Cite numbered passages like [1] when the context is numbered. Be concise and plain-spoken. "
     f"If the context does not answer the question, reply exactly: {NOT_FOUND}"),
    ("human", "Context ({source}):\n{context}\n\nQuestion: {question}"),
])

# The forecast IS the answer, so this prompt has no NOT_FOUND escape hatch: the model must
# turn the numbers into a direct reply ("Will it be hot?" -> yes/no + the figure).
ML_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a weather assistant. Below is a next-hour temperature forecast from a trained "
     "model. Answer the user's question directly using these numbers. For judgement words such "
     "as hot, warm or cold, decide from the temperature (about 35C+ very hot, 30-35C hot, "
     "20-30C pleasant to warm, below 20C cold) and say yes or no first. Always state the "
     "forecast temperature and the time it is for. Do not invent other data. Be concise."),
    ("human", "Forecast:\n{context}\n\nQuestion: {question}"),
])


class GraphState(TypedDict, total=False):
    """The notebook every node reads from and writes to."""
    question: str
    answer: str                         # empty until some node answers
    answered_by: str                    # "ml" | "rag"
    sources: list[SourceDocument]
    prediction: ModelPrediction | None
    tools_used: list[str]
    ml_error: str                       # why the forecaster failed, if it was tried


# ------------------------------------------------------------------ ML routing

_FORECAST_WORDS = re.compile(r"\b(forecast|predict\w*|next hour|coming hour|will be|expected)\b", re.I)
_TEMP_WORDS = re.compile(r"\b(temperature|temp|degrees|hot|warm|cold)\b", re.I)
_PLACE = re.compile(r"\b(?:in|for|at)\s+([A-Za-z]+)", re.I)
_NOT_A_PLACE = {"the", "a", "an", "this", "next", "coming", "one", "new", "delhi", "today", "tonight"}


def ml_can_answer(question: str) -> bool:
    """The trained model only forecasts next-hour temperature for New Delhi."""
    if not (_FORECAST_WORDS.search(question) and _TEMP_WORDS.search(question)):
        return False
    # "in Mumbai" -> not our model; "in the next hour" / "in Delhi" -> fine
    return all(w.lower() in _NOT_A_PLACE for w in _PLACE.findall(question))


def openai_chat_model():
    """LangChain chat model, or None when no OpenAI key is configured."""
    if not settings.openai_api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.openai_model_name, api_key=settings.openai_api_key, temperature=0.2)


# ------------------------------------------------------------------ the graph

class WeatherGraph:
    def __init__(self, retriever, llm=None, forecast: Callable[[], object] = forecast_temperature):
        self.retriever = retriever
        self.forecast = forecast
        # prompt | model | parser: a LangChain "chain". None means extractive fallback.
        parser = StrOutputParser()
        self.chain = (ANSWER_PROMPT | llm | parser) if llm is not None else None
        self.ml_chain = (ML_PROMPT | llm | parser) if llm is not None else None
        self.graph = self._build()

    # ---- helpers

    def _synthesize(self, question: str, context: str, source: str) -> str | None:
        """Ask the LLM to answer from `context`. None means the context didn't help."""
        if self.chain is None:
            return context  # no LLM: show the raw context
        try:
            text = self.chain.invoke({"question": question, "context": context, "source": source}).strip()
        except Exception:
            logger.exception("LLM call failed for source=%s", source)
            return context
        return None if not text or NOT_FOUND in text else text

    def _ml_answer(self, question: str, forecast_text: str) -> str:
        """Phrase the forecast as an answer to the question; raw forecast text if the LLM is unavailable."""
        if self.ml_chain is None:
            return forecast_text
        try:
            return self.ml_chain.invoke({"question": question, "context": forecast_text}).strip() or forecast_text
        except Exception:
            logger.exception("LLM call failed for the ML answer")
            return forecast_text

    @staticmethod
    def _used(state: GraphState, tool: str) -> list[str]:
        return [*state.get("tools_used", []), tool]

    # ---- nodes: each one tries ONE source and fills `answer` only on success

    def ml_node(self, state: GraphState) -> GraphState:
        question = state["question"]
        if not ml_can_answer(question):
            return {}
        result = self.forecast()
        used = self._used(state, "ml_forecaster")
        if "predicted_temperature" not in result.data:
            logger.warning("Forecaster failed: %s", result.text)
            return {"tools_used": used, "ml_error": result.text}  # fall through to RAG
        answer = self._ml_answer(question, result.text)
        return {"answer": answer, "answered_by": "ml", "tools_used": used,
                "prediction": ModelPrediction(**result.data)}

    def rag_node(self, state: GraphState) -> GraphState:
        question = state["question"]
        # Ignore weak matches; they are more likely to add irrelevant context than evidence.
        hits = [h for h in self.retriever.retrieve(question) if h.score >= MIN_SCORE]
        used = self._used(state, "rag_retriever")
        if not hits:
            return {"tools_used": used}
        # Number passages in the prompt so the LLM can cite its supporting evidence.
        context = "\n\n".join(
            f"[{i}] ({h.chunk.document_name} — {h.chunk.section})\n{h.chunk.text}"
            for i, h in enumerate(hits, 1)
        )
        answer = self._synthesize(question, context, "reference documents")
        if answer is None:
            return {"tools_used": used}  # documents don't cover it
        # Keep one source entry per section, using its best matching chunk score.
        best: dict[tuple[str, str], float] = {}
        for h in hits:
            key = (h.chunk.document_name, h.chunk.section)
            best[key] = max(best.get(key, 0.0), h.score)
        sources = [SourceDocument(document=d, section=s, relevance_score=round(sc, 4))
                   for (d, s), sc in best.items()]
        return {"answer": answer, "answered_by": "rag", "sources": sources, "tools_used": used}

    def no_answer_node(self, state: GraphState) -> GraphState:
        if state.get("ml_error"):  # a forecast was asked for but the model couldn't deliver
            return {"answer": f"{state['ml_error']} Please try again in a moment."}
        return {"answer": NO_ANSWER}

    # ---- wiring

    @staticmethod
    def _next_or_end(next_node: str):
        """Conditional edge: stop if this node answered, otherwise go to the next source."""
        return lambda state: END if state.get("answer") else next_node

    def _build(self):
        g = StateGraph(GraphState)
        g.add_node("ml", self.ml_node)
        g.add_node("rag", self.rag_node)
        g.add_node("no_answer", self.no_answer_node)

        g.add_edge(START, "ml")
        g.add_conditional_edges("ml", self._next_or_end("rag"), {END: END, "rag": "rag"})
        g.add_conditional_edges("rag", self._next_or_end("no_answer"), {END: END, "no_answer": "no_answer"})
        g.add_edge("no_answer", END)
        return g.compile()

    # ---- public API (same shape the API and CLI already use)

    def answer(self, question: str) -> AskResponse:
        final = self.graph.invoke({"question": question, "tools_used": [], "sources": []})
        return AskResponse(
            answer=final["answer"],
            sources=final.get("sources", []),
            tools_used=final.get("tools_used", []),
            answered_by={"ml": "ml_forecaster", "rag": "rag_retriever"}.get(final.get("answered_by", "")),
            model_prediction=final.get("prediction"),
        )
