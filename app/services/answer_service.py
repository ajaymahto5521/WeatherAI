"""
app/services/answer_service.py
==============================
Milestone 3: ties retrieval and the LLM into a grounded answer.

Flow:  question -> Retriever (top-k chunks) -> prompt with numbered context
       -> LLM -> AskResponse(answer, sources, tools_used)

Graceful degradation:
  - No LLM configured (no Azure key / openai package missing / call fails):
    return the best retrieved passages verbatim as an extractive answer.
  - No relevant chunks (all scores below MIN_SCORE): say so instead of
    letting the LLM answer from memory.
"""

from __future__ import annotations

import logging
from typing import Callable

from app.config import settings
from app.rag.retrieval import Retriever
from app.rag.vector_store import SearchHit
from app.schemas.api import AskResponse, SourceDocument

logger = logging.getLogger(__name__)

MIN_SCORE = 0.05  # below this a chunk is treated as unrelated
NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant to that question in my weather reference "
    "documents, so I can't give a grounded answer."
)

SYSTEM_PROMPT = (
    "You are a weather assistant. Answer the user's question using ONLY the numbered "
    "context passages provided. Cite passages like [1] after the claims they support. "
    "If the context does not contain the answer, say you don't know. Be concise and "
    "use plain language."
)

# An LLM is any function (system_prompt, user_prompt) -> answer text.
LLMFunc = Callable[[str, str], str]


def azure_llm() -> LLMFunc | None:
    """Build an Azure OpenAI chat function, or None if not configured/installed."""
    if not (settings.azure_openai_api_key and settings.azure_openai_endpoint):
        return None
    try:
        from openai import AzureOpenAI
    except ImportError:
        logger.warning("openai package not installed — LLM disabled")
        return None

    client = AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    return call


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    context = "\n\n".join(
        f"[{i}] ({h.chunk.document_name} — {h.chunk.section})\n{h.chunk.text}"
        for i, h in enumerate(hits, 1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


class AnswerService:
    def __init__(self, retriever: Retriever, llm: LLMFunc | None = None):
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str) -> AskResponse:
        hits = [h for h in self.retriever.retrieve(question) if h.score >= MIN_SCORE]
        if not hits:
            return AskResponse(answer=NO_CONTEXT_ANSWER, sources=[], tools_used=["rag_retriever"])

        text = None
        if self.llm is not None:
            try:
                text = self.llm(SYSTEM_PROMPT, build_prompt(question, hits)).strip() or None
            except Exception:
                logger.exception("LLM call failed — falling back to extractive answer")
        if text is None:
            text = "(LLM unavailable — showing the most relevant reference passages)\n\n" + \
                "\n\n".join(h.chunk.text for h in hits[:2])

        # De-duplicate sources, keeping the best score per (document, section).
        best: dict[tuple[str, str], float] = {}
        for h in hits:
            key = (h.chunk.document_name, h.chunk.section)
            best[key] = max(best.get(key, 0.0), h.score)
        sources = [
            SourceDocument(document=d, section=s, relevance_score=round(sc, 4))
            for (d, s), sc in best.items()
        ]
        return AskResponse(answer=text, sources=sources, tools_used=["rag_retriever"])


_service: AnswerService | None = None


def get_answer_service() -> AnswerService:
    """Lazily build the process-wide service (loads the index on first use)."""
    global _service
    if _service is None:
        _service = AnswerService(Retriever(), azure_llm())
    return _service
