"""
app/agents/agent.py
===================
Milestone 4: tool-calling agent.

The LLM sees the question plus tool schemas and decides which tools to call.
We execute them, feed results back, and repeat until it answers (capped at
MAX_STEPS). The response records which tools ran, the RAG sources, and any
model prediction.

NOTE: README planned LangGraph; langgraph isn't installed here and this loop
is the same ReAct pattern in ~40 lines. The chat function is injected, so
swapping in a LangGraph graph later only touches this file.

A chat function takes (messages, tool_schemas) and returns an OpenAI-style
assistant message dict: {"content": str|None, "tool_calls": [...]|None}.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from app.agents.tools import TOOL_SCHEMAS, ToolResult, build_registry
from app.config import settings
from app.schemas.api import AskResponse, ModelPrediction, SourceDocument

logger = logging.getLogger(__name__)

MAX_STEPS = 4
ChatFunc = Callable[[list[dict], list[dict]], dict]

SYSTEM_PROMPT = (
    "You are a weather assistant. Use tools to get facts rather than guessing: "
    "search_weather_knowledge for definitions/alerts/India climate, get_live_weather "
    "for current conditions in a city, forecast_temperature for the next-hour "
    "temperature in New Delhi. Ground answers in tool results, cite reference "
    "passages like [1], and say so when a tool is unavailable or the data doesn't "
    "answer the question. Be concise and plain-spoken."
)


def azure_chat() -> ChatFunc | None:
    if not (settings.azure_openai_api_key and settings.azure_openai_endpoint):
        return None
    try:
        from openai import AzureOpenAI
    except ImportError:
        logger.warning("openai package not installed — agent disabled")
        return None

    client = AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )

    def chat(messages: list[dict], tools: list[dict]) -> dict:
        msg = client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=messages, tools=tools, temperature=0.2,
        ).choices[0].message
        return {
            "content": msg.content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in (msg.tool_calls or [])
            ] or None,
        }

    return chat


class WeatherAgent:
    def __init__(self, chat: ChatFunc, registry: dict[str, Callable[..., ToolResult]]):
        self.chat = chat
        self.registry = registry

    def _run_tool(self, name: str, raw_args: str) -> ToolResult:
        fn = self.registry.get(name)
        if fn is None:
            return ToolResult(f"Unknown tool '{name}'.")
        try:
            args = json.loads(raw_args or "{}")
            return fn(**args)
        except Exception as exc:  # bad args from the LLM or a tool bug — report, don't crash
            logger.exception("Tool %s failed", name)
            return ToolResult(f"Tool '{name}' failed: {exc}")

    def answer(self, question: str) -> AskResponse:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tools_used: list[str] = []
        sources: dict[tuple[str, str], SourceDocument] = {}
        prediction: ModelPrediction | None = None
        text = ""

        for _ in range(MAX_STEPS):
            msg = self.chat(messages, TOOL_SCHEMAS)
            calls = msg.get("tool_calls")
            if not calls:
                text = (msg.get("content") or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
            for call in calls:
                name = call["function"]["name"]
                result = self._run_tool(name, call["function"]["arguments"])
                if name not in tools_used:
                    tools_used.append(name)
                for s in result.data.get("sources", []):
                    sources.setdefault((s["document"], s["section"]), SourceDocument(**s))
                if name == "forecast_temperature" and "predicted_temperature" in result.data:
                    prediction = ModelPrediction(**result.data)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.text})
        else:
            # Step cap hit while the model was still calling tools: force a final answer.
            messages.append({"role": "user", "content": "Answer now using what you have."})
            text = (self.chat(messages, [])["content"] or "").strip()

        return AskResponse(
            answer=text or "I wasn't able to produce an answer.",
            sources=list(sources.values()),
            tools_used=tools_used,
            model_prediction=prediction,
        )


def create_agent(retriever) -> WeatherAgent | None:
    chat = azure_chat()
    return WeatherAgent(chat, build_registry(retriever)) if chat else None
