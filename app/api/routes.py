"""
app/api/routes.py
=================
API route definitions for the Weather Intelligence Assistant.

This file defines WHAT URLs exist and WHAT they do.
The actual business logic lives in app/services/ and app/agents/ —
routes are thin wrappers that handle HTTP concerns only:
  - parse the request
  - call the service
  - return the response
  - handle errors

Why separate routes from logic?
  Routes change when the API contract changes.
  Logic changes when the business rules change.
  Keeping them separate means you can test logic without HTTP,
  and change URLs without touching business logic.

Current routes:
  GET  /health   — liveness check (no auth, always public)
  POST /ask      — main question-answering endpoint (Milestone 3+)
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request, status

from app.config import settings
from app.schemas.api import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    SourceDocument,
)

# Each module gets its own logger named after the module.
# Log messages from this file show "app.api.routes" in the output.
logger = logging.getLogger(__name__)

# APIRouter is like a mini-application — a collection of related routes.
# We create it here and register it on the main FastAPI app in main.py.
# The prefix "/api/v1" will be added when we register it (in main.py).
router = APIRouter()


# =============================================================================
# GET /health
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns HTTP 200 when the service is running. "
        "Used by Docker health checks, load balancers, and monitoring."
    ),
    tags=["Infrastructure"],
)
async def health_check() -> HealthResponse:
    """
    Liveness endpoint.

    This should:
    - Always return 200 if the process is alive.
    - Respond in < 50ms (no DB or LLM calls).
    - Be accessible without authentication.

    In later milestones we can add a 'readiness' check that also
    verifies DB connectivity and model loading.
    """
    logger.debug("Health check called.")
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
        version="0.1.0",  # We'll make this dynamic in a later milestone
    )


# =============================================================================
# POST /ask
# =============================================================================

@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error — bad request body."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
    },
    summary="Ask a weather question",
    description=(
        "Submit a plain-language weather question. "
        "The agent retrieves relevant documents, optionally calls tools, "
        "and returns a grounded answer with sources."
    ),
    tags=["Assistant"],
)
async def ask(request: AskRequest) -> AskResponse:
    """
    Main question-answering endpoint.

    Current state (Milestone 0): returns a placeholder response.
    The real implementation is wired in during Milestones 3–6.

    Milestone progression:
      M0  — placeholder stub (this code)
      M3  — RAG retrieval + LLM answer
      M4  — agent with live weather + forecast tools
      M6  — full integration with severity classifier
    """
    logger.info("Received /ask request: question='%s'", request.question[:80])

    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # STUB — replaced in Milestone 3 with real RAG + LLM logic
    # ------------------------------------------------------------------
    # For now we return a clearly labelled placeholder so the endpoint
    # is callable and testable while we build the real components.
    # ------------------------------------------------------------------

    stub_answer = (
        f"[STUB — Milestone 0] You asked: '{request.question}'. "
        "The full answer service will be implemented in Milestone 3. "
        "The RAG retriever, LLM, agent, and tools are not yet connected."
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info("/ask completed in %.1f ms", elapsed_ms)

    return AskResponse(
        answer=stub_answer,
        sources=[],          # No sources yet — RAG not connected
        tools_used=[],       # No tools yet — agent not connected
        model_prediction=None,
        severity=None,
    )
