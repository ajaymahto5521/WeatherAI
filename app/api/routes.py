"""HTTP routes: /health and /ask."""

import logging
import time

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.services.answer_service import get_answer_service
from app.schemas.api import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    SourceDocument,
)

logger = logging.getLogger(__name__)

router = APIRouter()


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
    """Liveness check; deliberately doesn't touch the DB or LLM."""
    logger.debug("Health check called.")
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
        version="0.1.0",
    )


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
    """Answer a weather question via the agent (RAG, tools, severity)."""
    logger.info("Received /ask request: question='%s'", request.question[:80])

    start_time = time.perf_counter()

    try:
        service = get_answer_service()
    except FileNotFoundError as exc:
        # vector index hasn't been built yet
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )

    # service is blocking (embeddings + LLM), keep it off the event loop
    response = await run_in_threadpool(service.answer, request.question)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info("/ask completed in %.1f ms", elapsed_ms)
    return response
