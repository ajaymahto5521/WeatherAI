"""
app/schemas/api.py
==================
Pydantic request and response schemas for all API endpoints.

Why Pydantic schemas?
- They define the exact shape of data flowing in and out of the API.
- FastAPI reads these to automatically validate incoming JSON.
- FastAPI reads these to generate OpenAPI documentation at /docs.
- If a request doesn't match the schema, FastAPI returns HTTP 422
  with a precise error message — before your code even runs.
- They act as a contract between the client and the server.

Pydantic v2 changes from v1:
- Use `model_config` instead of inner `class Config`.
- Use `model_validator` instead of `@validator`.
- Field aliases work the same way.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Health Check
# =============================================================================

class HealthResponse(BaseModel):
    """
    Response schema for GET /health.

    This is intentionally minimal — health checks should be fast and simple.
    """
    status: str = Field(
        description="Service status. 'ok' means the service is running."
    )
    environment: str = Field(
        description="Current runtime environment (development/staging/production)."
    )
    version: str = Field(
        description="Application version string."
    )


# =============================================================================
# /ask endpoint — Request
# =============================================================================

class AskRequest(BaseModel):
    """
    Request body for POST /ask.

    The client sends a JSON object with a 'question' field.

    Example:
        {"question": "What does a severe thunderstorm warning mean?"}
    """
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The weather question to answer.",
        examples=["What does a severe thunderstorm warning mean?"]
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        """
        Reject questions that are only whitespace.

        Pydantic's min_length=1 allows "   " (3 spaces).
        This validator strips and checks the result.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question must not be empty or whitespace only.")
        return stripped  # return the stripped version so downstream code is clean


# =============================================================================
# /ask endpoint — Response components
# =============================================================================

class SourceDocument(BaseModel):
    """
    A single retrieved source document chunk.

    Included in the response so the user can see WHERE the answer came from.
    This is the anti-hallucination mechanism — every factual claim should be
    traceable to a source.
    """
    document: str = Field(description="The source document name or filename.")
    section: str = Field(description="The section or heading within the document.")
    relevance_score: float | None = Field(
        default=None,
        description="Cosine similarity score (0.0–1.0). Higher = more relevant."
    )


class ModelPrediction(BaseModel):
    """
    Output from the trained ML forecasting model (Milestone 1+).

    Included when the agent decides to call the forecasting tool.
    """
    predicted_temperature: float | None = Field(
        default=None,
        description="Predicted temperature in Celsius."
    )
    prediction_date: str | None = Field(
        default=None,
        description="The date/time this prediction is for (ISO 8601)."
    )
    location: str | None = Field(
        default=None,
        description="Location the prediction applies to."
    )
    model_name: str | None = Field(
        default=None,
        description="Name/version of the model that produced this prediction."
    )


class AskResponse(BaseModel):
    """
    Response schema for POST /ask.

    Design principle: every piece of information in 'answer' should be
    traceable through 'sources' or 'tools_used'. This prevents the LLM
    from inventing information that the user mistakes for a grounded answer.

    Fields:
        answer        — The plain-language answer to the question.
        sources       — Document chunks that grounded the answer (from RAG).
        tools_used    — Names of tools the agent invoked to answer this question.
        model_prediction — Structured output from the forecasting model, if used.
        severity      — Severity classification result, if applicable.
    """
    answer: str = Field(
        description="Plain-language answer to the question."
    )
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Retrieved document chunks used to ground the answer."
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Names of tools the agent called to answer this question.",
        examples=[["rag_retriever"], ["live_weather_tool", "rag_retriever"]]
    )
    model_prediction: ModelPrediction | None = Field(
        default=None,
        description="Structured output from the forecasting model, if invoked."
    )
    severity: str | None = Field(
        default=None,
        description="Classified severity level (LOW/MODERATE/SEVERE/CRITICAL), if applicable."
    )


# =============================================================================
# Error response (used for consistent error formatting)
# =============================================================================

class ErrorResponse(BaseModel):
    """
    Standard error response body.

    FastAPI returns this shape for application-level errors.
    HTTP 4xx/5xx status codes use this schema for the body.
    """
    error: str = Field(description="Short error code or type.")
    detail: str = Field(description="Human-readable explanation of what went wrong.")
    request_id: str | None = Field(
        default=None,
        description="Trace ID for correlating logs. Added in a later milestone."
    )
