"""Request/response schemas for the API."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = Field(
        description="Service status. 'ok' means the service is running."
    )
    environment: str = Field(
        description="Current runtime environment (development/staging/production)."
    )
    version: str = Field(
        description="Application version string."
    )


class AskRequest(BaseModel):
    """Request body for POST /ask."""
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The weather question to answer.",
        examples=["What does a severe thunderstorm warning mean?"]
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        # min_length=1 lets "   " through, so strip and re-check
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question must not be empty or whitespace only.")
        return stripped


class SourceDocument(BaseModel):
    """A retrieved document chunk that backs the answer."""
    document: str = Field(description="The source document name or filename.")
    section: str = Field(description="The section or heading within the document.")
    relevance_score: float | None = Field(
        default=None,
        description="Cosine similarity score (0.0–1.0). Higher = more relevant."
    )


class ModelPrediction(BaseModel):
    """Output of the ML forecasting model, when the agent used it."""
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
    """Response for POST /ask: the answer plus what it was based on."""
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


class ErrorResponse(BaseModel):
    """Standard error body."""
    error: str = Field(description="Short error code or type.")
    detail: str = Field(description="Human-readable explanation of what went wrong.")
    request_id: str | None = Field(
        default=None,
        description="Trace ID for correlating logs. Not populated yet."
    )
