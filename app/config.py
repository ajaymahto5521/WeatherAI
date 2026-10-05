"""App settings (env / .env) and logging setup."""

import logging
import sys
from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings read from environment variables or .env (e.g. postgres_host <- POSTGRES_HOST)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment."
    )
    app_debug: bool = Field(
        default=True,
        description="Enable debug mode (verbose logging, auto-reload)."
    )
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Minimum log level to emit."
    )
    app_host: str = Field(default="0.0.0.0", description="Host to bind to.")
    app_port: int = Field(default=8000, description="Port to listen on.")

    # LLM (Azure OpenAI)
    openai_api_type: str = Field(default="azure")
    azure_openai_api_key: str = Field(
        default="",
        description="Azure OpenAI API key. Required for LLM features."
    )
    azure_openai_endpoint: str = Field(
        default="",
        description="Azure OpenAI endpoint URL."
    )
    azure_openai_deployment_name: str = Field(
        default="gpt-4o",
        description="Azure OpenAI deployment (model) name."
    )
    azure_openai_api_version: str = Field(
        default="2024-02-01",
        description="Azure OpenAI API version string."
    )

    # Database (PostgreSQL + pgvector)
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="weather_intelligence")
    postgres_user: str = Field(default="weather_user")
    postgres_password: str = Field(
        default="",
        description="PostgreSQL password. Set in .env."
    )

    # External APIs
    openweather_api_key: str = Field(
        default="",
        description="OpenWeatherMap API key. Required for live weather tool."
    )

    # Embeddings
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model used for embeddings (small, runs offline)."
    )

    # RAG
    rag_chunk_size: int = Field(
        default=500,
        description="Target character length of each document chunk."
    )
    rag_chunk_overlap: int = Field(
        default=50,
        description="Characters of overlap between consecutive chunks."
    )
    rag_top_k: int = Field(
        default=4,
        description="Number of chunks to retrieve per query."
    )

    # Trained model paths (relative to project root)
    forecasting_model_path: str = Field(
        default="models/forecaster.joblib",
        description="Path to the saved scikit-learn forecasting model."
    )
    severity_model_path: str = Field(
        default="models/severity_classifier",
        description="Directory containing the fine-tuned severity classifier."
    )

    @computed_field  # type: ignore[misc]
    @property
    def postgres_url(self) -> str:
        """SQLAlchemy connection URL (psycopg2 driver)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached Settings instance, so .env is only read once."""
    return Settings()


settings: Settings = get_settings()


def setup_logging() -> None:
    """Configure the root logger to write to stdout. Call once at startup."""
    log_level = getattr(logging, settings.app_log_level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # stdout rather than stderr so Docker picks it up as regular logs
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))
    handler.setLevel(log_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # don't stack handlers if this gets called twice
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    if not settings.app_debug:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)
