"""
app/config.py
=============
Central configuration and logging for the Weather Intelligence Assistant.

How it works:
  - Settings are defined as a Pydantic BaseSettings class.
  - pydantic-settings reads values from environment variables OR a .env file.
  - Type validation happens automatically at startup — bad config fails fast.
  - A module-level `settings` singleton is imported across the app.
  - A `setup_logging()` function configures the root logger once at startup.

Usage:
    from app.config import settings, logger

    logger.info("Server started on port %s", settings.app_port)
    db_url = settings.postgres_url  # computed property
"""

import logging
import sys
from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Settings Class
# =============================================================================

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    pydantic-settings reads each field from the matching env var
    (case-insensitive). For example, `postgres_host` reads POSTGRES_HOST.

    If a required variable is missing and has no default, the app
    raises a ValidationError immediately at startup — which is exactly
    what we want. Fail early, fail loudly.
    """

    # Tells pydantic-settings to also read from a .env file in the project root.
    # `extra = "ignore"` means unknown env vars don't cause errors.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # LLM — Azure OpenAI
    # These have no defaults — the app will refuse to start without them
    # once the LLM is wired in (Milestone 3+). For now they are optional.
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Database — PostgreSQL + pgvector
    # -------------------------------------------------------------------------
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="weather_intelligence")
    postgres_user: str = Field(default="weather_user")
    postgres_password: str = Field(
        default="",
        description="PostgreSQL password. Set in .env — never hard-code."
    )

    # -------------------------------------------------------------------------
    # External APIs
    # -------------------------------------------------------------------------
    openweather_api_key: str = Field(
        default="",
        description="OpenWeatherMap API key. Required for live weather tool."
    )

    # -------------------------------------------------------------------------
    # Embedding model
    # -------------------------------------------------------------------------
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "sentence-transformers model name for generating embeddings. "
            "'all-MiniLM-L6-v2' is 80MB, fast, and works offline."
        )
    )

    # -------------------------------------------------------------------------
    # RAG settings
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Trained model paths (relative to project root)
    # -------------------------------------------------------------------------
    forecasting_model_path: str = Field(
        default="models/forecaster.joblib",
        description="Path to the saved scikit-learn forecasting model."
    )
    severity_model_path: str = Field(
        default="models/severity_classifier",
        description="Directory containing the fine-tuned severity classifier."
    )

    # -------------------------------------------------------------------------
    # Computed properties
    # These are not read from env — they are derived from other fields.
    # @computed_field means pydantic treats them as read-only derived fields.
    # -------------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def postgres_url(self) -> str:
        """
        Full SQLAlchemy-compatible PostgreSQL connection string.
        Built from individual host/port/db/user/password fields.

        Format: postgresql+psycopg2://user:password@host:port/dbname
        The '+psycopg2' part tells SQLAlchemy to use our installed driver.
        """
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def is_production(self) -> bool:
        """Convenience flag — use to guard production-only behaviour."""
        return self.app_env == "production"


# =============================================================================
# Singleton: one Settings object shared across the whole application.
# Python caches module imports, so this runs only once per process.
# =============================================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings instance.

    We use @lru_cache so that even if get_settings() is called many times,
    only one Settings object is ever created — meaning the .env file is
    read exactly once, at startup.

    FastAPI dependency injection uses this pattern:
        def my_route(settings: Settings = Depends(get_settings)):
            ...
    """
    return Settings()


# Module-level shortcut — most modules just do: from app.config import settings
settings: Settings = get_settings()


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging() -> None:
    """
    Configure the root Python logger.

    Call this exactly once, at application startup (in main.py).

    Design decisions:
    - We configure the ROOT logger so all libraries (uvicorn, sqlalchemy, etc.)
      respect our level settings.
    - We use a structured format that includes timestamp, level, module, and
      message — making logs easy to read and grep.
    - In development we output to stdout (visible in terminal).
    - In production you'd typically add a file handler or ship to a log aggregator.
    """
    log_level = getattr(logging, settings.app_log_level.upper(), logging.INFO)

    # Format:  2024-01-15 14:23:01,123 | INFO     | app.config       | Message here
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # StreamHandler sends logs to stdout.
    # We use stdout (not stderr) so Docker logs capture them correctly.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))
    handler.setLevel(log_level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging() is accidentally called twice
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Quiet down overly verbose third-party loggers in non-debug mode
    if not settings.app_debug:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


# Module-level logger for this file specifically.
# Each module creates its own logger using its __name__,
# so log messages show exactly which file they came from.
logger = logging.getLogger(__name__)
