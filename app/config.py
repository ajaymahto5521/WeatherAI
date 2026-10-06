"""App settings (env / .env) and logging setup."""

import logging
import sys
from functools import lru_cache
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read from environment variables or .env (postgres_host <- POSTGRES_HOST)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = True
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Azure OpenAI (LLM features stay off while the key is empty)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment_name: str = "gpt-4o"
    azure_openai_api_version: str = "2024-02-01"

    # PostgreSQL + pgvector
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "weather_intelligence"
    postgres_user: str = "weather_user"
    postgres_password: str = ""

    # OpenWeatherMap (live weather tool)
    openweather_api_key: str = ""

    # RAG chunking and retrieval
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 100
    rag_top_k: int = 4

    # Trained models (relative to project root)
    forecasting_model_path: str = "models/forecaster.joblib"
    severity_model_path: str = "models/severity_classifier"

    @computed_field  # type: ignore[misc]
    @property
    def postgres_url(self) -> str:
        """SQLAlchemy connection URL (psycopg2 driver)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached Settings instance, so .env is only read once."""
    return Settings()


settings: Settings = get_settings()


def setup_logging() -> None:
    """Configure the root logger to write to stdout. Call once at startup."""
    level = settings.app_log_level
    root = logging.getLogger()
    root.setLevel(level)

    # skip if already configured, so repeat calls don't stack handlers
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)  # stdout so Docker treats it as normal logs
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S"
        ))
        root.addHandler(handler)

    if not settings.app_debug:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
