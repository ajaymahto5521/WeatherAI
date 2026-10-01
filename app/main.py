"""
app/main.py
===========
FastAPI application entry point for the Weather Intelligence Assistant.

This file:
  1. Creates the FastAPI application instance.
  2. Registers the lifespan handler (startup + shutdown logic).
  3. Registers API routers (the actual routes).
  4. Adds any global middleware (CORS, error handling, etc.).

It does NOT contain business logic — that lives in app/services/ and app/agents/.

How to run:
    # Development (hot reload — auto-restarts when you save a file)
    uvicorn app.main:app --reload

    # Production (no reload, multiple workers)
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

The "app.main:app" argument means:
    "In the Python module app.main, find the object named app."
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings, setup_logging

# Get a logger for this module.
# setup_logging() hasn't run yet here — it runs inside lifespan below.
logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan — startup and shutdown logic
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    AsyncContextManager that runs at application startup and shutdown.

    FastAPI calls everything BEFORE `yield` at startup.
    FastAPI calls everything AFTER `yield` at shutdown.

    Why use lifespan instead of @app.on_event("startup")?
    The @on_event approach is deprecated in FastAPI 0.95+.
    Lifespan is the modern replacement — it uses standard Python
    context manager syntax and is easier to reason about.

    What we do here:
    Startup:
      - Configure logging (must be first so all subsequent logs work)
      - Log application info
      - In later milestones: load ML models into memory once

    Shutdown:
      - Log that we're shutting down
      - In later milestones: close DB connections, flush buffers
    """
    # ---- STARTUP ----
    setup_logging()  # Configure logging FIRST — everything after this can log

    logger.info("=" * 60)
    logger.info("Weather Intelligence Assistant starting up")
    logger.info("Environment : %s", settings.app_env)
    logger.info("Debug mode  : %s", settings.app_debug)
    logger.info("Log level   : %s", settings.app_log_level)
    logger.info("Host:Port   : %s:%s", settings.app_host, settings.app_port)
    logger.info("=" * 60)

    # Placeholder for future startup tasks:
    # logger.info("Loading forecasting model...")
    # forecasting_model = load_forecasting_model(settings.forecasting_model_path)
    # app.state.forecasting_model = forecasting_model

    # logger.info("Connecting to PostgreSQL...")
    # app.state.db_pool = await create_db_pool(settings.postgres_url)

    logger.info("Startup complete. Service is ready.")

    yield  # <-- Application runs here (handling requests)

    # ---- SHUTDOWN ----
    logger.info("Weather Intelligence Assistant shutting down...")

    # Placeholder for future cleanup:
    # await app.state.db_pool.close()

    logger.info("Shutdown complete.")


# =============================================================================
# FastAPI Application Instance
# =============================================================================

app = FastAPI(
    title="Weather Intelligence Assistant",
    description=(
        "An end-to-end AI assistant that answers weather questions using "
        "RAG over authoritative documents, a trained ML forecasting model, "
        "live weather data, and a fine-tuned severity classifier."
    ),
    version="0.1.0",
    # Lifespan wires in our startup/shutdown logic
    lifespan=lifespan,
    # docs_url="/docs" is the default — interactive Swagger UI
    # redoc_url="/redoc" is also generated automatically
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# Middleware
# =============================================================================

# CORS — Cross-Origin Resource Sharing
# This allows browser-based frontends to call our API.
# In development we allow all origins. TIGHTEN this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_debug else [],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Global Exception Handler
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for unhandled exceptions.

    Without this, FastAPI returns a raw 500 error with minimal info.
    With this, we:
      1. Log the full traceback (visible in server logs)
      2. Return a consistent JSON error body to the client
      3. Never expose internal details to the client in production

    The `exc_info=True` argument tells the logger to include the
    full Python traceback in the log output.
    """
    logger.error(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
        exc_info=True,  # includes the full traceback in the log
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "detail": (
                str(exc) if settings.app_debug
                else "An unexpected error occurred. Check server logs."
            ),
        },
    )


# =============================================================================
# Register Routers
# =============================================================================

# Include our routes — all endpoints defined in app/api/routes.py
# are now accessible at their defined paths.
#
# We do NOT use a prefix like "/api/v1" for now to keep URLs simple.
# When we add a v2 API, we'd use:
#   app.include_router(router_v1, prefix="/api/v1")
#   app.include_router(router_v2, prefix="/api/v2")
app.include_router(router)


# =============================================================================
# Root redirect
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """
    Redirect root URL to the interactive API docs.
    `include_in_schema=False` hides this from the OpenAPI spec.
    """
    return JSONResponse(
        content={
            "message": "Weather Intelligence Assistant API",
            "docs": "/docs",
            "health": "/health",
        }
    )
