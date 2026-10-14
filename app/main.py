"""FastAPI entry point. Run with: uvicorn app.main:app --reload"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.api.routes import router
from app import monitoring
from app.config import settings, setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    logger.info("Weather Intelligence Assistant starting up")
    logger.info("Environment : %s", settings.app_env)
    logger.info("Debug mode  : %s", settings.app_debug)
    logger.info("Log level   : %s", settings.app_log_level)
    logger.info("Host:Port   : %s:%s", settings.app_host, settings.app_port)
    logger.info("Startup complete. Service is ready.")

    yield

    logger.info("Weather Intelligence Assistant shutting down...")
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Weather Intelligence Assistant",
    description=(
        "An end-to-end AI assistant that answers weather questions using "
        "RAG over authoritative documents, a trained ML forecasting model, "
        "live weather data, and a fine-tuned severity classifier."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Wide open in debug only; no cross-origin access otherwise.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        # use the route template so path parameters don't explode label cardinality
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        monitoring.record_request(request.method, path, status_code, time.perf_counter() - start)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(monitoring.render_metrics())


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unhandled errors and return a JSON 500; details only shown in debug."""
    logger.error(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
        exc_info=True,
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


app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse(
        content={
            "message": "Weather Intelligence Assistant API",
            "docs": "/docs",
            "health": "/health",
        }
    )
