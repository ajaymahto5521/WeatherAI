"""Thin MLflow wrapper so training scripts still run if mlflow isn't installed.

Runs go to a local SQLite store (mlflow.db) unless MLFLOW_TRACKING_URI is set.
View them with:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import mlflow
except ImportError:  # tracking is optional
    mlflow = None


@contextmanager
def start_run(experiment: str, run_name: str):
    """Open an MLflow run, or do nothing when mlflow is unavailable."""
    if mlflow is None:
        logger.warning("mlflow not installed; skipping experiment tracking")
        yield None
        return
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_params(params: dict) -> None:
    if mlflow is not None and mlflow.active_run():
        mlflow.log_params(params)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    if mlflow is not None and mlflow.active_run():
        mlflow.log_metrics(metrics, step=step)


def log_artifact(path) -> None:
    if mlflow is not None and mlflow.active_run():
        mlflow.log_artifact(str(path))
