"""In-process service metrics (Prometheus text format) and a forecast prediction log.

Metrics are per-process and reset on restart; the prediction log (JSON lines) is
what you'd compare against training data to spot input drift.
"""

import json
import logging
import threading
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

PREDICTION_LOG = Path(__file__).resolve().parent.parent / "logs" / "forecast_predictions.jsonl"

_lock = threading.Lock()
_requests: dict[tuple[str, str, int], int] = defaultdict(int)
_latency_sum: dict[tuple[str, str], float] = defaultdict(float)
_latency_count: dict[tuple[str, str], int] = defaultdict(int)


def record_request(method: str, path: str, status_code: int, seconds: float) -> None:
    with _lock:
        _requests[(method, path, status_code)] += 1
        _latency_sum[(method, path)] += seconds
        _latency_count[(method, path)] += 1


def render_metrics() -> str:
    lines = ["# TYPE http_requests_total counter"]
    with _lock:
        for (method, path, code), n in sorted(_requests.items()):
            lines.append(f'http_requests_total{{method="{method}",path="{path}",status="{code}"}} {n}')
        lines.append("# TYPE http_request_duration_seconds summary")
        for (method, path), total in sorted(_latency_sum.items()):
            labels = f'method="{method}",path="{path}"'
            lines.append(f"http_request_duration_seconds_sum{{{labels}}} {total:.6f}")
            lines.append(f"http_request_duration_seconds_count{{{labels}}} {_latency_count[(method, path)]}")
    return "\n".join(lines) + "\n"


def reset() -> None:
    with _lock:
        _requests.clear()
        _latency_sum.clear()
        _latency_count.clear()


def log_forecast(features: dict, prediction: float, model_type: str) -> None:
    """Append one forecast (inputs + output) to the prediction log; never raises."""
    try:
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTION_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"features": features, "prediction": prediction,
                                "model_type": model_type}) + "\n")
    except OSError:
        logger.warning("Could not write forecast prediction log", exc_info=True)
