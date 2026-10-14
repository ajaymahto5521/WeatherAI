from fastapi.testclient import TestClient

from app import monitoring
from app.main import app


def test_metrics_endpoint_counts_requests():
    monitoring.reset()
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    body = client.get("/metrics").text
    assert 'http_requests_total{method="GET",path="/health",status="200"} 1' in body
    assert 'http_request_duration_seconds_count{method="GET",path="/health"} 1' in body


def test_log_forecast_writes_jsonl(tmp_path, monkeypatch):
    log = tmp_path / "logs" / "p.jsonl"
    monkeypatch.setattr(monitoring, "PREDICTION_LOG", log)
    monitoring.log_forecast({"humidity": 40.0}, 31.5, "RidgeRegression")
    assert '"prediction": 31.5' in log.read_text(encoding="utf-8")
