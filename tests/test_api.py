from fastapi.testclient import TestClient

import app as app_module

app = app_module.app

def _metric_value(metrics_text: str, metric_name: str) -> float:
    for line in metrics_text.splitlines():
        if line.startswith(f"{metric_name} "):
            return float(line.split(" ", maxsplit=1)[1])
    raise AssertionError(f"metric {metric_name} not found")


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_name" in data
    assert "model_version" in data
    assert "model_loaded" in data


def test_ready():
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_returns_503_when_model_missing(monkeypatch):
    def _raise_load_error(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.mlflow.pyfunc, "load_model", _raise_load_error)
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "model is not loaded"


def test_predict_success():
    with TestClient(app) as client:
        before = client.get("/metrics")
        before_count = _metric_value(
            before.text, "iris_successful_predictions_total"
        )

        response = client.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4, 0.2]},
        )

        after = client.get("/metrics")
        after_count = _metric_value(
            after.text, "iris_successful_predictions_total"
        )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)
    assert after_count >= before_count + 1


def test_predict_validation_error():
    with TestClient(app) as client:
        before = client.get("/metrics")
        before_count = _metric_value(
            before.text, "iris_request_validation_errors_total"
        )

        response = client.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4]},
        )

        after = client.get("/metrics")
        after_count = _metric_value(
            after.text, "iris_request_validation_errors_total"
        )

    assert response.status_code == 422
    assert "expected 4 features" in response.text
    assert after_count >= before_count + 1


def test_metrics_prometheus_format():
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "iris_requests_total" in response.text
    assert "iris_prediction_errors_total" in response.text
    assert "iris_request_validation_errors_total" in response.text
    assert "iris_successful_predictions_total" in response.text


def test_predict_internal_error_returns_500(monkeypatch):
    class BrokenModel:
        def predict(self, _):
            raise RuntimeError("boom")

    with TestClient(app) as client:
        monkeypatch.setattr(app_module, "model", BrokenModel())
        response = client.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4, 0.2]},
        )
    assert response.status_code == 500
    assert response.json()["detail"] == "internal prediction error"


def test_predict_returns_503_when_model_missing(monkeypatch):
    def _raise_load_error(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.mlflow.pyfunc, "load_model", _raise_load_error)
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4, 0.2]},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "model is not loaded"
