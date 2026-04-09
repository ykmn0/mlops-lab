from fastapi.testclient import TestClient

import app as app_module

app = app_module.app


class _DummyModel:
    def predict(self, _):
        class _DummyPrediction:
            def tolist(self):
                return [0]

        return _DummyPrediction()


class _DummyModelVersion:
    def __init__(self, version: str):
        self.version = version


def _metric_value(metrics_text: str, metric_name: str) -> float:
    for line in metrics_text.splitlines():
        if line.startswith(f"{metric_name} "):
            return float(line.split(" ", maxsplit=1)[1])
    raise AssertionError(f"metric {metric_name} not found")


def test_health(monkeypatch):
    # Keep this test isolated from external MLflow state by forcing
    # successful model load during app startup.
    monkeypatch.setattr(
        app_module.mlflow.pyfunc,
        "load_model",
        lambda _: _DummyModel(),
    )
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_name" in data
    assert "model_version" in data
    assert "model_loaded" in data


def test_ready(monkeypatch):
    monkeypatch.setattr(
        app_module.mlflow.pyfunc,
        "load_model",
        lambda _: _DummyModel(),
    )
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


def test_health_returns_degraded_when_model_missing(monkeypatch):
    def _raise_load_error(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.mlflow.pyfunc, "load_model", _raise_load_error)
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_info_uses_registry_alias_resolution(monkeypatch):
    monkeypatch.setattr(
        app_module.mlflow.pyfunc,
        "load_model",
        lambda _: _DummyModel(),
    )
    monkeypatch.setattr(
        app_module.MlflowClient,
        "get_model_version_by_alias",
        lambda *_: _DummyModelVersion("42"),
    )
    with TestClient(app) as client:
        response = client.get("/info")

    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["model_name"] == "iris-model"
    assert data["model_version"] == "42"
    assert data["model_source"].startswith("mlflow:")


def test_info_falls_back_when_alias_resolution_fails(monkeypatch):
    monkeypatch.setattr(
        app_module.mlflow.pyfunc,
        "load_model",
        lambda _: _DummyModel(),
    )

    def _raise_alias_error(*_):
        raise RuntimeError("alias lookup failed")

    monkeypatch.setattr(
        app_module.MlflowClient,
        "get_model_version_by_alias",
        _raise_alias_error,
    )
    with TestClient(app) as client:
        ready = client.get("/ready")
        info = client.get("/info")

    assert ready.status_code == 200
    assert info.status_code == 200
    data = info.json()
    assert data["model_loaded"] is True
    assert data["model_name"] == app_module.MODEL_NAME
    assert data["model_version"] == app_module.MODEL_VERSION


def test_predict_success(monkeypatch):
    monkeypatch.setattr(
        app_module.mlflow.pyfunc,
        "load_model",
        lambda _: _DummyModel(),
    )
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


def test_predict_validation_error(monkeypatch):
    monkeypatch.setattr(
        app_module.mlflow.pyfunc,
        "load_model",
        lambda _: _DummyModel(),
    )
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


def test_metrics_prometheus_format(monkeypatch):
    monkeypatch.setattr(
        app_module.mlflow.pyfunc,
        "load_model",
        lambda _: _DummyModel(),
    )
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

    monkeypatch.setattr(
        app_module.mlflow.pyfunc,
        "load_model",
        lambda _: BrokenModel(),
    )
    with TestClient(app) as client:
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
