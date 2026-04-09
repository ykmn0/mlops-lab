import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import generate_latest
from pydantic import BaseModel
from pydantic import field_validator

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_NAME = os.getenv("MODEL_NAME", "iris-random-forest")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_URI = os.getenv("MODEL_URI", "models:/iris-model@champion")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
model_load_error = None
model_source = "none"
loaded_model_name = MODEL_NAME
loaded_model_version = MODEL_VERSION

REQUESTS_TOTAL = Counter("iris_requests_total", "Total prediction requests.")
PREDICTION_ERRORS_TOTAL = Counter(
    "iris_prediction_errors_total",
    "Total failed predictions.",
)
REQUEST_VALIDATION_ERRORS_TOTAL = Counter(
    "iris_request_validation_errors_total",
    "Total request validation errors.",
)
SUCCESSFUL_PREDICTIONS_TOTAL = Counter(
    "iris_successful_predictions_total",
    "Total successful predictions.",
)
PREDICTION_DURATION_MS = Histogram(
    "iris_prediction_duration_milliseconds",
    "Prediction duration in milliseconds.",
)
MODEL_INFO = Gauge(
    "iris_model_info",
    "Model metadata.",
    labelnames=("model_name", "model_version"),
)


def _resolve_registry_metadata(model_uri: str) -> tuple[str, str]:
    """Resolve model name/version from models:/ URI when possible."""
    if not model_uri.startswith("models:/"):
        return MODEL_NAME, MODEL_VERSION

    target = unquote(model_uri.removeprefix("models:/"))
    if "@" in target:
        model_name, alias = target.split("@", maxsplit=1)
        version = MlflowClient().get_model_version_by_alias(model_name, alias).version
        return model_name, str(version)
    if "/" in target:
        model_name, version = target.split("/", maxsplit=1)
        return model_name, version

    return target, MODEL_VERSION


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, model_load_error, model_source, loaded_model_name, loaded_model_version
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        model_source = f"mlflow:{MODEL_URI}"
        model_load_error = None
        loaded_model_name, loaded_model_version = _resolve_registry_metadata(MODEL_URI)
    except Exception as mlflow_error:
        model = None
        model_source = "none"
        model_load_error = str(mlflow_error)
        loaded_model_name = MODEL_NAME
        loaded_model_version = MODEL_VERSION
        logger.exception("failed to load model from MLflow uri=%s", MODEL_URI)

    MODEL_INFO.labels(
        model_name=loaded_model_name,
        model_version=loaded_model_version,
    ).set(1)
    logger.info(
        "startup model_name=%s model_version=%s model_loaded=%s model_source=%s",
        loaded_model_name,
        loaded_model_version,
        model is not None,
        model_source,
    )
    yield


app = FastAPI(lifespan=lifespan)


class IrisInput(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_features_len(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("expected 4 features")
        return value


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    model_loaded: bool


class PredictResponse(BaseModel):
    prediction: list[int]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    is_loaded = model is not None
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_name=loaded_model_name,
        model_version=loaded_model_version,
        model_loaded=is_loaded,
    )


@app.get("/ready")
def ready() -> dict[str, str]:
    if model is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, str | bool]:
    return {
        "model_name": loaded_model_name,
        "model_version": loaded_model_version,
        "model_loaded": model is not None,
        "model_source": model_source,
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    REQUEST_VALIDATION_ERRORS_TOTAL.inc()
    logger.warning(
        "validation failed path=%s errors=%s",
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(data: IrisInput) -> PredictResponse:
    REQUESTS_TOTAL.inc()
    if model is None:
        PREDICTION_ERRORS_TOTAL.inc()
        if model_load_error:
            logger.error("predict requested but model not loaded: %s", model_load_error)
        raise HTTPException(status_code=503, detail="model is not loaded")

    try:
        with PREDICTION_DURATION_MS.time():
            prediction = model.predict([data.features]).tolist()
    except Exception:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.exception("prediction failed input=%s", data.features)
        raise HTTPException(status_code=500, detail="internal prediction error")

    SUCCESSFUL_PREDICTIONS_TOTAL.inc()
    logger.info("input=%s, pred=%s", data.features, prediction)
    return PredictResponse(prediction=prediction)
