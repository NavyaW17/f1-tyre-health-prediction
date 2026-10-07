"""FastAPI application for tyre remaining-life predictions."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model_loader import load_model
from app.predictor import predict_tyre_health
from app.schemas import PredictionResponse, TyreInput

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Validate model availability at startup so bad deployments fail early."""
    try:
        load_model()
    except Exception:
        logger.exception("Unable to load prediction model during startup")
        raise
    yield


app = FastAPI(
    title="Tyre Degradation Predictor",
    version="1.0.0",
    description="Estimate remaining tyre life with feature-level SHAP explanations.",
    lifespan=lifespan,
)
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected errors while keeping internal details out of responses."""
    logger.exception("Unhandled error for %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "An internal error occurred."})


@app.get("/health", tags=["operations"])
def health_check() -> dict[str, str]:
    """Report process health; startup ensures the model is available."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
def predict(payload: TyreInput) -> PredictionResponse:
    """Return a remaining-life estimate and top SHAP contributors."""
    try:
        return PredictionResponse(**predict_tyre_health(payload))
    except (FileNotFoundError, ValueError) as exc:
        logger.exception("Prediction model unavailable")
        raise HTTPException(status_code=503, detail="Prediction model is unavailable.") from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction could not be completed.") from exc
