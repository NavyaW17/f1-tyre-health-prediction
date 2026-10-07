"""Load the versioned, joblib-serialized model artifact."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "tyre_model.pkl"


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    """Load the model artifact once per process; configure with MODEL_PATH."""
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `make train` first."
        )
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"Invalid model artifact at {model_path}")
    logger.info("Loaded tyre prediction model from %s", model_path)
    return artifact
