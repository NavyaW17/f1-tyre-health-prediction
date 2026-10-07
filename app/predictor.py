"""Prediction orchestration and SHAP explanation formatting."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

from app.model_loader import load_model
from app.schemas import TyreInput
from app.utils import FEATURE_LABELS, FEATURE_NAMES, health_status

logger = logging.getLogger(__name__)


def explain_prediction(
    artifact: dict[str, Any], row: pd.DataFrame, transformed: np.ndarray
) -> list[str]:
    """Return the three largest signed tree-SHAP contributions."""
    estimator = artifact["pipeline"].named_steps["regressor"]
    explainer = shap.TreeExplainer(estimator)
    values = np.asarray(explainer.shap_values(transformed)).reshape(-1)
    ranked = sorted(
        zip(FEATURE_NAMES, values, strict=True), key=lambda item: abs(float(item[1])), reverse=True
    )[:3]
    factors: list[str] = []
    for feature, contribution in ranked:
        label = FEATURE_LABELS[feature]
        numeric_value = float(row.iloc[0][feature])
        if feature in {"mileage_km", "average_speed_kmh"}:
            display_value = f"{numeric_value:,.0f} km"
        elif feature == "tyre_age_months":
            display_value = f"{numeric_value:.0f} months"
        elif feature == "tyre_pressure_psi":
            display_value = f"{numeric_value:.1f} PSI"
        elif feature == "temperature_c":
            display_value = f"{numeric_value:.1f} °C"
        else:
            display_value = f"{numeric_value:.1f}/10"
        direction = "increased" if contribution >= 0 else "reduced"
        factors.append(
            f"{label} ({display_value}) {direction} the estimate by "
            f"{abs(float(contribution)):.1f} percentage points"
        )
    return factors


def predict_tyre_health(payload: TyreInput) -> dict[str, Any]:
    """Predict remaining tyre life and return local SHAP explanations."""
    artifact = load_model()
    row = pd.DataFrame([payload.model_dump()], columns=FEATURE_NAMES)
    pipeline = artifact["pipeline"]
    imputer = pipeline.named_steps["imputer"]
    transformed = imputer.transform(row)
    prediction = float(pipeline.named_steps["regressor"].predict(transformed)[0])
    remaining_life = round(float(np.clip(prediction, 0, 100)), 1)
    factors = explain_prediction(artifact, row, transformed)
    return {
        "remaining_life_percentage": remaining_life,
        "health_status": health_status(remaining_life),
        "top_factors": factors,
        "model_version": str(artifact.get("model_version", "unknown")),
    }
