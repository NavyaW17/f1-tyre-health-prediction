"""Training artifact and SHAP-backed prediction tests."""

from pathlib import Path

import joblib
from app.predictor import predict_tyre_health
from app.schemas import TyreInput
from ml.generate_dataset import generate_dataset, write_dataset
from ml.train import train_model


def test_training_artifact_predicts_and_explains(tmp_path: Path, monkeypatch) -> None:
    """A freshly trained artifact supports bounded predictions and SHAP output."""
    dataset_path = tmp_path / "dataset.csv"
    model_path = tmp_path / "tyre_model.pkl"
    write_dataset(dataset_path, generate_dataset(rows=300))
    metrics = train_model(dataset_path, model_path)
    assert metrics["mae"] >= 0
    assert model_path.exists()
    artifact = joblib.load(model_path)
    monkeypatch.setattr("app.predictor.load_model", lambda: artifact)
    result = predict_tyre_health(
        TyreInput(
            mileage_km=45_000,
            tyre_pressure_psi=32,
            temperature_c=35,
            tyre_age_months=18,
            average_speed_kmh=70,
            road_quality_score=6,
        )
    )
    assert 0 <= result["remaining_life_percentage"] <= 100
    assert result["health_status"] in {"Excellent", "Good", "Moderate", "Poor", "Critical"}
    assert len(result["top_factors"]) == 3
    assert result["model_version"] == "1.0.0"
