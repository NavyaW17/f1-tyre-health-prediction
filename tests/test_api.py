"""HTTP contract and input validation tests."""

from fastapi.testclient import TestClient

from app import main

VALID_PAYLOAD = {
    "mileage_km": 45_000,
    "tyre_pressure_psi": 32,
    "temperature_c": 35,
    "tyre_age_months": 18,
    "average_speed_kmh": 70,
    "road_quality_score": 6,
}


def test_health_and_prediction_contract(monkeypatch) -> None:
    """Health and valid request return the documented response shapes."""
    monkeypatch.setattr(main, "load_model", lambda: {"pipeline": object()})
    monkeypatch.setattr(
        main,
        "predict_tyre_health",
        lambda _: {
            "remaining_life_percentage": 72.4,
            "health_status": "Good",
            "top_factors": ["Mileage reduced the estimate by 8.2 percentage points"],
            "model_version": "1.0.0",
        },
    )
    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "healthy"}
        response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["health_status"] == "Good"
    assert len(response.json()["top_factors"]) == 1
    assert response.json()["model_version"] == "1.0.0"


def test_prediction_rejects_out_of_range_features(monkeypatch) -> None:
    """Pydantic rejects feature measurements outside documented limits."""
    monkeypatch.setattr(main, "load_model", lambda: {"pipeline": object()})
    invalid = {**VALID_PAYLOAD, "mileage_km": 100_001}
    with TestClient(main.app) as client:
        response = client.post("/predict", json=invalid)
    assert response.status_code == 422
