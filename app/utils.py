"""Shared feature metadata and health-status utilities."""

from typing import Final

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "mileage_km",
    "tyre_pressure_psi",
    "temperature_c",
    "tyre_age_months",
    "average_speed_kmh",
    "road_quality_score",
)

FEATURE_LABELS: Final[dict[str, str]] = {
    "mileage_km": "Mileage",
    "tyre_pressure_psi": "Tyre pressure",
    "temperature_c": "Temperature",
    "tyre_age_months": "Tyre age",
    "average_speed_kmh": "Average speed",
    "road_quality_score": "Road quality",
}


def health_status(remaining_life_percentage: float) -> str:
    """Map a remaining-life percentage to the user-facing health band."""
    if remaining_life_percentage >= 80:
        return "Excellent"
    if remaining_life_percentage >= 60:
        return "Good"
    if remaining_life_percentage >= 40:
        return "Moderate"
    if remaining_life_percentage >= 20:
        return "Poor"
    return "Critical"
