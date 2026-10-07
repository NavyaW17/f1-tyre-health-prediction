"""Request and response schemas for the prediction API."""

from pydantic import BaseModel, ConfigDict, Field


class TyreInput(BaseModel):
    """Validated vehicle and tyre measurements used by the model."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mileage_km: float = Field(ge=0, le=100_000, description="Vehicle mileage in kilometres")
    tyre_pressure_psi: float = Field(ge=20, le=40, description="Tyre pressure in PSI")
    temperature_c: float = Field(ge=10, le=70, description="Ambient temperature in Celsius")
    tyre_age_months: float = Field(ge=0, le=60, description="Tyre age in months")
    average_speed_kmh: float = Field(ge=20, le=140, description="Average speed in km/h")
    road_quality_score: float = Field(ge=1, le=10, description="Road quality, where 10 is best")


class PredictionResponse(BaseModel):
    """Estimated remaining life and local model explanation."""

    remaining_life_percentage: float = Field(ge=0, le=100)
    health_status: str
    top_factors: list[str]
    model_version: str
