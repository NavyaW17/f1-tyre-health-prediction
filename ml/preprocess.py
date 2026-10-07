"""Dataset loading, validation, and deterministic train/test splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.utils import FEATURE_NAMES

TARGET_NAME = "remaining_life_percentage"
DATASET_PATH = Path(__file__).resolve().parent / "dataset.csv"
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "mileage_km": (0, 100_000),
    "tyre_pressure_psi": (20, 40),
    "temperature_c": (10, 70),
    "tyre_age_months": (0, 60),
    "average_speed_kmh": (20, 140),
    "road_quality_score": (1, 10),
}


def load_and_split_dataset(
    dataset_path: Path = DATASET_PATH, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Validate data, discard rows without labels, and make a held-out split.

    Feature nulls are intentionally retained for the training-only pipeline
    imputer. This avoids fitting preprocessing statistics on the test set.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    frame = pd.read_csv(dataset_path)
    expected = [*FEATURE_NAMES, TARGET_NAME]
    missing_columns = sorted(set(expected) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing_columns)}")
    frame = frame[expected].copy()
    for column in expected:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[TARGET_NAME])
    if frame.empty:
        raise ValueError("Dataset contains no rows with a valid target")
    for feature, (minimum, maximum) in FEATURE_RANGES.items():
        values = frame[feature].dropna()
        if ((values < minimum) | (values > maximum)).any():
            raise ValueError(f"Feature {feature} has values outside [{minimum}, {maximum}]")
    targets = frame[TARGET_NAME]
    if ((targets < 0) | (targets > 100)).any():
        raise ValueError("Target values must be between 0 and 100")
    return train_test_split(
        frame[list(FEATURE_NAMES)],
        targets,
        test_size=test_size,
        random_state=random_state,
    )
