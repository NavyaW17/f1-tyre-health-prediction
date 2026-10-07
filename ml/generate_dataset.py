"""Generate a reproducible synthetic tyre degradation dataset."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "ml" / "dataset.csv"
FEATURE_NAMES = (
    "mileage_km",
    "tyre_pressure_psi",
    "temperature_c",
    "tyre_age_months",
    "average_speed_kmh",
    "road_quality_score",
)


def generate_dataset(rows: int = 10_000, seed: int = 42) -> list[dict[str, float]]:
    """Build synthetic measurements and a noisy, bounded remaining-life target.

    The target is a transparent simulation for demonstration and must not be
    interpreted as a calibrated safety model for real-world tyre decisions.
    """
    if rows < 100:
        raise ValueError("rows must be at least 100")
    rng = random.Random(seed)
    samples: list[dict[str, float]] = []
    for _ in range(rows):
        sample = {
            "mileage_km": rng.uniform(0, 100_000),
            "tyre_pressure_psi": rng.uniform(20, 40),
            "temperature_c": rng.uniform(10, 70),
            "tyre_age_months": rng.uniform(0, 60),
            "average_speed_kmh": rng.uniform(20, 140),
            "road_quality_score": rng.uniform(1, 10),
        }
        pressure = sample["tyre_pressure_psi"]
        pressure_penalty = max(0, 30 - pressure) * 1.15
        pressure_penalty += max(0, pressure - 35) * 0.9
        temperature_penalty = max(0, abs(sample["temperature_c"] - 25) - 12) * 0.22
        remaining_life = (
            100
            - sample["mileage_km"] * 0.00068
            - sample["tyre_age_months"] * 0.38
            - (10 - sample["road_quality_score"]) * 1.7
            - temperature_penalty
            - pressure_penalty
            - max(0, sample["average_speed_kmh"] - 100) * 0.06
            + rng.gauss(0, 3.5)
        )
        sample["remaining_life_percentage"] = min(100, max(0, remaining_life))
        samples.append({name: round(value, 3) for name, value in sample.items()})
    return samples


def write_dataset(output: Path, samples: list[dict[str, float]]) -> None:
    """Write generated samples in the stable training feature order."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=[*FEATURE_NAMES, "remaining_life_percentage"]
        )
        writer.writeheader()
        writer.writerows(samples)


def main() -> None:
    """Write the generated dataset to the configured output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_dataset(args.output, generate_dataset(args.rows, args.seed))
    print(f"Wrote {args.rows:,} rows to {args.output}")


if __name__ == "__main__":
    main()
