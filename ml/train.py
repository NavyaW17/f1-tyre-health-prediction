"""Train and evaluate the versioned XGBoost tyre-life regressor."""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from app.utils import FEATURE_NAMES
from ml.preprocess import DATASET_PATH, load_and_split_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "tyre_model.pkl"
logger = logging.getLogger(__name__)


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create the imputation and gradient-boosted tree pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "regressor",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=450,
                    max_depth=5,
                    learning_rate=0.045,
                    subsample=0.9,
                    colsample_bytree=0.95,
                    reg_lambda=2.0,
                    random_state=random_state,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def train_model(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit the model, evaluate held-out metrics, and save an artifact."""
    x_train, x_test, y_train, y_test = load_and_split_dataset(
        dataset_path=dataset_path, random_state=random_state
    )
    pipeline = build_pipeline(random_state)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }
    artifact = {
        "pipeline": pipeline,
        "model_version": "1.0.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "metrics": metrics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    logger.info("Saved model artifact to %s", model_path)
    return metrics


def main() -> None:
    """Train using optional input/output paths and print evaluation metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    metrics = train_model(args.dataset, args.output, args.seed)
    print(f"MAE:  {metrics['mae']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
