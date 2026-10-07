# Tyre Degradation Predictor v1.0

A full-stack demonstration that estimates remaining tyre life from mileage, pressure, ambient temperature, tyre age, average speed, and road quality. The backend trains an XGBoost regressor, serves predictions with FastAPI, and returns local SHAP feature contributions. The responsive React dashboard includes a session trend and CSV export.

> **Important:** The included 10,000-row dataset is synthetic. This model is a software/ML portfolio demonstration, not a validated tyre inspection, maintenance recommendation, or safety system. Do not use it to decide whether a tyre is safe to drive on. Replace the synthetic data with representative, quality-controlled measurements and perform domain validation before any real-world use.

## Architecture

- `ml/generate_dataset.py` generates seeded synthetic samples with bounded target noise.
- `ml/preprocess.py` validates feature domains and splits data; missing feature values are imputed inside the fitted training pipeline to avoid test leakage.
- `ml/train.py` trains an XGBoost regressor, prints held-out MAE/RMSE/R², and saves the pipeline and metadata to `models/tyre_model.pkl`.
- `app/` contains request schemas, artifact loading, prediction/explanation logic, and the FastAPI application.
- `frontend/` is a Vite, React, TypeScript, and Tailwind CSS dashboard.
- `tests/` tests the API contract and a real training-to-SHAP prediction flow.

## Requirements

- Python 3.12
- Node.js 20 or newer and npm
- Docker Engine with Docker Compose (optional)

## Local Setup

From the repository root:

```bash
cp .env.example .env
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Generate the dataset (10,000 rows, seed 42):

```bash
python -m ml.generate_dataset
```

Train/evaluate the model and save `models/tyre_model.pkl`:

```bash
python -m ml.train
```

Start the API in a terminal:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env
```

The API is at `http://localhost:8000`; interactive OpenAPI documentation is at `http://localhost:8000/docs`. Health check: `GET /health`. Prediction: `POST /predict`.

Start the dashboard in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (usually `http://localhost:5173`). The dashboard defaults to `http://localhost:8000` for its API. Set `VITE_API_URL` before `npm run dev` to use a different endpoint.

## API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "mileage_km": 45000,
    "tyre_pressure_psi": 32,
    "temperature_c": 35,
    "tyre_age_months": 18,
    "average_speed_kmh": 70,
    "road_quality_score": 6
  }'
```

The response contains `remaining_life_percentage`, a status band, up to three signed SHAP contributors, and `model_version`. Inputs are range-validated; invalid values return HTTP 422.

## Docker Compose

From the project root:

```bash
cp .env.example .env
docker compose up --build
```

The API is exposed on port 8000 and the dashboard on port 5173. The API image creates its reproducible synthetic dataset and model during build. Compose waits for the API health check before starting the frontend. To stop the services, run `docker compose down`.

The browser must be able to reach the API URL. For a remote deployment, set `VITE_API_URL` in `.env` to the externally reachable API origin before building the frontend image, and set `CORS_ORIGINS` to the frontend origin. Vite embeds `VITE_API_URL` at build time.

## Tests

```bash
pytest -q
```

Build-check the frontend:

```bash
cd frontend
npm install
npm run build
```

## Data and Model Notes

The generated label is a deliberately transparent synthetic function: mileage, tyre age, low road-quality scores, pressure outside a nominal band, temperature extremes, and sustained higher speed can reduce the simulated estimate. Random noise prevents the label from being a perfectly deterministic formula. The reported test metrics measure recovery of this synthetic function only; they do not establish real-world accuracy.

`models/tyre_model.pkl` is generated and git-ignored. Rebuild it with `python -m ml.train`. Model metadata records version, training timestamp, feature order, and evaluation metrics. Joblib artifacts are executable Python serialization: only load artifacts from trusted sources.

Prediction history is held in the browser session and can be downloaded as CSV. No user data is persisted by the API. SHAP values describe how model features contributed to an individual estimate; they are not causal explanations.

## Configuration

Copy `.env.example` to `.env` and edit as needed:

- `LOG_LEVEL`: API logging level (`INFO` by default).
- `MODEL_PATH`: model artifact location (container default shown in the example).
- `CORS_ORIGINS`: comma-separated allowed browser origins.
- `VITE_API_URL`: API origin compiled into the frontend bundle.
