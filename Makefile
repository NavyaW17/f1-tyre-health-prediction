.PHONY: data train api test

data:
	python -m ml.generate_dataset

train: data
	python -m ml.train

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env

test:
	pytest -q
