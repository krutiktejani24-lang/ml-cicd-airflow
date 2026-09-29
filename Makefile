.PHONY: install lint format test train airflow-up airflow-down

install:
	pip install -r requirements-dev.txt

lint:
	flake8 src tests airflow
	black --check src tests airflow

format:
	black src tests airflow

test:
	PYTHONPATH=src pytest --cov=mlpipeline

train:
	PYTHONPATH=src python -m mlpipeline.cli all

airflow-up:
	AIRFLOW_UID=$$(id -u) docker compose up -d

airflow-down:
	docker compose down
