"""Airflow DAG: download -> preprocess -> train -> evaluate -> (deploy | skip)."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# Make the `mlpipeline` package importable even if PYTHONPATH is not set.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mlpipeline.data import download_data  # noqa: E402
from mlpipeline.deploy import promote_model  # noqa: E402
from mlpipeline.evaluate import evaluate_model, should_deploy  # noqa: E402
from mlpipeline.preprocess import preprocess  # noqa: E402
from mlpipeline.train import train_model  # noqa: E402


def _download():
    download_data()


def _preprocess():
    preprocess()


def _train():
    n_estimators = int(Variable.get("n_estimators", default_var=100))
    seed = int(Variable.get("random_seed", default_var=42))
    return train_model(n_estimators=n_estimators, seed=seed)  # pushed to XCom


def _evaluate(ti):
    metrics = evaluate_model()
    ti.xcom_push(key="metrics", value=metrics)
    if should_deploy(metrics["accuracy"]):
        return "deploy_model"
    return "skip_deployment"


def _deploy():
    promote_model()


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="ml_training_pipeline",
    description="Download data, train a model, deploy only if accuracy improves",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "assignment"],
) as dag:
    download = PythonOperator(task_id="download_data", python_callable=_download)
    prep = PythonOperator(task_id="preprocess_data", python_callable=_preprocess)
    train = PythonOperator(task_id="train_model", python_callable=_train)
    evaluate = BranchPythonOperator(task_id="evaluate_model", python_callable=_evaluate)
    deploy = PythonOperator(task_id="deploy_model", python_callable=_deploy)
    skip = EmptyOperator(task_id="skip_deployment")

    download >> prep >> train >> evaluate >> [deploy, skip]
