import json

from mlpipeline.deploy import promote_model
from mlpipeline.evaluate import evaluate_model, should_deploy
from mlpipeline.train import train_model


def test_train_evaluate_and_promote(processed, tmp_path):
    train_csv, test_csv = processed
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    info = train_model(train_csv, model_path, n_estimators=20, seed=1)
    assert model_path.exists()
    assert info["train_accuracy"] > 0.9

    metrics = evaluate_model(model_path, test_csv, metrics_path)
    assert metrics["accuracy"] > 0.8

    prod = tmp_path / "prod"
    promote_model(model_path, metrics_path, prod)
    assert (prod / "model.joblib").exists()
    assert json.loads((prod / "metrics.json").read_text())["accuracy"] == metrics["accuracy"]


def test_should_deploy_logic(tmp_path):
    best = tmp_path / "metrics.json"
    assert should_deploy(0.5, best)  # no baseline yet -> deploy
    best.write_text(json.dumps({"accuracy": 0.9}))
    assert should_deploy(0.95, best)
    assert not should_deploy(0.9, best)  # equal is not an improvement
    assert not should_deploy(0.8, best)
