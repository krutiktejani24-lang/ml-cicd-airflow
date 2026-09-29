import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

from . import config


def evaluate_model(
    model_path: Path = config.MODEL_PATH,
    test_csv: Path = config.TEST_CSV,
    metrics_path: Path = config.METRICS_PATH,
) -> dict:
    model = joblib.load(model_path)
    test = pd.read_csv(test_csv)
    X, y = test.drop(columns=[config.TARGET]), test[config.TARGET]
    accuracy = round(float(accuracy_score(y, model.predict(X))), 4)
    metrics = {"accuracy": accuracy, "n_test": len(test)}
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Evaluation: {metrics}")
    return metrics


def load_best_accuracy(best_path: Path = config.BEST_METRICS_PATH) -> Optional[float]:
    best_path = Path(best_path)
    if not best_path.exists():
        return None
    return float(json.loads(best_path.read_text())["accuracy"])


def should_deploy(new_accuracy: float, best_path: Path = config.BEST_METRICS_PATH) -> bool:
    """Deploy if there is no production model yet, or accuracy strictly improved."""
    best = load_best_accuracy(best_path)
    return best is None or new_accuracy > best
