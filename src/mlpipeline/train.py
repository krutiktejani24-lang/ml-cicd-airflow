from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import config


def train_model(
    train_csv: Path = config.TRAIN_CSV,
    model_path: Path = config.MODEL_PATH,
    n_estimators: int = 100,
    seed: int = 42,
) -> dict:
    df = pd.read_csv(train_csv)
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    model.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    info = {
        "n_estimators": n_estimators,
        "seed": seed,
        "train_accuracy": round(float(model.score(X, y)), 4),
    }
    print(f"Model trained and saved to {model_path}: {info}")
    return info
