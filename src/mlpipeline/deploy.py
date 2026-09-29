import shutil
from pathlib import Path

from . import config


def promote_model(
    model_path: Path = config.MODEL_PATH,
    metrics_path: Path = config.METRICS_PATH,
    prod_dir: Path = config.PROD_DIR,
) -> Path:
    """'Deploy' = copy the model + its metrics into models/production."""
    prod_dir = Path(prod_dir)
    prod_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_path, prod_dir / "model.joblib")
    shutil.copy(metrics_path, prod_dir / "metrics.json")
    print(f"Model promoted to {prod_dir}")
    return prod_dir
