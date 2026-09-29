import os
from pathlib import Path

ROOT = Path(os.getenv("ML_HOME", Path(__file__).resolve().parents[2]))

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
TARGET = "species"

DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "iris.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

PROD_DIR = MODELS_DIR / "production"
PROD_MODEL_PATH = PROD_DIR / "model.joblib"
BEST_METRICS_PATH = PROD_DIR / "metrics.json"
