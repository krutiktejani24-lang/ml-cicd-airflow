from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing rows and normalise column names / labels."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna().reset_index(drop=True)
    df[config.TARGET] = df[config.TARGET].astype(str).str.strip()
    return df


def preprocess(
    raw_csv: Path = config.RAW_CSV,
    out_dir: Path = config.PROCESSED_DIR,
    test_size: float = 0.2,
    seed: int = 42,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = clean(pd.read_csv(raw_csv))
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[config.TARGET]
    )
    train_path, test_path = out_dir / "train.csv", out_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Preprocessed: {len(train)} train rows, {len(test)} test rows")
    return train_path, test_path
