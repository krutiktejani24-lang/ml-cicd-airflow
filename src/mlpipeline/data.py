from pathlib import Path

import pandas as pd
import requests

from . import config


def _fallback_iris() -> pd.DataFrame:
    """Offline fallback so the pipeline still works without internet."""
    from sklearn.datasets import load_iris

    bunch = load_iris(as_frame=True)
    df = bunch.frame.drop(columns=["target"])
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df[config.TARGET] = [bunch.target_names[i] for i in bunch.target]
    return df


def download_data(url: str = config.DATA_URL, dest: Path = config.RAW_CSV) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"Downloaded data from {url}")
    except requests.RequestException as exc:
        print(f"Download failed ({exc}); using scikit-learn's bundled Iris data.")
        _fallback_iris().to_csv(dest, index=False)
    return dest
