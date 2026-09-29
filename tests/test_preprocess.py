import pandas as pd

from mlpipeline.preprocess import clean, preprocess


def test_clean_drops_na_and_normalises(dirty_df):
    out = clean(dirty_df)
    assert len(out) == 2
    assert "sepal_length" in out.columns
    assert set(out["species"]) == {"setosa", "virginica"}


def test_preprocess_split_sizes(iris_csv, tmp_path):
    train_path, test_path = preprocess(iris_csv, tmp_path / "out", test_size=0.2)
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    assert len(train) + len(test) == 150
    assert len(test) == 30
