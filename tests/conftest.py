import pandas as pd
import pytest
from sklearn.datasets import load_iris


@pytest.fixture
def iris_csv(tmp_path):
    bunch = load_iris(as_frame=True)
    df = bunch.frame.drop(columns=["target"])
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df["species"] = [bunch.target_names[i] for i in bunch.target]
    path = tmp_path / "iris.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def processed(iris_csv, tmp_path):
    from mlpipeline.preprocess import preprocess

    return preprocess(iris_csv, tmp_path / "processed")


@pytest.fixture
def dirty_df():
    return pd.DataFrame(
        {
            " Sepal_Length ": [5.1, None, 6.2],
            "species": [" setosa ", "setosa", "virginica"],
        }
    )
