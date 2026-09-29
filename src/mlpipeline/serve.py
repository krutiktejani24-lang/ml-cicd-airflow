from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config

app = FastAPI(title="Iris model (staging)")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(config.PROD_MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    features: List[float]  # sepal_length, sepal_width, petal_length, petal_width


@app.get("/health")
def health():
    get_model()
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    model = get_model()
    cols = list(model.feature_names_in_)
    if len(req.features) != len(cols):
        raise HTTPException(status_code=422, detail=f"Expected {len(cols)} features: {cols}")
    pred = model.predict(pd.DataFrame([req.features], columns=cols))[0]
    return {"prediction": str(pred)}
