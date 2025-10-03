from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from joblib import load
from .schemas import PatientInput, PredictOut

app = FastAPI(title="MedDx-Pro API")

MODEL_CANDIDATES = [
    Path("artifacts/model_adv.joblib"),
    Path("artifacts/model.joblib"),
]

def _load_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return load(p)
    raise FileNotFoundError("no model artifact found")

_model = _load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut)
def predict(x: PatientInput):
    row = pd.DataFrame([x.model_dump()])
    if hasattr(_model, "predict_proba"):
        prob = float(_model.predict_proba(row)[:, 1][0])
    else:
        prob = float(_model.predict(row)[0])
    pred = int(prob >= 0.5)
    return PredictOut(prob=prob, pred=pred)
