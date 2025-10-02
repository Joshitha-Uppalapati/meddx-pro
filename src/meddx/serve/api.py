from typing import Dict
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load

app = FastAPI(title="MedDx-Pro API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("artifacts/model.joblib")
TRAIN_PATH = Path("data/processed/train.csv")

_model = load(MODEL_PATH)
_cols = [c for c in pd.read_csv(TRAIN_PATH, nrows=1).columns if c != "target"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Dict[str, float]):
    x = {k: payload.get(k, 0.0) for k in _cols}
    df = pd.DataFrame([x], columns=_cols)
    p = _model.predict_proba(df)[:, 1] if hasattr(_model, "predict_proba") else _model.predict(df)
    return {"prob": float(p[0])}
