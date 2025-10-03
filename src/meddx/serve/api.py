from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from joblib import load
from .schemas import PatientInput, PredictOut
from typing import Dict
61343af (M7: fix pydantic model_dump and add best_model to val_metrics.json)
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from joblib import load
bdf73a2 (M5: FastAPI /predict + Streamlit UI + API test and Make targets)

app = FastAPI(title="MedDx-Pro API")

from .schemas import PatientInput, PredictOut

app = FastAPI(title="MedDx-Pro API")

61343af (M7: fix pydantic model_dump and add best_model to val_metrics.json)
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
61343af (M7: fix pydantic model_dump and add best_model to val_metrics.json)

def _load_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return load(p)
    raise FileNotFoundError("no model artifact found")

_model = load(MODEL_PATH)
_cols = [c for c in pd.read_csv(TRAIN_PATH, nrows=1).columns if c != "target"]
bdf73a2 (M5: FastAPI /predict + Streamlit UI + API test and Make targets)
_model = _load_model()
61343af (M7: fix pydantic model_dump and add best_model to val_metrics.json)

@app.get("/health")
def health():
    return {"status": "ok"}

61343af (M7: fix pydantic model_dump and add best_model to val_metrics.json)
@app.post("/predict", response_model=PredictOut)
def predict(x: PatientInput):
    row = pd.DataFrame([x.model_dump()])
    if hasattr(_model, "predict_proba"):
        prob = float(_model.predict_proba(row)[:, 1][0])
    else:
        prob = float(_model.predict(row)[0])
    pred = int(prob >= 0.5)
    return PredictOut(prob=prob, pred=pred)
@app.post("/predict")
def predict(payload: Dict[str, float]):
    x = {k: payload.get(k, 0.0) for k in _cols}
    df = pd.DataFrame([x], columns=_cols)
    p = _model.predict_proba(df)[:, 1] if hasattr(_model, "predict_proba") else _model.predict(df)
    return {"prob": float(p[0])}
bdf73a2 (M5: FastAPI /predict + Streamlit UI + API test and Make targets)
61343af (M7: fix pydantic model_dump and add best_model to val_metrics.json)
