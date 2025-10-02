from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MedDx-Pro API", version="0.1.0")

class PatientInput(BaseModel):
    age: int = 50  # placeholder; we'll expand later

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PatientInput):
    # dummy response for now
    return {"risk_probability": 0.42, "version": "0.1.0"}
