from fastapi.testclient import TestClient
import pandas as pd
from meddx.serve.api import app

def test_predict_endpoint():
    client = TestClient(app)
    val = pd.read_csv("data/processed/val.csv")
    payload = val.drop(columns=["target"]).iloc[0].to_dict()
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    prob = r.json()["prob"]
    assert 0.0 <= prob <= 1.0
