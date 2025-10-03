
import os
import json
import requests
import streamlit as st
from pathlib import Path
from joblib import load
import pandas as pd

API_URL = os.environ.get("MEDDX_API", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="MedDx-Pro", page_icon="🩺", layout="centered")
st.title("MedDx-Pro")
st.caption("Demo app • Not for clinical use")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("age", 1, 120, 57)
    sex = st.selectbox("sex", [0, 1], index=1)
    cp = st.selectbox("cp", [0, 1, 2, 3], index=0)
    trestbps = st.number_input("trestbps", 60, 260, 130)
    chol = st.number_input("chol", 100, 700, 245)
    fbs = st.selectbox("fbs", [0, 1], index=0)
with col2:
    restecg = st.selectbox("restecg", [0, 1, 2], index=1)
    thalach = st.number_input("thalach", 50, 250, 150)
    exang = st.selectbox("exang", [0, 1], index=0)
    oldpeak = st.number_input("oldpeak", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("slope", [0, 1, 2], index=2)
    ca = st.selectbox("ca", [0, 1, 2, 3, 4], index=0)
    thal = st.selectbox("thal", [0, 1, 2, 3], index=2)

payload = {
    "age": int(age),
    "sex": int(sex),
    "cp": int(cp),
    "trestbps": int(trestbps),
    "chol": int(chol),
    "fbs": int(fbs),
    "restecg": int(restecg),
    "thalach": int(thalach),
    "exang": int(exang),
    "oldpeak": float(oldpeak),
    "slope": int(slope),
    "ca": int(ca),
    "thal": int(thal),
}

st.subheader("Feature vector")
st.json(payload)

def try_api(data):
    try:
        r = requests.post(API_URL, json=data, timeout=5)
        if r.ok:
            j = r.json()
            return float(j["prob"]), int(j["pred"]), "api"
    except Exception:
        return None

def local_predict(data):
    model_path = Path("artifacts/model_adv.joblib")
    if not model_path.exists():
        model_path = Path("artifacts/model.joblib")
    model = load(model_path)
    df = pd.DataFrame([data])
    prob = float(model.predict_proba(df)[:,1][0]) if hasattr(model,"predict_proba") else float(model.predict(df)[0])
    pred = int(prob >= 0.5)
    return prob, pred, "local"

if st.button("Predict"):
    out = try_api(payload)
    if out is None:
        prob, pred, mode = local_predict(payload)
    else:
        prob, pred, mode = out
    st.success(f"Risk probability: {prob:.3f} • Pred: {pred} • Mode: {mode}")

st.divider()
st.subheader("Explanation")
img = None
for p in ["reports/shap_beeswarm.png", "reports/shap_bar.png", "reports/perm_importance.png"]:
    if Path(p).exists():
        img = p
        break
if img:
    st.image(img, caption=Path(img).name)
else:
    st.write("No explanation image available.")
