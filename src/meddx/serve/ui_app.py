import json
import requests
import pandas as pd
import streamlit as st
<<<<<<< HEAD
def main():
    st.set_page_config(page_title="MedDx Pro", page_icon="🩺", layout="centered")
    st.title("MedDx Pro 🚀")
    st.write("Hello! Your Hugging Face Space is working 🎉")
=======

st.set_page_config(page_title="MedDx-Pro", layout="centered")
st.title("MedDx-Pro")
st.caption("Demo app • Not for clinical use")

API_URL = "http://127.0.0.1:8000/predict"
df = pd.read_csv("data/processed/val.csv")
X = df.drop(columns=["target"])
idx = st.number_input("Row index from validation set", min_value=0, max_value=len(X)-1, value=0, step=1)
row = X.iloc[int(idx)].to_dict()

st.subheader("Feature vector")
st.json(json.loads(pd.Series(row).to_json()))

if st.button("Predict"):
    r = requests.post(API_URL, json=row, timeout=10)
    if r.ok:
        st.success(f"Risk probability: {r.json()['prob']:.3f}")
    else:
        st.error(f"API error: {r.status_code}")
>>>>>>> bdf73a2 (M5: FastAPI /predict + Streamlit UI + API test and Make targets)
