import streamlit as st

st.set_page_config(page_title="MedDx-Pro", page_icon="🫀", layout="centered")
st.title("MedDx-Pro (Demo)")
st.caption("Educational demo • Not for clinical use")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
if st.button("Predict"):
    st.success(f"Dummy risk probability: 0.42 (age={age})")
