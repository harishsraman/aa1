
###############################
# 🩺 THYROID PREDICTOR — STREAMLIT
###############################

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import shap
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="🩺 Thyroid Prediction System", layout="wide")

# ===========================
# 🔥 LOAD MODEL FROM DRIVE
# ===========================
FILE_ID = "1OeXItzXaTEvZFgOg6FldYJjznF7p4D5Y"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    gdown.download(URL, "model.pkl", quiet=False)
    return joblib.load("model.pkl")

model = load_model()
st.success("🚀 Model loaded from Google Drive (cached)")


# ===========================
# LOAD FEATURE COLUMN STRUCTURE
# ===========================
with open("feature_columns.json") as f:
    feature_cols = json.load(f)


# ===========================
# UI LAYOUT
# =============================
st.title("🩺 Thyroid Cancer Risk Prediction + Explainability")

tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🧬 Medical History", "🧪 Lab Values"])
data = {}

# TAB 1 ------------------------
with tab1:
    colA, colB, colC = st.columns(3)
    with colA: data["Age"] = st.number_input("Age",1,120,30)
    with colB: data["Gender"] = st.radio("Gender",["Male","Female"],horizontal=True)
    with colC: data["Country"] = st.radio("Country",["India","China","Nigeria","Russia","Brazil"],horizontal=True)

# TAB 2 ------------------------
with tab2:
    colD,colE,colF = st.columns(3)
    yn=["No","Yes"]
    with colD:
        data["Family_History"] = st.radio("Family History",yn,horizontal=True)
        data["Smoking"] = st.radio("Smoking",yn,horizontal=True)
    with colE:
        data["Obesity"] = st.radio("Obesity",yn,horizontal=True)
        data["Diabetes"] = st.radio("Diabetes",yn,horizontal=True)
    with colF:
        data["Radiation_Exposure"] = st.radio("Radiation Exposure",yn,horizontal=True)
        data["Iodine_Deficiency"] = st.radio("Iodine Deficiency",yn,horizontal=True)

# TAB 3 ------------------------
with tab3:
    colX,colY,colZ,colW=st.columns(4)
    with colX: data["TSH_Level"]=st.number_input("TSH Level",step=0.01)
    with colY: data["T3_Level"]=st.number_input("T3 Level",step=0.01)
    with colZ: data["T4_Level"]=st.number_input("T4 Level",step=0.01)
    with colW: data["Nodule_Size"]=st.number_input("Nodule Size",step=0.01)


# ===========================
# PROCESS + PREDICT
# ===========================
def convert(df):
    df=pd.get_dummies(df)
    for c in feature_cols:
        if c not in df.columns: df[c]=0
    return df[feature_cols]

X = convert(pd.DataFrame([data]))

if st.button("🔍 Predict"):
    pred = model.predict(X)[0]
    st.success(f"🎯 Model predicts: **{pred}**")

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        st.subheader("📊 Contribution of each factor")
        shap.summary_plot(shap_vals, X, plot_type="bar")
        st.pyplot()
    except:
        st.warning("SHAP could not run due to model size")

st.caption("Built & engineered by Group 5 — Explainable AI Powered")
