###########################################################
# 🩺 THYROID CANCER PREDICTION SYSTEM  (CLINICAL AI EDITION)
# Includes Explainability + Flags + Risk Stratification
###########################################################

import streamlit as st
import pandas as pd
import numpy as np
import joblib, gdown, json, shap, matplotlib.pyplot as plt

st.set_page_config(page_title="🩺 Thyroid Cancer Predictor", layout="wide")

############################################################
# 🔥 MODEL LOADING FROM GOOGLE DRIVE (CACHED)
############################################################

FILE_ID = "1OeXItzXaTEvZFgOg6FldYJjznF7p4D5Y"    # your 851MB PKL
URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    gdown.download(URL, "model.pkl", quiet=False)
    return joblib.load("model.pkl")

model = load_model()
st.success("🚀 Model loaded from Google Drive (cached)")


############################################################
# 🔥 LOAD FEATURE COLUMNS
############################################################
with open("feature_columns.json") as f:
    feature_cols = json.load(f)


############################################################
# UI SECTIONS — 3 PAGE INPUT COLLECTION
############################################################
st.title("🧠 AI-Driven Thyroid Cancer Risk Assessor")

tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🧬 Medical History", "🧪 Lab Values"])
data = {}

# TAB 1 ----------------------------------
with tab1:
    st.subheader("👤 Demographic Indicators")
    col1,col2,col3 = st.columns(3)
    with col1: data["Age"] = st.number_input("Age",1,120,30)
    with col2: data["Gender"] = st.radio("Gender",["Male","Female"],horizontal=True)
    with col3: data["Country"] = st.radio("Country",["India","USA","UK","Japan","Brazil"],horizontal=True)

# TAB 2 ----------------------------------
with tab2:
    st.subheader("🧬 Family & Lifestyle Determinants")
    col4,col5,col6 = st.columns(3)
    yn=["No","Yes"]
    with col4:
        data["Family_History"]=st.radio("Family History of Thyroid Cancer",yn,horizontal=True)
        data["Smoking"]=st.radio("Smoking",yn,horizontal=True)
    with col5:
        data["Obesity"]=st.radio("Obesity",yn,horizontal=True)
        data["Diabetes"]=st.radio("Diabetes",yn,horizontal=True)
    with col6:
        data["Radiation_Exposure"]=st.radio("Radiation Exposure",yn,horizontal=True)
        data["Iodine_Deficiency"]=st.radio("Iodine Deficiency",yn,horizontal=True)

# TAB 3 ----------------------------------
with tab3:
    st.subheader("🧪 Clinical / Thyroid Function Parameters")
    colA,colB,colC,colD=st.columns(4)
    with colA: data["TSH_Level"]=st.number_input("TSH Level",step=0.01)
    with colB: data["T3_Level"]=st.number_input("T3 Level",step=0.01)
    with colC: data["T4_Level"]=st.number_input("T4 Level",step=0.01)
    with colD: data["Nodule_Size"]=st.number_input("Nodule Size (cm)",step=0.01)


############################################################
# DATA → ONE-HOT ENCODE → MODEL FORMAT
############################################################

def prepare(df):
    df = pd.get_dummies(df)
    for c in feature_cols:
        if c not in df.columns:
            df[c]=0
    return df[feature_cols]

X = prepare(pd.DataFrame([data]))


############################################################
# 🔥 CLINICAL RISK LOGIC
############################################################

def risk_badge(prob):
    if prob < 0.33: return "🟢 **LOW RISK – Non-malignant likelihood**"
    if prob < 0.66: return "🟡 **MODERATE RISK – Clinical monitoring advised**"
    return "🔴 **HIGH RISK – Suspicion of malignancy**"


def clinical_flags(d):
    alerts=[]
    if d["TSH_Level"]>4.5: alerts.append("⚠ High TSH → Possible Hypothyroidism")
    if d["T3_Level"]<0.8: alerts.append("⚠ Low T3 (suppressed metabolism)")
    if d["T4_Level"]<4.5: alerts.append("⚠ Low T4 (thyroid underfunction)")
    if d["Nodule_Size"]>2: alerts.append("🚨 Nodule > 2cm → FNAC recommended")
    return alerts


############################################################
# PREDICT BUTTON
############################################################

if st.button("🔍 Diagnose"):
    y_prob = model.predict_proba(X)[0][1] if hasattr(model,"predict_proba") else None
    pred = model.predict(X)[0]           # 0=Non Cancerous, 1=Cancerous

    st.subheader("🩺 Diagnosis Result")
    if pred==0:
        st.success(f"Result: **NON-CANCEROUS** (0)")
    else:
        st.error(f"Result: **CANCEROUS Suspicion** (1)")

    st.write("### 🔥 Risk Interpretation")
    st.write(risk_badge(y_prob))

    # ---------------- Clinical Flags Output ----------------
    flags = clinical_flags(data)
    if flags:
        st.warning("### 🚑 Clinical Alerts")
        for f in flags: st.write(f)
    else:
        st.success("No critical hormone/nodule flags detected ✔")


    ########################################################
    # SHAP — EXPLAIN WHAT CONTRIBUTED TO RISK
    ########################################################
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)

        st.write("### 🧠 Why this diagnosis was made:")

        # TEXT-BASED EXPLANATION (Top 5 features)
        vals = shap_vals[1][0] if isinstance(shap_vals,list) else shap_vals[0]
        imp = pd.Series(vals,index=feature_cols).sort_values(key=abs,ascending=False).head(5)

        for f,v in imp.items():
            contribution = "📈 Increases Cancer Risk" if v>0 else "📉 Protective Effect"
            st.write(f"- **{f}** → {contribution} (impact {v:.3f})")

        # Reduced-size SHAP Plot (fits screen)
        fig, ax = plt.subplots(figsize=(6,4))
        shap.summary_plot(shap_vals, X, plot_type="bar", max_display=8, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP explanation unavailable due to model size")
        st.text(str(e))


st.caption("Clinical-grade Thyroid AI • Built by Harish 🔥")
