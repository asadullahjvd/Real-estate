import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("real_estate.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()

# Feature order scaler expects (all 16 including MEDV placeholder)
SCALER_COLS = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
               'DIS','RAD','TAX','PTRATIO','B','LSTAT',
               'MEDV','TAXRM','QUALITY_INDEX']

MEDV_IDX = SCALER_COLS.index('MEDV')

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏠 Boston House Price Predictor")
st.write("Fill in the details below and click **Predict Price**.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Neighbourhood")
    crim    = st.number_input("Crime Rate (CRIM)",               min_value=0.0,   max_value=100.0, value=0.5,   step=0.1)
    zn      = st.number_input("Residential Zone % (ZN)",         min_value=0.0,   max_value=100.0, value=10.0,  step=1.0)
    indus   = st.number_input("Industrial Area % (INDUS)",       min_value=0.0,   max_value=100.0, value=10.0,  step=0.5)
    chas    = st.selectbox(   "Near Charles River? (CHAS)",      options=[0, 1])
    nox     = st.number_input("Nitric Oxide Conc. (NOX)",        min_value=0.0,   max_value=1.0,   value=0.5,   step=0.01)
    b       = st.number_input("B value",                         min_value=0.0,   max_value=400.0, value=390.0, step=1.0)
    rad     = st.number_input("Highway Access Index (RAD)",      min_value=1.0,   max_value=25.0,  value=5.0,   step=1.0)

with col2:
    st.subheader("🏡 Property")
    rm      = st.number_input("Avg Rooms per Dwelling (RM)",     min_value=1.0,   max_value=10.0,  value=6.0,   step=0.1)
    age     = st.number_input("Old Units % (AGE)",               min_value=0.0,   max_value=100.0, value=60.0,  step=1.0)
    dis     = st.number_input("Distance to Employment (DIS)",    min_value=0.0,   max_value=15.0,  value=4.0,   step=0.1)
    tax     = st.number_input("Property Tax Rate (TAX)",         min_value=100.0, max_value=800.0, value=300.0, step=1.0)
    ptratio = st.number_input("Pupil-Teacher Ratio (PTRATIO)",   min_value=10.0,  max_value=25.0,  value=15.0,  step=0.1)
    lstat   = st.number_input("Lower Status Population % (LSTAT)", min_value=0.0, max_value=40.0,  value=10.0,  step=0.1)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Price", use_container_width=True, type="primary"):

    # Engineered features (same as notebook)
    taxrm         = tax / rm if rm != 0 else 0
    quality_index = (rm / lstat * ptratio) if lstat != 0 else 0

    # Build DataFrame with all 16 cols (MEDV=0 as placeholder)
    input_df = pd.DataFrame([[
        crim, zn, indus, chas, nox, rm, age,
        dis, rad, tax, ptratio, b, lstat,
        0.0, taxrm, quality_index
    ]], columns=SCALER_COLS)

    # Scale
    scaled       = scaler.transform(input_df)
    scaled_df    = pd.DataFrame(scaled, columns=SCALER_COLS)

    # Drop MEDV before predicting
    X            = scaled_df.drop(columns=['MEDV'])
    pred_scaled  = model.predict(X)[0]

    # Inverse transform MEDV:  scaled → log space → actual
    pred_log     = pred_scaled * scaler.scale_[MEDV_IDX] + scaler.mean_[MEDV_IDX]
    pred_medv    = np.expm1(pred_log)      # reverse log1p applied during outlier handling
    price_usd    = pred_medv * 1000

    st.success(f"### 💰 Predicted Price: ${price_usd:,.0f}")

    st.divider()
    st.subheader("📊 Input Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Rooms",        f"{rm}")
    c2.metric("Lower Status %",   f"{lstat}%")
    c3.metric("Crime Rate",       f"{crim}")
    c1.metric("Tax Rate",         f"${tax}")
    c2.metric("Pupil-Teacher",    f"{ptratio}")
    c3.metric("TAXRM (engineered)", f"{taxrm:.2f}")

    st.caption("*Dataset from 1978 Boston. MEDV is median home value in $1,000s.*")
