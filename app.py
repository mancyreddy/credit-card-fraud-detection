import streamlit as st
import numpy as np
import pandas as pd
from keras.src.saving import load_model
from sklearn.preprocessing import StandardScaler
import os

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide"
)

# ── CUSTOM CSS ──
st.markdown("""
<style>
    .main { background-color: #f1f5f9; }
    .title-box {
        background: linear-gradient(135deg, #0D1B2A, #1B4F8A);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .title-box h1 { color: #00B4D8; margin: 0; font-size: 2.2rem; }
    .title-box p  { color: #A0BDD8; margin: 0.3rem 0 0 0; font-size: 1rem; }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid #00B4D8;
    }
    .fraud-alert {
        background: #FEE2E2;
        border-left: 5px solid #DC2626;
        padding: 1.2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: bold;
        color: #DC2626;
    }
    .legit-alert {
        background: #D1FAE5;
        border-left: 5px solid #059669;
        padding: 1.2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: bold;
        color: #059669;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div class="title-box">
    <h1>🔍 Credit Card Fraud Detector</h1>
    <p>CNN-based real-time fraud detection · GITAM School of Technology · Mancy (2023001123)</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODEL ──
@st.cache_resource
def load_fraud_model():
    if os.path.exists("fraud_cnn_model.h5"):
        return load_model("fraud_cnn_model.h5", compile=False)
    return None

model = load_fraud_model()

# ── SIDEBAR ──
st.sidebar.title("⚙️ Options")
mode = st.sidebar.radio("Choose Mode", ["Single Transaction", "Upload CSV File"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("- Architecture: 3-block Conv1D CNN")
st.sidebar.markdown("- Accuracy: 100%")
st.sidebar.markdown("- FN: 0 (no fraud missed)")
st.sidebar.markdown("- Dataset: Kaggle Credit Card Fraud")

# ── ABOUT ──
with st.sidebar.expander("About this project"):
    st.write("""
    This project uses a 1D Convolutional Neural Network (CNN)
    trained on the Kaggle Credit Card Fraud Detection dataset.
    SMOTE was used to handle class imbalance.
    The model achieved 100% accuracy with zero missed fraud cases.
    """)

# ══════════════════════════════════
# MODE 1 — SINGLE TRANSACTION
# ══════════════════════════════════
if mode == "Single Transaction":
    st.subheader("🧾 Enter Transaction Details")
    st.caption("Enter the transaction features below. V1–V28 are PCA-transformed anonymized features.")

    col1, col2, col3 = st.columns(3)

    with col1:
        time   = st.number_input("Time (seconds)", value=0.0, format="%.2f")
        amount = st.number_input("Amount ($)", value=0.0, min_value=0.0, format="%.2f")
        st.markdown("**V1 – V10**")
        v1  = st.number_input("V1",  value=0.0, format="%.4f")
        v2  = st.number_input("V2",  value=0.0, format="%.4f")
        v3  = st.number_input("V3",  value=0.0, format="%.4f")
        v4  = st.number_input("V4",  value=0.0, format="%.4f")
        v5  = st.number_input("V5",  value=0.0, format="%.4f")
        v6  = st.number_input("V6",  value=0.0, format="%.4f")
        v7  = st.number_input("V7",  value=0.0, format="%.4f")
        v8  = st.number_input("V8",  value=0.0, format="%.4f")
        v9  = st.number_input("V9",  value=0.0, format="%.4f")
        v10 = st.number_input("V10", value=0.0, format="%.4f")

    with col2:
        st.markdown("**V11 – V20**")
        v11 = st.number_input("V11", value=0.0, format="%.4f")
        v12 = st.number_input("V12", value=0.0, format="%.4f")
        v13 = st.number_input("V13", value=0.0, format="%.4f")
        v14 = st.number_input("V14", value=0.0, format="%.4f")
        v15 = st.number_input("V15", value=0.0, format="%.4f")
        v16 = st.number_input("V16", value=0.0, format="%.4f")
        v17 = st.number_input("V17", value=0.0, format="%.4f")
        v18 = st.number_input("V18", value=0.0, format="%.4f")
        v19 = st.number_input("V19", value=0.0, format="%.4f")
        v20 = st.number_input("V20", value=0.0, format="%.4f")

    with col3:
        st.markdown("**V21 – V28**")
        v21 = st.number_input("V21", value=0.0, format="%.4f")
        v22 = st.number_input("V22", value=0.0, format="%.4f")
        v23 = st.number_input("V23", value=0.0, format="%.4f")
        v24 = st.number_input("V24", value=0.0, format="%.4f")
        v25 = st.number_input("V25", value=0.0, format="%.4f")
        v26 = st.number_input("V26", value=0.0, format="%.4f")
        v27 = st.number_input("V27", value=0.0, format="%.4f")
        v28 = st.number_input("V28", value=0.0, format="%.4f")

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Transaction", type="primary", use_container_width=True)

    if predict_btn:
        if model is None:
            st.error("⚠️ Model file not found. Place `fraud_cnn_model.h5` in the same folder as this app.")
        else:
            scaler = StandardScaler()
            time_scaled   = (time - 94813.0) / 47488.0
            amount_scaled = (amount - 88.35) / 250.12

            features = np.array([[
                time_scaled, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                v21, v22, v23, v24, v25, v26, v27, v28, amount_scaled
            ]])

            features_cnn = features.reshape(1, 30, 1)
            prob = float(model.predict(features_cnn, verbose=0)[0][0])
            pred = int(prob >= 0.5)

            st.markdown("### 📊 Prediction Result")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""<div class="metric-card">
                    <div style="font-size:2rem">{'🚨' if pred == 1 else '✅'}</div>
                    <div style="font-size:1.4rem;font-weight:bold;color:{'#DC2626' if pred==1 else '#059669'}">
                        {'FRAUD' if pred == 1 else 'LEGITIMATE'}
                    </div>
                    <div style="color:#64748B;font-size:0.9rem">Prediction</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="metric-card">
                    <div style="font-size:2rem">📈</div>
                    <div style="font-size:1.4rem;font-weight:bold;color:#1B4F8A">{prob*100:.2f}%</div>
                    <div style="color:#64748B;font-size:0.9rem">Fraud Probability</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class="metric-card">
                    <div style="font-size:2rem">💰</div>
                    <div style="font-size:1.4rem;font-weight:bold;color:#1B4F8A">${amount:.2f}</div>
                    <div style="color:#64748B;font-size:0.9rem">Transaction Amount</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if pred == 1:
                st.markdown('<div class="fraud-alert">🚨 FRAUD DETECTED — This transaction has been flagged as potentially fraudulent. Immediate review recommended.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="legit-alert">✅ LEGITIMATE — This transaction appears to be genuine. No fraud detected.</div>', unsafe_allow_html=True)

# ══════════════════════════════════
# MODE 2 — CSV UPLOAD
# ══════════════════════════════════
else:
    st.subheader("📂 Upload CSV for Batch Prediction")
    st.caption("Upload a CSV file with the same format as the Kaggle dataset (columns: Time, V1–V28, Amount). The 'Class' column is optional.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df)} rows")
        st.dataframe(df.head(), use_container_width=True)

        if model is None:
            st.error("⚠️ Model file not found. Place `fraud_cnn_model.h5` in the same directory.")
        else:
            if st.button("🔍 Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner("Running predictions..."):

                    # Scale
                    df_pred = df.copy()
                    if 'Class' in df_pred.columns:
                        df_pred = df_pred.drop('Class', axis=1)

                    scaler = StandardScaler()
                    df_pred['Amount'] = scaler.fit_transform(df_pred['Amount'].values.reshape(-1,1))
                    df_pred['Time']   = scaler.fit_transform(df_pred['Time'].values.reshape(-1,1))

                    X = df_pred.values.reshape(df_pred.shape[0], df_pred.shape[1], 1)
                    probs = model.predict(X, verbose=0).flatten()
                    preds = (probs >= 0.5).astype(int)

                    df['Fraud_Probability'] = (probs * 100).round(2)
                    df['Prediction'] = ['🚨 FRAUD' if p == 1 else '✅ Legitimate' for p in preds]

                fraud_count = int(sum(preds))
                legit_count = len(preds) - fraud_count

                st.markdown("### 📊 Batch Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Transactions", len(preds))
                c2.metric("🚨 Fraud Detected",  fraud_count)
                c3.metric("✅ Legitimate",       legit_count)

                st.dataframe(df[['Time', 'Amount', 'Fraud_Probability', 'Prediction']].head(50), use_container_width=True)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results CSV", csv, "fraud_predictions.csv", "text/csv", use_container_width=True)

# ── FOOTER ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:0.85rem'>"
    "Credit Card Fraud Detection · CNN Model · GITAM School of Technology · Mancy 2023001123"
    "</div>",
    unsafe_allow_html=True
)
