import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ===============================
# Load artifacts
# ===============================
MODEL_PATH = "Model_Final.keras"
SCALER_PATH = "scaler_random.joblib"
THRESHOLD_PATH = "threshold.txt"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read())

# ===============================
# UI
# ===============================
st.set_page_config(page_title="LSTM NDVI Sawah", layout="centered")
st.title("🌾 Klasifikasi Lahan Sawah Berbasis NDVI")
st.write(
    "Aplikasi ini menggunakan **model LSTM** untuk mengklasifikasikan "
    "**Sawah** dan **Non-Sawah** berdasarkan data **NDVI multitemporal**."
)

# ===============================
# Upload data
# ===============================
uploaded_file = st.file_uploader(
    "Upload data NDVI (CSV, 1 baris = 1 segmen)", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data NDVI")
    st.dataframe(df)

    try:
        # pastikan hanya numeric
        X = df.values.astype(float)

        # scaling
        X_scaled = scaler.transform(X)

        # reshape ke LSTM: (samples, timesteps, features)
        X_lstm = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        if st.button("🔍 Prediksi"):
            preds = model.predict(X_lstm)

            st.subheader("📊 Hasil Prediksi")

            results = []
            for i, p in enumerate(preds):
                prob = float(p[0])
                label = "Sawah 🌾" if prob >= threshold else "Non-Sawah 🏙️"
                results.append(
                    {"Segmen": i + 1, "Probabilitas": round(prob, 3), "Kelas": label}
                )

            result_df = pd.DataFrame(results)
            st.dataframe(result_df)

            st.success("Prediksi selesai.")

    except Exception as e:
        st.error("Terjadi kesalahan pada format data.")
        st.exception(e)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption(
    "Deployment model LSTM NDVI • "
    "Digunakan untuk keperluan akademik dan penelitian"
)
