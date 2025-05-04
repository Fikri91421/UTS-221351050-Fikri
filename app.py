import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import os

st.set_page_config(page_title="Prediksi Hepatitis C", layout="centered")

# -------------------------------
# Validasi file model & scaler
# -------------------------------
if not os.path.exists("scaler.pkl"):
    st.error("❌ File scaler.pkl tidak ditemukan!")
    st.stop()

if not os.path.exists("hepatitis_model.tflite"):
    st.error("❌ File hepatitis_model.tflite tidak ditemukan!")
    st.stop()

# -------------------------------
# Load model & scaler
# -------------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

interpreter = tf.lite.Interpreter(model_path="hepatitis_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# Mapping Kelas
# -------------------------------
inv_category_map = {
    0: 'Blood Donor',
    1: 'Suspect Blood Donor',
    2: 'Hepatitis',
    3: 'Fibrosis',
    4: 'Cirrhosis'
}

# -------------------------------
# UI Streamlit
# -------------------------------
st.title("🧬 Prediksi Kategori Pasien Hepatitis C")
st.markdown("Masukkan hasil tes darah pasien untuk memprediksi **kategori penyakit hati** berdasarkan model Machine Learning.")

# -------------------------------
# Input dari user
# -------------------------------
age = st.number_input("Umur (Age)", min_value=0, value=32)
sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])
alb = st.number_input("ALB (Albumin)", value=38.5)
alp = st.number_input("ALP (Alkaline Phosphatase)", value=70.0)
alt = st.number_input("ALT (Alanine Transaminase)", value=25.0)
ast = st.number_input("AST (Aspartate Transaminase)", value=30.0)
bil = st.number_input("BIL (Bilirubin)", value=5.0)
che = st.number_input("CHE (Cholinesterase)", value=8.0)
chol = st.number_input("CHOL (Cholesterol)", value=4.5)
crea = st.number_input("CREA (Creatinine)", value=80.0)
ggt = st.number_input("GGT (Gamma Glutamyl)", value=25.0)
prot = st.number_input("PROT (Protein)", value=70.0)

# -------------------------------
# Prediksi
# -------------------------------
if st.button("🔍 Prediksi Kategori"):
    try:
        # Encode dan scale input
        sex_encoded = 0 if sex == "Male" else 1
        input_data = np.array([[age, sex_encoded, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]])
        input_scaled = scaler.transform(input_data).astype(np.float32)

        # Masukkan ke model
        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Prediksi kelas
        predicted_class = int(np.argmax(output))
        result = inv_category_map.get(predicted_class, "Unknown")

        st.success(f"✅ Prediksi: **{result}**")
        st.write(f"Probabilitas kelas: {output[0]}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")