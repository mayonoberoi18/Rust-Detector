import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import gdown
import os

st.set_page_config(page_title="Iron Rust Detector", page_icon="🔩", layout="centered")

# Dark modern theme
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #0f0f23, #1a1a2e); color: white; }
    h1 { background: linear-gradient(90deg, #00ffcc, #ff00cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8em; }
    .footer { text-align: center; margin-top: 60px; color: #00ffcc; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    try:
        logo = Image.open("illuminat.image.png")
        st.image(logo, width=90)
    except:
        pass  # fallback if logo missing

with col2:
    st.title("Iron Rust Detector")

st.write("Upload a clear photo of iron to detect rust")

# Load model from Google Drive (change the file_id)
@st.cache_resource
def load_model():
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"   # ← Replace with your actual File ID
    url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "rust_model.h5"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model (first time only, may take 30-60 seconds)..."):
            gdown.download(url, model_path, quiet=False)
    
    model = tf.keras.models.load_model(model_path)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Upload section
uploaded_file = st.file_uploader("Choose an image of iron...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing (standard for Teachable Machine)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    # Prediction
    with st.spinner("Analyzing..."):
        prediction = model.predict(data)[0]
    
    class_names = ["Not Rusted", "Rusted"]   # Change order if your Class 0 is Rusted

    st.subheader("Result:")
    for i, name in enumerate(class_names):
        conf = prediction[i] * 100
        st.progress(conf / 100, text=f"{name}: {conf:.1f}%")

    # Highlight main result
    max_idx = np.argmax(prediction)
    if max_idx == 1:
        st.error(f"🔴 **Rusted** — Confidence: {prediction[1]*100:.1f}%")
    else:
        st.success(f"🟢 **Not Rusted** — Confidence: {prediction[0]*100:.1f}%")

# Footer
st.markdown('<div class="footer">Made by Mayon Oberoi • Illuminat</div>', unsafe_allow_html=True)
