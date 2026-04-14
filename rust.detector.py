import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import gdown
import os

st.set_page_config(page_title="Iron Rust Detector", page_icon="🔩", layout="centered")

# Dark theme
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
        pass

with col2:
    st.title("Iron Rust Detector")

st.write("Upload a photo of iron to detect whether it is rusted or not")

# Load model from Google Drive
@st.cache_resource
def load_model():
    file_id = "https://drive.google.com/file/d/1O4thLhDJD7sIYr1-3b_yc_OISMn1PrdF/view?usp=drive_link"   # ← REPLACE WITH rust_model.h5 FILE ID
    url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "rust_model.h5"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... (first time only)"):
            gdown.download(url, model_path, quiet=False)
    
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

# Load class names from labels.txt
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except:
    class_names = ["Oxidized/Rusted iron", "Unoxidized iron"]

# Upload image
uploaded_file = st.file_uploader("Choose an image of iron...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess (standard for Teachable Machine)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    # Predict
    with st.spinner("Analyzing the image..."):
        prediction = model.predict(data)[0]

    st.subheader("Prediction Result:")
    for i, name in enumerate(class_names):
        conf = float(prediction[i] * 100)
        st.progress(conf / 100, text=f"{name}: {conf:.1f}%")

    # Highlight the main result
    max_idx = int(np.argmax(prediction))
    result_name = class_names[max_idx]
    confidence = prediction[max_idx] * 100

    if "rusted" in result_name.lower() or "oxidized" in result_name.lower():
        st.error(f"🔴 **{result_name}** — Confidence: {confidence:.1f}%")
    else:
        st.success(f"🟢 **{result_name}** — Confidence: {confidence:.1f}%")

st.markdown('<div class="footer">Made by Mayon Oberoi • Illuminat</div>', unsafe_allow_html=True)
