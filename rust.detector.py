import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO

# Page config
st.set_page_config(page_title="Iron Rust Detector", page_icon="🔩", layout="centered")

# Custom CSS for dark Maybot-style theme
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #0f0f23, #1a1a2e); color: white; }
    h1 { background: linear-gradient(90deg, #00ffcc, #ff00cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .footer { text-align: center; margin-top: 50px; color: #00ffcc; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Header with your logo and title
col1, col2 = st.columns([1, 4])
with col1:
    try:
        logo = Image.open("illuminat.image.png")  # Put your logo in the same folder
        st.image(logo, width=100)
    except:
        st.image("https://via.placeholder.com/100", width=100)  # fallback

with col2:
    st.title("Iron Rust Detector")

st.markdown("Upload a photo of iron to check if it is **rusted** or **not rusted**.")

# Load the model (cached so it loads only once)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("rust_model.h5")
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (Teachable Machine usually uses 224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict
    prediction = model.predict(data)
    
    # Assuming Class 0 = Not Rusted, Class 1 = Rusted (check your classes order)
    class_names = ["Not Rusted", "Rusted"]
    confidence = prediction[0] * 100

    st.subheader("Prediction Result:")
    for i, class_name in enumerate(class_names):
        st.markdown(f"**{class_name}**: {confidence[i]:.1f}%")

    # Highlight the highest confidence
    max_idx = np.argmax(confidence)
    if max_idx == 1:
        st.error(f"🔴 **Rusted** with {confidence[1]:.1f}% confidence")
    else:
        st.success(f"🟢 **Not Rusted** with {confidence[0]:.1f}% confidence")

# Footer
st.markdown('<div class="footer">Made by Mayon Oberoi • Illuminat</div>', unsafe_allow_html=True)
