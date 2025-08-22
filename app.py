import os
import gdown
import tensorflow as tf
from tensorflow import keras
import numpy as np
import streamlit as st
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "cat_dog_classifier1.h5"
# ✅ Use direct download link (converted from your Drive share link)
DRIVE_URL = "https://drive.google.com/uc?id=1Vn5zGrlIKIC7E9PB2OWsk9HphXh8tFa4"

# -----------------------------
# DOWNLOAD MODEL IF NEEDED
# -----------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model not found locally. Downloading..."):
            try:
                gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                st.stop()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    return keras.models.load_model(MODEL_PATH)

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match training input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# -----------------------------
# STREAMLIT APP
# -----------------------------
def main():
    st.title("🐱🐶 Cat and Dog Classifier")
    st.write("Upload an image to classify it as a **cat** or a **dog**.")

    # Ensure model is available
    download_model()
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")  # Ensure 3 channels
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)[0][0]

            if prediction > 0.5:
                st.success("Prediction: 🐶 **Dog**")
            else:
                st.success("Prediction: 🐱 **Cat**")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
