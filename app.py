import os
import gdown
import tensorflow as tf
from tensorflow import keras
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = "cat_dog_classifier1.h5"

# ✅ New Google Drive link (converted to direct download format)
DRIVE_URL = "https://drive.google.com/uc?id=1IPtus1oq835st3RJmZbhqkujbXJz3Sot"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model not found locally. Downloading..."):
            try:
                gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                st.stop()

@st.cache_resource  # Caches model so it doesn't reload each time
def load_model():
    return keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def main():
    st.title("🐱🐶 Cat and Dog Classifier")
    st.write("Upload an image to classify it as a cat or a dog.")

    download_model()
    model = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            st.success("Prediction: 🐶 Dog")
        else:
            st.success("Prediction: 🐱 Cat")

# ✅ Fixing typo here
if __name__ == "__main__":
    main()
