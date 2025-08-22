import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ==============================
# Model Setup
# ==============================
MODEL_PATH = "cat_dog_classifier1.keras"
# Direct download link from Google Drive
DRIVE_URL = "https://drive.google.com/uc?id=1kGVQh-vwNCDnOAzcVYEOwQQwMsxrDsyU"

@st.cache_resource  # caches the loaded model for faster reloads
def load_model():
    # Step 1: Download model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
    
    # Step 2: Load the model safely
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ==============================
# Image Preprocessing
# ==============================
def preprocess_image(image):
    image = image.resize((150, 150))       # resize to model input size
    image = np.array(image) / 255.0        # normalize pixel values
    image = np.expand_dims(image, axis=0)  # add batch dimension
    return image

# ==============================
# Streamlit App
# ==============================
def main():
    st.title("🐱🐶 Cat and Dog Classifier")
    st.write("Upload an image to classify it as a cat or a dog.")

    # Load model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)[0][0]

            if prediction > 0.5:
                st.success("It's a 🐶 Dog!")
            else:
                st.success("It's a 🐱 Cat!")

if __name__ == "__main__":
    main()
