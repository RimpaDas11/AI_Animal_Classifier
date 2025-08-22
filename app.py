import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# ==============================
# Model Setup
# ==============================
MODEL_PATH = "cat_dog_classifier1.keras"
# Use the direct download link from Google Drive
DRIVE_URL = "https://drive.google.com/uc?id=1kGVQh-vwNCDnOAzcVYEOwQQwMsxrDsyU"

@st.cache_resource
def load_model():
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(DRIVE_URL)
            # Ensure the request succeeded
            if r.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(r.content)
                st.success("Model downloaded successfully!")
            else:
                st.error(f"Failed to download model: status code {r.status_code}")
                st.stop()

    # Load the model
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ==============================
# Image Preprocessing
# ==============================
def preprocess_image(image):
    img = image.resize((150, 150))  # resize to model input size
    img = np.array(img) / 255.0      # normalize
    img = np.expand_dims(img, axis=0)  # batch dimension
    return img

# ==============================
# Streamlit App
# ==============================
def main():
    st.title("🐱🐶 Cat and Dog Classifier")
    st.write("Upload an image to classify it as a cat or a dog.")

    model = load_model()

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
