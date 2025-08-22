import os
import gdown
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from keras.saving import legacy_serialization   # 👈 important fix

# Paths
MODEL_PATH = "cat_dog_classifier1.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1IPtus1oq835st3RJmZbhqkujbXJz3Sot"

# ---------------------------
# Download model if missing
# ---------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model not found locally. Downloading..."):
            try:
                gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                st.stop()

# ---------------------------
# Load model (patched for legacy format)
# ---------------------------
@st.cache_resource
def load_model():
    return keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False   # 👈 disables strict checks
    )

# ---------------------------
# Preprocess uploaded image
# ---------------------------
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# ---------------------------
# Main Streamlit app
# ---------------------------
def main():
    st.title("🐱🐶 Cat and Dog Classifier")
    st.write("Upload an image to classify it as a **Cat or Dog**.")

    # Download & load model
    download_model()
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        # Prediction
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            st.success("Prediction: 🐶 Dog")
        else:
            st.success("Prediction: 🐱 Cat")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()
