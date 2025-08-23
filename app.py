import os
import gdown
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ---------------------------
# Paths and URLs
# ---------------------------
MODEL_FOLDER = "cat_dog_classifier_saved_model"
# Replace with the proper Google Drive link to your SavedModel folder ZIP if needed
DRIVE_URL = "https://drive.google.com/uc?id=YOUR_SAVEDMODEL_FILE_ID"

# ---------------------------
# Download model if missing
# ---------------------------
def download_model():
    if not os.path.exists(MODEL_FOLDER):
        with st.spinner("Model not found locally. Downloading..."):
            try:
                # If it's a folder zipped in Drive, you may need to download and unzip
                zip_path = "saved_model.zip"
                gdown.download(DRIVE_URL, zip_path, quiet=False)
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(MODEL_FOLDER)
                os.remove(zip_path)
                st.success("✅ Model downloaded and extracted successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                st.stop()

# ---------------------------
# Load model safely with caching
# ---------------------------
@st.cache_resource  # Cache model in memory
def load_model_safe(model_path=MODEL_FOLDER):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ---------------------------
# Preprocess uploaded image
# ---------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((150, 150))  # Resize to match model input
    img_array = np.array(image) / 255.0  # Normalize pixels
    if img_array.shape[-1] == 4:  # Remove alpha channel if present
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# ---------------------------
# Main Streamlit app
# ---------------------------
def main():
    st.set_page_config(page_title="Cat & Dog Classifier", page_icon="🐱🐶")
    st.title("🐱🐶 Cat and Dog Classifier")
    st.write("Upload an image to classify it as a **Cat or Dog**.")

    # Ensure model is available
    download_model()
    model = load_model_safe()
    if model is None:
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)

            # Prediction
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)[0][0]

            if prediction > 0.5:
                st.success("Prediction: 🐶 Dog")
            else:
                st.success("Prediction: 🐱 Cat")
        except Exception as e:
            st.error(f"Error processing image: {e}")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()
