import os
import gdown
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf


# ---------------------------
# Paths and URLs
# ---------------------------
MODEL_PATH = "cat_dog_classifier1.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1IPtus1oq835st3RJmZbhqkujbXJz3Sot"

# ---------------------------
# Optional: Register custom layers/activations
# ---------------------------
# from your_custom_module import CustomLayer, custom_activation
# get_custom_objects().update({"CustomLayer": CustomLayer, "custom_activation": custom_activation})

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
# Load model safely with caching
# ---------------------------
@st.cache_resource  # Cache model in memory for faster reloads
def load_model_safe(model_path=MODEL_PATH):
    try:
        model = load_model(model_path, compile=False, safe_mode=False)
        st.success("Model loaded successfully!")
        return model
    except TypeError as e:
        st.error(f"TypeError while loading model: {e}")
    except OSError as e:
        st.error(f"OSError: Model file not found or corrupted. {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None

# ---------------------------
# Preprocess uploaded image
# ---------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((150, 150))  # Resize to model input
    image_array = np.array(image) / 255.0  # Normalize pixels
    if image_array.shape[-1] == 4:  # Remove alpha channel if present
        image_array = image_array[..., :3]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

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
        st.stop()  # Stop app if model failed to load

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)

            # Prediction
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)[0][0]

            # Display result
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
