import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import requests, os

# ==============================
# Model Setup
# ==============================
MODEL_PATH = "cat_dog_classifier1.keras"
DRIVE_URL = "https://drive.google.com/uc?id=1kGVQh-vwNCDnOAzcVYEOwQQwMsxrDsyU"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(DRIVE_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
        st.success("Model downloaded successfully!")

    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"TensorFlow loader failed: {e}")
        return keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

# ==============================
# Preprocessing Function
# ==============================
def preprocess_image(image):
    img = image.resize((150, 150))  # same size used during training
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # shape (1, 150, 150, 3)
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
