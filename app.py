import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import gdown
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "cat_dog_classifier1.keras"
FILE_ID = "1kGVQh-vwNCDnOAzcVYEOwQQwMsxrDsyU"  # Your real file ID
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# -----------------------------
# DOWNLOAD MODEL (if not exists)
# -----------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(URL, MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model. Make sure the file is a valid .keras model. Error: {e}")
        return None

model = load_model()

# -----------------------------
# PREPROCESS IMAGE
# -----------------------------
def preprocess_image(image):
    img = image.resize((224, 224))  # Match your training size
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0   # normalize
    return img_array

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("🐱🐶 Cat and Dog Classifier")
st.write("Upload an image to classify it as a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)

        # Binary classifier (Cat=0, Dog=1) assumption
        if prediction[0][0] > 0.5:
            st.success("🐶 This looks like a **Dog**!")
        else:
            st.success("🐱 This looks like a **Cat**!")
