import streamlit as st
import numpy as np
from keras.src.saving import load_model
from PIL import Image
import gdown
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "cat_dog_classifier1.keras"
FILE_ID = "1KIucvTlFOZCDuknnRSXGzx2cDpDp1Nz-"  # Your new .keras model
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# -----------------------------
# CLEAN UP OLD/INVALID FILES
# -----------------------------
if os.path.exists("cat_dog_classifier1.h5"):
    os.remove("cat_dog_classifier1.h5")  # delete old .h5 file
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 100000:  # <100 KB likely broken
    os.remove(MODEL_PATH)

# -----------------------------
# DOWNLOAD MODEL IF NOT EXISTS
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
        
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model. Make sure the file is a valid .keras model. Error: {e}")
        return None

model = load_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    img = image.resize((150, 150))  # Match your training input size
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize
    return img_array

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("🐱🐶 Cat and Dog Classifier")
st.write("Upload an image to classify it as a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        if model is not None:
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)

            # Binary classifier (Cat=0, Dog=1)
            if prediction[0][0] > 0.5:
                st.success("🐶 This looks like a **Dog**!")
            else:
                st.success("🐱 This looks like a **Cat**!")
        else:
            st.error("Model not loaded. Please check the .keras file.")
