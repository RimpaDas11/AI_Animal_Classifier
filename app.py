import tensorflow as tf
from tensorflow import keras
import numpy as np
import streamlit as st
from PIL import Image


# Load the saved model
model = keras.models.load_model("cat_dog_classifier1.h5")


# Streamlit UI
st.title("Cat and Dog Classification")
st.write("Upload an image to classify it as a cat or dog.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((150, 150))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input shape

    # Make a prediction
    prediction = model.predict(image)

    # Display the result
    if prediction[0] > 0.5:
        st.write("Prediction: 🐶 Dog")
    else:
        st.write("Prediction: 🐈 Cat")
