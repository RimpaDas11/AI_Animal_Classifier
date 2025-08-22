import tensorflow as tf
from tensorflow import keras
import numpy as np
import streamlit as st
from PIL import Image

# Load the saved model
model = keras.models.load_model("cat_dog_classifier1.keras")

# Streamlit UI
st.title("Cat and Dog Classification")
st.write("Upload an image to classify it as a cat or dog.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((150, 150))           # Resize to match model input
    img_array = np.array(img) / 255.0      # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)

    # Display the result
    if prediction[0][0] > 0.5:  # prediction[0][0] because model.predict returns [[prob]]
        st.write("Prediction: 🐶 Dog")
    else:
        st.write("Prediction: 🐈 Cat")
