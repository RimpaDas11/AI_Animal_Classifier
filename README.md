# 🐶🐱 Cat & Dog Image Classifier (AI)

This is a simple web application built using **Streamlit** that classifies uploaded images as either a **cat** or a **dog**, powered by a pre-trained deep learning model (Keras/TensorFlow).

---

## 🚀 Features

- Upload a `.jpg`, `.jpeg`, or `.png` image
- Model will classify it as **cat** or **dog**
- Uses a pre-trained `.h5` model hosted on **Google Drive**
- Automatically downloads the model if not found locally

---

## 🧠 Model Info

- Trained on a cat/dog dataset
- Model is a binary classifier (output > 0.5 = Dog, else Cat)
- Not stored in the GitHub repo (downloaded on-demand using `gdown`)

---

## 🔧 How to Run Locally

1. **Clone the repo**:

   ```bash
   git clone https://github.com/RimpaDas11/AI_Animal_Classifier.git
   cd AI_Animal_Classifier
2.Install dependencies:
pip install -r requirements.txt
3.Run the Streamlit app:
streamlit run app.py
📁Model Download
The .h5 model file will be downloaded automatically from Google Drive when the app runs, using gdown.
🧑‍💻 Author

Rimpa Das
GitHub: @RimpaDas11