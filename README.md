
# 🐾 AI Animal Classifier (Cat vs Dog) — Streamlit App

This is a simple web application built using **Streamlit** that classifies uploaded images as either a **cat** or a **dog**, powered by a pre-trained deep learning model (Keras/TensorFlow).




## 🚀 Features

- Upload a `.jpg`, `.jpeg`, or `.png` image
- Model will classify it as **cat** or **dog**
- Uses a pre-trained `.h5` model hosted on **Google Drive**
- Automatically downloads the model if not found locally



## 🧠 Model Info

- Trained on a cat/dog dataset
- Model is a binary classifier (output > 0.5 = Dog, else Cat)
- Not stored in the GitHub repo (downloaded on-demand using `gdown`)



## 🔧 How to Run Locally

1. **Clone the repo**:
```bash
git clone https://github.com/RimpaDas11/AI_Animal_Classifier.git
cd AI_Animal_Classifier
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**:
```bash
streamlit run app.py
```



## 📦 Model Download

- The `.h5` model file (`cat_dog_classifier1.h5`) is automatically downloaded from Google Drive the first time the app runs, using `gdown`.
- If you prefer to download it manually, use this link:

🔗 **Model Download**:  
https://drive.google.com/file/d/1IPtus1oq835st3RJmZbhqkujbXJz3Sot/view?usp=sharing

After downloading, place the file in the project root directory.





## 👩‍💻 Author

**Rimpa Das**  
GitHub: [@RimpaDas11](https://github.com/RimpaDas11)



## 📌 Notes

- Works with `.jpg`, `.jpeg`, and `.png` images
- Requires Python 3.7+
