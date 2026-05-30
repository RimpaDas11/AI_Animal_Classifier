# 🐾 AI Animal Classifier

<div align="center">

### Deep Learning-Based Cat vs Dog Image Classification System

Upload an image and let Artificial Intelligence determine whether it contains a Cat 🐱 or a Dog 🐶.

---

**Python • TensorFlow • Keras • Streamlit • Computer Vision • Deep Learning**

</div>

---

# 🚀 Overview

AI Animal Classifier is a Deep Learning-powered web application that identifies whether an uploaded image contains a cat or a dog.

The application utilizes a trained Convolutional Neural Network (CNN) model developed using TensorFlow and Keras. Users can upload an image through an intuitive Streamlit interface and instantly receive a prediction.

The project demonstrates practical implementation of Computer Vision, Deep Learning, Model Deployment, and Interactive Web Applications.

---

# 🌟 Project Highlights

✅ Deep Learning Image Classification

✅ Cat vs Dog Recognition

✅ Streamlit Web Interface

✅ TensorFlow & Keras Integration

✅ Automated Model Download

✅ User-Friendly Design

✅ Real-Time Prediction

---

# 🎯 Problem Statement

Image classification is one of the most important applications of Artificial Intelligence.

This project aims to automatically classify animal images into two categories:

* 🐱 Cat
* 🐶 Dog

using a trained Deep Learning model.

The goal is to demonstrate how machine learning models can be deployed as interactive web applications for real-world use.

---

# ✨ Features

## 🖼️ Image Upload

Supports:

* JPG
* JPEG
* PNG

file formats.

---

## 🧠 Deep Learning Prediction

Uses a trained CNN model to analyze image content.

---

## ⚡ Instant Results

Generates predictions within seconds.

---

## ☁️ Automatic Model Loading

The application automatically downloads the trained model when required.

---

## 🎨 Interactive User Interface

Built using Streamlit for a smooth user experience.

---

# 🏗️ System Architecture

```text
User Uploads Image
          │
          ▼
Image Preprocessing
(Resize & Normalize)
          │
          ▼
CNN Model
(TensorFlow/Keras)
          │
          ▼
Prediction Engine
          │
          ▼
Classification Result
          │
          ▼
Cat 🐱 or Dog 🐶
```

---

# 🧠 How It Works

### Step 1 — Upload Image

The user uploads an image through the Streamlit interface.

### Step 2 — Image Processing

The image is:

* Resized
* Normalized
* Converted into model-compatible format

---

### Step 3 — Model Inference

The trained CNN model processes the image and predicts the probability score.

---

### Step 4 — Classification

```text
Probability > 0.5  → Dog 🐶
Probability ≤ 0.5 → Cat 🐱
```

---

### Step 5 — Result Display

The final prediction is displayed on the screen.

---

# 🛠️ Technology Stack

| Category                 | Technology           |
| ------------------------ | -------------------- |
| Programming Language     | Python               |
| Deep Learning            | TensorFlow           |
| Neural Network Framework | Keras                |
| Web Framework            | Streamlit            |
| Image Processing         | PIL                  |
| Numerical Computing      | NumPy                |
| Model Distribution       | Google Drive + gdown |

---

# 📂 Project Structure

```text
AI_Animal_Classifier/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   └── cat_dog_classifier1.h5
│
├── assets/
│
└── images/
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/RimpaDas11/AI_Animal_Classifier.git
cd AI_Animal_Classifier
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will automatically open in your browser.

---

# 📦 Model Management

The trained model is not stored directly in the repository.

When the application starts:

1. Checks for the model file.
2. Downloads the model from Google Drive if necessary.
3. Loads the model automatically.

This approach keeps the repository lightweight and deployment-friendly.

---

# 📊 Model Information

### Model Type

Convolutional Neural Network (CNN)

### Classification Task

Binary Image Classification

### Classes

* Cat
* Dog

### Framework

TensorFlow / Keras

### Output

Probability-Based Prediction

---

# 📸 Demo

Include screenshots of:

* Home Page
* Image Upload Screen
* Prediction Result
* Cat Classification
* Dog Classification

Adding screenshots significantly improves GitHub presentation.

---

# 💡 Applications

### 🐾 Pet Identification

Automatic classification of pet images.

### 🎓 Educational Learning

Demonstrates Deep Learning and Computer Vision concepts.

### 🤖 AI-Powered Applications

Can serve as a foundation for advanced image recognition systems.

### 📱 Smart Mobile Applications

Can be integrated into mobile and web platforms.

---

# 📈 Skills Demonstrated

This project showcases:

* Deep Learning
* Computer Vision
* Convolutional Neural Networks
* TensorFlow
* Keras
* Streamlit
* Model Deployment
* Python Development

---

# 🔮 Future Enhancements

### 🐾 Multi-Class Animal Detection

Support additional animal species.

### 📊 Confidence Scores

Display prediction probabilities.

### 📷 Webcam Support

Real-time image classification.

### ☁️ Cloud Deployment

Deploy using:

* Streamlit Cloud
* Render
* Railway

### 📱 Mobile Optimization

Improve accessibility across devices.

---

# ⚠️ Disclaimer

This project is intended for educational and learning purposes.

Prediction accuracy depends on image quality, lighting conditions, and model performance.

Results should not be considered guaranteed classifications.

---

# 👩‍💻 Developer

## Rimpa Das

B.Tech Computer Science & Engineering
Brainware University

Passionate about Artificial Intelligence, Machine Learning, Computer Vision, and Full-Stack Development.

### Skills Demonstrated

* Deep Learning
* Machine Learning
* TensorFlow & Keras
* Python Programming
* Computer Vision
* Streamlit Development

### Related Projects

* AI-Based Animal Classifier
* Air Drawing using Hand Gesture Recognition
* Silent Communication – Gesture Read Using AI
* Creative Showcase

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork the repository

🚀 Share it with others

---

<div align="center">

# 🐾 AI Animal Classifier

### Teaching Machines to Recognize Cats and Dogs

**Deep Learning • Computer Vision • TensorFlow • Streamlit**

Built with ❤️ by Rimpa Das

</div>
