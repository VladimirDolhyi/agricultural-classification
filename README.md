# 🌾 Agricultural Crops Image Classification Web App

This project is an image classification web application built using **Machine Learning** and **Flask**.  
The goal is to classify images of agricultural crops into one of **30 different crop classes** using a trained neural network model.

The project combines:
- Data analysis & preprocessing
- Training a CNN-based image classifier
- Deployment of the trained model via a Flask web interface

---

## 📌 Dataset

**Agricultural Crops Image Classification Dataset**  
Source: Kaggle  
🔗 https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification

The dataset contains images of agricultural crops organized into folders, where each folder name represents a class label (e.g. maize, wheat, rice, tomato, etc.).

Total number of classes: **30**

---

## 🧠 Model Training

The model was trained using **Google Colab** with GPU support.

### Key points:
- Image size: `128 x 128`
- Normalization: pixel values scaled to `[0, 1]`
- Model: CNN / Transfer Learning (e.g. MobileNetV2)
- Validation split: 20%
- Optimizer: Adam
- Loss function: Categorical Crossentropy

📎 Google Colab notebook (public access):  
🔗 **https://colab.research.google.com/drive/1doyXrzPbLtsiQ-7F2estMGVIxn2Y3kTa?usp=sharing**

After training, the model was saved as:
crop_classifier.h5
and downloaded for local inference in the Flask application.

---

## 🚀 Web Application (Flask)

The Flask web app allows users to:
1. Upload an image of a crop
2. Get a predicted crop class from the trained model

### Pages:
- **Home page** – image upload form
- **Result page** – prediction result with uploaded image preview

### Technologies used:
- Flask
- TensorFlow / Keras
- HTML (Jinja2 templates)

---

## 📂 Project Structure

```text
agricultural_classification/
│
├── app.py                 # Flask application
├── crop_classifier.h5     # Trained model
├── data/
│   └── Agricultural-crops/ # Dataset (folders = classes)
├── static/
│   └── uploads/           # Uploaded images
├── templates/
│   ├── index.html         # Upload page
│   └── result.html        # Prediction result page
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run the Project Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/VladimirDolhyi/agricultural-classification-.git
cd agricultural-classification-
```
### 2️⃣ Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run Flask app
```bash
python app.py
```
### #️⃣ Open in browser
```bash
http://127.0.0.1:5000/
```
## 🖼 Example

**Input image:**

![Input image](static/uploads/image2.jpeg)

**Prediction result:** chili
