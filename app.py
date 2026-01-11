from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Load model
model = load_model(os.path.join(BASE_DIR, "crop_classifier_light.h5"))

CLASSES = [
    "Cherry", "Coffee-plant", "Cucumber", "Fox_nut(Makhana)", "Lemon",
    "Olive-tree", "Pearl_millet(bajra)", "Tobacco-plant", "almond", "banana",
    "cardamom", "chilli", "clove", "coconut", "cotton", "gram", "jowar",
    "jute", "maize", "mustard-oil", "papaya", "pineapple", "rice",
    "soyabean", "sugarcane", "sunflower", "tea", "tomato",
    "vigna-radiati(Mung)", "wheat"
]

# Uploads
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_name = CLASSES[class_idx]
    return render_template(
        "result.html",
        filename=file.filename,
        prediction=class_name
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
