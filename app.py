from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "crop_classifier.h5")
IMG_SIZE = (128, 128)

CLASSES = [
    "Cherry",
    "Coffee-plant",
    "Cucumber",
    "Fox_nut(Makhana)",
    "Lemon",
    "Olive-tree",
    "Pearl_millet(bajra)",
    "Tobacco-plant",
    "almond",
    "banana",
    "cardamom",
    "chilli",
    "clove",
    "coconut",
    "cotton",
    "gram",
    "jowar",
    "jute",
    "maize",
    "mustard-oil",
    "papaya",
    "pineapple",
    "rice",
    "soyabean",
    "sugarcane",
    "sunflower",
    "tea",
    "tomato",
    "vigna-radiati(Mung)",
    "wheat",
]

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")


def prepare_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)
    return img_array


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            error = "Please upload an image file."
        else:
            try:
                x = prepare_image(file)
                preds = model.predict(x, verbose=0)
                class_idx = int(np.argmax(preds))
                prediction = CLASSES[class_idx]
            except Exception as e:
                error = f"Prediction error: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
