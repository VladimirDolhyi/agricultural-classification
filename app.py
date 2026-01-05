from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


app = Flask(__name__)

# Limit upload size (2 MB)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = load_model(os.path.join(BASE_DIR, "crop_classifier.h5"))

# Data directory
# Automatically obtaining classes
data_dir = os.path.join(BASE_DIR, "data", "Agricultural-crops")
CLASS_NAMES = sorted(os.listdir(data_dir))
print("Classes detected:", CLASS_NAMES)

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

    prediction = model(img_array, training=False).numpy()
    class_idx = np.argmax(prediction)
    class_name = CLASS_NAMES[class_idx]

    return render_template("result.html", filename=file.filename, prediction=class_name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
