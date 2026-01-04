from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("crop_classifier.h5")


# Automatically obtaining classes
data_dir = "data/Agricultural-crops"
CLASS_NAMES = sorted(os.listdir(data_dir))
print("Classes detected:", CLASS_NAMES)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"
    file = request.files["file"]
    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_name = CLASS_NAMES[class_idx]

    return render_template("result.html", filename=file.filename, prediction=class_name)


if __name__ == "__main__":
    app.run(debug=True)
