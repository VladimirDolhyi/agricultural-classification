from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_classifier.tflite")
IMG_SIZE = (128, 128)

CLASSES = [
    "Cherry", "Coffee-plant", "Cucumber", "Fox_nut(Makhana)", "Lemon",
    "Olive-tree", "Pearl_millet(bajra)", "Tobacco-plant", "almond", "banana",
    "cardamom", "chilli", "clove", "coconut", "cotton", "gram", "jowar",
    "jute", "maize", "mustard-oil", "papaya", "pineapple", "rice",
    "soyabean", "sugarcane", "sunflower", "tea", "tomato",
    "vigna-radiati(Mung)", "wheat"
]

# ---------- Load TFLite ----------
try:
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    USE_TFLITE = True
    print("Using tflite-runtime")
except ImportError:
    USE_TFLITE = False
    print("tflite-runtime not available (local Windows)")


def prepare_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_image(x):
    if USE_TFLITE:
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])
        return preds
    else:
        raise RuntimeError("TFLite runtime not available")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    prediction = None
    error = None

    file = request.files.get("file")
    if not file or file.filename == "":
        error = "Please upload an image file."
    else:
        try:
            x = prepare_image(file)
            preds = predict_image(x)
            class_idx = int(np.argmax(preds))
            prediction = CLASSES[class_idx]
        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
