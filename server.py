"""
Flask backend for the Pet Scanner UI.

Serves the frontend (templates/index.html + static/) and exposes:
    POST /predict   -> accepts an image file, returns {label, confidence, raw}

Run with:
    python server.py
Then open http://127.0.0.1:5000
"""

import io
import os

from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

IMG_SIZE = (160, 160)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cat_dogs_model.keras")
CLASS_NAMES = ["cat", "dog"]  # index 0 -> cat, index 1 -> dog (alphabetical folder order)

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")


def predict_image(pil_image: Image.Image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    raw = float(model.predict(img_array, verbose=0)[0][0])

    if raw > 0.5:
        label = CLASS_NAMES[1]
        confidence = raw
    else:
        label = CLASS_NAMES[0]
        confidence = 1 - raw

    return label, confidence, raw


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    try:
        pil_image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Could not read image"}), 400

    label, confidence, raw = predict_image(pil_image)

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "raw": round(raw, 4),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)