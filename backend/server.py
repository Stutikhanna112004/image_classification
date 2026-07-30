import io
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMG_SIZE = (160, 160)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cat_dogs_model.keras")

app = Flask(__name__)

# Restrict this to your actual Vercel URL once you have it, e.g.:
# CORS(app, origins=["https://pet-scanner.vercel.app"])
CORS(app)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

def predict_image(pil_image: Image.Image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    raw = float(model.predict(img_array, verbose=0)[0][0])

    if raw > 0.5:
        label, confidence = "dog", raw
    else:
        label, confidence = "cat", 1 - raw

    return label, confidence, raw

@app.route("/")
def health():
    # simple health check so you can confirm the API is alive
    return jsonify({"status": "ok", "service": "pet-scanner-api"})

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