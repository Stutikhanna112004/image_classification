import io
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cat_dogs_model.tflite")
IMG_SIZE = (160, 160)

app = Flask(__name__)
CORS(app)

print("Loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded.")

def predict_image(pil_image: Image.Image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)  # no /255.0 — model's baked-in preprocess_input handles this
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    raw = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

    if raw > 0.5:
        label, confidence = "dog", raw
    else:
        label, confidence = "cat", 1 - raw

    return label, confidence, raw

@app.route("/")
def health():
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