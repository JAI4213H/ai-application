"""Minimal Flask API for the MNIST digit recognizer."""

import base64
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model


app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("digit_model.keras")
ACCURACY_PATH = Path(__file__).with_name("model_metrics.json")
accuracy = None
if ACCURACY_PATH.exists():
    try:
        accuracy = json.loads(ACCURACY_PATH.read_text(encoding="utf-8")).get("test_accuracy")
    except (OSError, ValueError, TypeError):
        accuracy = None
try:
    model = load_model(MODEL_PATH) if zipfile.is_zipfile(MODEL_PATH) else None
except (OSError, ValueError):
    model = None


@app.get("/")
def home():
    return render_template("index.html", accuracy=accuracy)


@app.post("/predict")
def predict():
    if model is None:
        return jsonify(error="Model not found. Run: python model.py"), 503

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")
        image = image.resize((28, 28))
        pixels = np.asarray(image, dtype="float32") / 255.0
        probabilities = model.predict(pixels[None, ..., None], verbose=0)[0]
    except (ValueError, OSError, base64.binascii.Error):
        return jsonify(error="Please draw a valid digit."), 400

    digit = int(np.argmax(probabilities))
    return jsonify(
        digit=digit,
        confidence=round(float(probabilities[digit]) * 100, 1),
        probabilities=[round(float(value) * 100, 1) for value in probabilities],
    )


if __name__ == "__main__":
    app.run(debug=True)
