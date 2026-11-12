"""Train a small MNIST CNN and save it for the Flask app."""

from pathlib import Path
import struct
import json

import numpy as np
import tensorflow as tf


MODEL_PATH = Path(__file__).with_name("digit_model.keras")
ACCURACY_PATH = Path(__file__).with_name("model_metrics.json")
DATA_PATH = Path(__file__).with_name("archive")


def read_idx(images_file, labels_file):
    """Read the standard MNIST IDX image and label files."""
    with open(images_file, "rb") as file:
        _, count, rows, columns = struct.unpack(">IIII", file.read(16))
        images = np.frombuffer(file.read(), dtype="uint8").reshape(count, rows, columns)
    with open(labels_file, "rb") as file:
        _, count = struct.unpack(">II", file.read(8))
        labels = np.frombuffer(file.read(), dtype="uint8")[:count]
    return images, labels


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])


def main():
    x_train, y_train = read_idx(DATA_PATH / "train-images.idx3-ubyte", DATA_PATH / "train-labels.idx1-ubyte")
    x_test, y_test = read_idx(DATA_PATH / "t10k-images.idx3-ubyte", DATA_PATH / "t10k-labels.idx1-ubyte")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)
    ACCURACY_PATH.write_text(json.dumps({"test_accuracy": round(float(test_accuracy) * 100, 2)}), encoding="utf-8")
    print(f"Saved {MODEL_PATH.name} | test accuracy: {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
