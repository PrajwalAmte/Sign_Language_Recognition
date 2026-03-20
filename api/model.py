import os
import numpy as np
from keras.models import load_model


LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
MODEL_PATH = os.getenv("MODEL_PATH", "models/smnist.h5")


class SignPredictor:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        # Warm up
        self.model.predict(np.zeros((1, 28, 28, 1)), verbose=0)

    def predict(self, pixels: list[float]) -> tuple[str, float, int]:
        arr = np.array(pixels, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        probs = self.model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx]), idx
