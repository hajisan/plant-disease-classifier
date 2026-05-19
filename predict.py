import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import numpy as np
from tensorflow import keras
from keras.applications.resnet50 import preprocess_input
from PIL import Image

MODEL_PATH = "plant_disease_model.keras"
LABELS_PATH = "class_names.json"
IMAGE_SIZE = (100, 100)

def predict(image_path):
    model = keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH) as f:
        labels = json.load(f)

    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    img = preprocess_input(img)

    output = model.predict(img, verbose=0)
    class_index = int(np.argmax(output[0]))
    confidence = output[0][class_index] * 100

    print(f"Prediction : {labels[str(class_index)]}")
    print(f"Confidence : {confidence:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
