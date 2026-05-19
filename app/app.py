import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import numpy as np
import gradio as gr
from tensorflow import keras
from keras.applications.resnet50 import preprocess_input
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "plant_disease_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "class_names.json")

model = keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    labels = json.load(f)

def predict(image):
    img = Image.fromarray(image).convert("RGB").resize((100, 100))
    img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    img = preprocess_input(img)
    output = model.predict(img, verbose=0)
    return {labels[str(i)]: float(output[0][i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="Plant Disease Classifier",
    description="Upload et plantebillede — modellen forudsiger sygdom på tværs af 35 klasser (7 planter).",
)

demo.launch()
