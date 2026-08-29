import json
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "brain_tumor_model.keras")
image_path = sys.argv[1]
model = tf.keras.models.load_model(model_path)
image = Image.open(image_path).convert("RGB").resize((224, 224))
raw = np.asarray(image, dtype=np.float32)
variants = {"raw_0_255": raw, "normalized_0_1": raw / 255.0}
output = {}
for name, array in variants.items():
    values = model.predict(array[None, ...], verbose=0)[0]
    output[name] = {"prediction": CLASS_NAMES[int(np.argmax(values))], "probabilities": {label: round(float(values[index]) * 100, 2) for index, label in enumerate(CLASS_NAMES)}}
print(json.dumps(output, indent=2))
