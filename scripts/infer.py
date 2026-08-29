import base64
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (224, 224)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_gradcam(model, image, predicted_index):
    efficientnet = model.layers[0]
    resized = image.resize(IMG_SIZE)
    # Match app.py: the Keras EfficientNet artifact performs its own input rescaling, so it expects raw 0–255 pixels.
    array = np.asarray(resized, dtype=np.float32)
    tensor = np.expand_dims(array, axis=0)
    _ = efficientnet(tensor, training=False)

    grad_model = tf.keras.models.Model(
        inputs=efficientnet.input,
        outputs=[efficientnet.get_layer("top_conv").output, efficientnet.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, features = grad_model(tensor, training=False)
        head_output = features
        for layer in model.layers[1:]:
            head_output = layer(head_output)
        class_score = head_output[:, predicted_index]

    gradients = tape.gradient(class_score, conv_outputs)
    pooled = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap).numpy()
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (image.height, image.width)).numpy().squeeze()

    figure = plt.figure(figsize=(7, 6), frameon=False)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(image)
    axis.imshow(heatmap_resized, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axis.axis("off")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def main():
    if len(sys.argv) != 2:
        raise ValueError("Expected one image path")
    image_path = sys.argv[1]
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "brain_tumor_model.keras")
    image = Image.open(image_path).convert("RGB")
    # Match app.py and the embedded EfficientNet rescaling layer: keep pixels in the raw 0–255 range.
    array = np.asarray(image.resize(IMG_SIZE), dtype=np.float32)
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    result = {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": round(float(probabilities[predicted_index]) * 100, 2),
        "probabilities": {name: round(float(probabilities[index]) * 100, 2) for index, name in enumerate(CLASS_NAMES)},
        "originalImage": image_to_base64(image),
        "gradcam": generate_gradcam(model, image, predicted_index),
        "imageWidth": image.width,
        "imageHeight": image.height,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
