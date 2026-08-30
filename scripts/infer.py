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
OUTPUT_MAX_DIM = 512  # NEW: cap the size of images sent back to the client

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "brain_tumor_model.keras",
)

MODEL = None


def get_model():
    global MODEL

    if MODEL is None:
        print("[Inference] Loading model...", file=sys.stderr, flush=True)
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        print("[Inference] Model loaded.", file=sys.stderr, flush=True)

    return MODEL


def downscale_for_output(image: Image.Image) -> Image.Image:
    if max(image.size) <= OUTPUT_MAX_DIM:
        return image

    ratio = OUTPUT_MAX_DIM / max(image.size)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    return image.resize(new_size, Image.LANCZOS)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_gradcam(model, image, predicted_index):
    efficientnet = model.layers[0]

    resized = image.resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    tensor = np.expand_dims(array, axis=0)

    _ = efficientnet(tensor, training=False)

    grad_model = tf.keras.models.Model(
        inputs=efficientnet.input,
        outputs=[
            efficientnet.get_layer("top_conv").output,
            efficientnet.output,
        ],
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

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap,
    ).numpy()

    output_image = downscale_for_output(image)

    heatmap_resized = (
        tf.image.resize(
            heatmap[..., np.newaxis],
            (output_image.height, output_image.width),
        )
        .numpy()
        .squeeze()
    )

    figure = plt.figure(figsize=(6, 5), frameon=False)
    axis = figure.add_axes([0, 0, 1, 1])

    axis.imshow(output_image)
    axis.imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    axis.axis("off")

    buffer = io.BytesIO()

    figure.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=90,
    )

    plt.close(figure)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def analyze(image_path: str):
    model = get_model()

    image = Image.open(image_path).convert("RGB")

    array = np.asarray(
        image.resize(IMG_SIZE),
        dtype=np.float32,
    )

    probabilities = model.predict(
        np.expand_dims(array, axis=0),
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))

    output_image = downscale_for_output(image)

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index]) * 100,
            2,
        ),
        "probabilities": {
            name: round(
                float(probabilities[index]) * 100,
                2,
            )
            for index, name in enumerate(CLASS_NAMES)
        },
        "originalImage": image_to_base64(output_image),
        "gradcam": generate_gradcam(
            model,
            image,
            predicted_index,
        ),
        "imageWidth": output_image.width,
        "imageHeight": output_image.height,
    }


def main():
    for line in sys.stdin:
        line = line.strip()

        if not line:
            continue

        try:
            request = json.loads(line)
            image_path = request["image_path"]

            result = analyze(image_path)

            print(
                json.dumps(
                    {
                        "ok": True,
                        "result": result,
                    }
                ),
                flush=True,
            )

        except Exception as error:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(error),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()