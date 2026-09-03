import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (300, 300)
OUTPUT_MAX_DIM = 512

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_model.tflite")
FEATURE_MODEL_PATH = os.path.join(BASE_DIR, "model", "feature_extractor.tflite")

DENSE_W = np.load(os.path.join(BASE_DIR, "model", "dense_w.npy"))
DENSE_B = np.load(os.path.join(BASE_DIR, "model", "dense_b.npy"))
DENSE1_W = np.load(os.path.join(BASE_DIR, "model", "dense1_w.npy"))
DENSE1_B = np.load(os.path.join(BASE_DIR, "model", "dense1_b.npy"))

INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None

FEATURE_INTERPRETER = None
FEATURE_INPUT_DETAILS = None
FEATURE_OUTPUT_DETAILS = None


def get_interpreter():
    global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS

    if INTERPRETER is None:
        print("[Inference] Loading prediction model...", file=sys.stderr, flush=True)
        INTERPRETER = Interpreter(model_path=MODEL_PATH)
        INTERPRETER.allocate_tensors()
        INPUT_DETAILS = INTERPRETER.get_input_details()
        OUTPUT_DETAILS = INTERPRETER.get_output_details()
        print("[Inference] Prediction model loaded.", file=sys.stderr, flush=True)

    return INTERPRETER


def get_feature_interpreter():
    global FEATURE_INTERPRETER, FEATURE_INPUT_DETAILS, FEATURE_OUTPUT_DETAILS

    if FEATURE_INTERPRETER is None:
        print("[Inference] Loading feature extractor...", file=sys.stderr, flush=True)
        FEATURE_INTERPRETER = Interpreter(model_path=FEATURE_MODEL_PATH)
        FEATURE_INTERPRETER.allocate_tensors()
        FEATURE_INPUT_DETAILS = FEATURE_INTERPRETER.get_input_details()
        FEATURE_OUTPUT_DETAILS = FEATURE_INTERPRETER.get_output_details()
        print("[Inference] Feature extractor loaded.", file=sys.stderr, flush=True)

    return FEATURE_INTERPRETER


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


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def generate_gradcam(image_array: np.ndarray, predicted_index: int, output_image: Image.Image) -> str:
    feature_interpreter = get_feature_interpreter()

    input_data = np.expand_dims(image_array, axis=0)
    feature_interpreter.set_tensor(FEATURE_INPUT_DETAILS[0]["index"], input_data)
    feature_interpreter.invoke()
    features = feature_interpreter.get_tensor(FEATURE_OUTPUT_DETAILS[0]["index"])[0]

    pooled = features.mean(axis=(0, 1))

    z1 = pooled @ DENSE_W + DENSE_B
    a1 = np.maximum(z1, 0)

    dz2 = np.zeros(len(CLASS_NAMES))
    dz2[predicted_index] = 1.0

    da1 = DENSE1_W @ dz2
    dz1 = da1 * (z1 > 0)
    dpooled = DENSE_W @ dz1

    cam = np.maximum(np.sum(features * dpooled, axis=-1), 0)

    max_value = cam.max()
    if max_value > 0:
        cam = cam / max_value

    heatmap_img = Image.fromarray((cam * 255).astype(np.uint8))
    heatmap_resized = heatmap_img.resize(
        (output_image.width, output_image.height), Image.BILINEAR
    )
    heatmap_arr = np.asarray(heatmap_resized, dtype=np.float32) / 255.0

    base = np.asarray(output_image, dtype=np.float32) / 255.0
    heat_rgb = colorize_heatmap(heatmap_arr)

    alpha = 0.45
    overlay = base * (1 - alpha) + heat_rgb * alpha
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def analyze(image_path: str):
    interpreter = get_interpreter()

    image = Image.open(image_path).convert("RGB")

    array = np.asarray(image.resize(IMG_SIZE), dtype=np.float32)
    input_data = np.expand_dims(array, axis=0)

    interpreter.set_tensor(INPUT_DETAILS[0]["index"], input_data)
    interpreter.invoke()
    probabilities = interpreter.get_tensor(OUTPUT_DETAILS[0]["index"])[0]

    predicted_index = int(np.argmax(probabilities))

    output_image = downscale_for_output(image)

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": round(float(probabilities[predicted_index]) * 100, 2),
        "probabilities": {
            name: round(float(probabilities[index]) * 100, 2)
            for index, name in enumerate(CLASS_NAMES)
        },
        "originalImage": image_to_base64(output_image),
        "gradcam": generate_gradcam(array, predicted_index, output_image),
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
            print(json.dumps({"ok": True, "result": result}), flush=True)
        except Exception as error:
            print(json.dumps({"ok": False, "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()