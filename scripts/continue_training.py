"""
Continue fine-tuning brain_tumor_model.keras from its current weights.
Same top-30-unfrozen backbone config as train_model.py Phase 2, just more
epochs with a larger early-stopping patience since it hadn't converged yet.
"""

import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "brain-tumor-mri-dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")
MODEL_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_model.keras")
REPORT_OUT = os.path.join(BASE_DIR, "model", "training_report.json")

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels="inferred", label_mode="categorical",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.1, subset="training", seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels="inferred", label_mode="categorical",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.1, subset="validation", seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, labels="inferred", label_mode="categorical",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_ds, val_ds, test_ds


def evaluate(model, test_ds):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("Confusion matrix (rows=true, cols=pred):", CLASS_NAMES)
    for row in cm:
        print(row)
    return report, cm


def main():
    print("[Continue] Loading datasets...", flush=True)
    train_ds, val_ds, test_ds = load_datasets()
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    print("[Continue] Loading model...", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)

    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-7
        ),
    ]

    print("[Continue] Training more epochs...", flush=True)
    model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)

    print("[Continue] Test evaluation:", flush=True)
    report, cm = evaluate(model, test_ds)

    print(f"[Continue] Saving model to {MODEL_PATH}", flush=True)
    model.save(MODEL_PATH)

    with open(REPORT_OUT, "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm, "class_order": CLASS_NAMES}, f, indent=2)

    print("[Continue] Done.", flush=True)


if __name__ == "__main__":
    main()
