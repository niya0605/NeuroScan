"""
Retrain brain_tumor_model.keras on the standard 4-class Brain Tumor MRI Dataset
(glioma, meningioma, notumor, pituitary), fine-tuning an EfficientNetB0 backbone.

Usage:
    python3 scripts/train_model.py

Expects the dataset at data/brain-tumor-mri-dataset/{Training,Testing}/<class>/*.jpg
(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Writes:
    model/brain_tumor_model.keras   (new trained model)
    model/training_report.json      (per-class precision/recall/f1, confusion matrix)
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
MODEL_OUT = os.path.join(BASE_DIR, "model", "brain_tumor_model.keras")
REPORT_OUT = os.path.join(BASE_DIR, "model", "training_report.json")

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        subset="training",
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        subset="validation",
        seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_ds, val_ds, test_ds


def build_model():
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = augmentation(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, backbone


def evaluate(model, test_ds):
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("Confusion matrix (rows=true, cols=pred):")
    print(CLASS_NAMES)
    for row in cm:
        print(row)
    return report, cm


def main():
    print("[Train] Loading datasets...", flush=True)
    train_ds, val_ds, test_ds = load_datasets()

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    print("[Train] Building model...", flush=True)
    model, backbone = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
    ]

    print("[Train] Phase 1: training head with frozen backbone...", flush=True)
    model.fit(train_ds, validation_data=val_ds, epochs=12, callbacks=callbacks)

    print("[Train] Phase 1 test evaluation:", flush=True)
    evaluate(model, test_ds)

    print("[Train] Phase 2: fine-tuning top of backbone...", flush=True)
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
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

    print("[Train] Final test evaluation:", flush=True)
    report, cm = evaluate(model, test_ds)

    print(f"[Train] Saving model to {MODEL_OUT}", flush=True)
    model.save(MODEL_OUT)

    with open(REPORT_OUT, "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm, "class_order": CLASS_NAMES}, f, indent=2)

    print("[Train] Done.", flush=True)


if __name__ == "__main__":
    main()
