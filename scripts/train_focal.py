"""
Continue fine-tuning brain_tumor_model.keras on the merged dataset using
categorical focal loss instead of plain cross-entropy. Focal loss down-weights
already-easy/well-classified examples and concentrates gradient on hard ones
(gamma controls how aggressively) -- a different mechanism than class_weight
(which just uniformly reweights a class regardless of how hard each example
already is), specifically aimed at the glioma/meningioma boundary confusion
that class-weighting already failed to move.
"""

import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (300, 300)
BATCH_SIZE = 16
GAMMA = 2.0  # standard focal loss focusing parameter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_DIR = os.path.join(BASE_DIR, "data", "merged")
MODEL_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_model.keras")
REPORT_OUT = os.path.join(BASE_DIR, "model", "training_report.json")

AUTOTUNE = tf.data.AUTOTUNE


def categorical_focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    return loss_fn


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(MERGED_DIR, "Training"), labels="inferred", label_mode="categorical",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.1, subset="training", seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(MERGED_DIR, "Training"), labels="inferred", label_mode="categorical",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.1, subset="validation", seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(MERGED_DIR, "Testing"), labels="inferred", label_mode="categorical",
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
    print("[Focal] Loading merged datasets...", flush=True)
    train_ds, val_ds, test_ds = load_datasets()
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    print("[Focal] Loading current model (round 4)...", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)

    print("[Focal] Baseline test evaluation:", flush=True)
    evaluate(model, test_ds)

    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-6, weight_decay=1e-5),
        loss=categorical_focal_loss(gamma=GAMMA),
        metrics=["accuracy"],
    )

    checkpoint_path = os.path.join(BASE_DIR, "model", "brain_tumor_model.checkpoint.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    print(f"[Focal] Fine-tuning with focal loss (gamma={GAMMA})...", flush=True)
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

    print("[Focal] Final test evaluation:", flush=True)
    report, cm = evaluate(model, test_ds)

    print(f"[Focal] Saving model to {MODEL_PATH}", flush=True)
    model.save(MODEL_PATH)

    with open(REPORT_OUT, "w") as f:
        json.dump(
            {"classification_report": report, "confusion_matrix": cm,
             "class_order": CLASS_NAMES, "img_size": list(IMG_SIZE), "focal_gamma": GAMMA},
            f, indent=2,
        )

    print("[Focal] Done.", flush=True)


if __name__ == "__main__":
    main()
