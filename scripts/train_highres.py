"""
Continue fine-tuning brain_tumor_model.keras at higher resolution (300x300
instead of 224x224, closer to the dataset's native 512x512) with the full
EfficientNetB0 backbone unfrozen (BatchNorm layers excepted, to keep their
running statistics stable on a small dataset).

Warm-starts from the current model/brain_tumor_model.keras weights --
EfficientNetB0 is fully convolutional up to the pooling layer, so existing
conv weights transfer cleanly to a new input resolution.
"""

import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (300, 300)
BATCH_SIZE = 16  # smaller batch: higher res costs more memory/compute per image

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


def rebuild_at_new_resolution(old_model):
    """Rebuild the functional model graph at IMG_SIZE, transferring weights
    layer-by-layer (conv/dense weights are resolution-independent)."""
    old_backbone = old_model.get_layer("efficientnetb0")
    old_dense = old_model.get_layer("dense")
    old_dense1 = old_model.get_layer("dense_1")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    new_backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=(*IMG_SIZE, 3),
    )
    new_backbone.set_weights(old_backbone.get_weights())

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = new_backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.get_layer("dense").set_weights(old_dense.get_weights())
    model.get_layer("dense_1").set_weights(old_dense1.get_weights())

    return model, new_backbone


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
    print("[HighRes] Loading datasets at", IMG_SIZE, flush=True)
    train_ds, val_ds, test_ds = load_datasets()
    train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    print("[HighRes] Loading current model and rebuilding at new resolution...", flush=True)
    old_model = tf.keras.models.load_model(MODEL_PATH)
    model, backbone = rebuild_at_new_resolution(old_model)

    print("[HighRes] Baseline test evaluation (before high-res fine-tune):", flush=True)
    evaluate(model, test_ds)

    backbone.trainable = True
    frozen_bn = 0
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
    print(f"[HighRes] Unfroze full backbone, kept {frozen_bn} BatchNorm layers frozen", flush=True)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = os.path.join(BASE_DIR, "model", "brain_tumor_model.checkpoint.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-7
        ),
        # Save after every epoch so an interrupted run doesn't lose all progress --
        # only overwrites when val_accuracy improves, so this always holds the best
        # weights seen so far even if the process gets killed mid-run.
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    print("[HighRes] Fine-tuning full backbone at high resolution...", flush=True)
    model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks)

    print("[HighRes] Final test evaluation:", flush=True)
    report, cm = evaluate(model, test_ds)

    print(f"[HighRes] Saving model to {MODEL_PATH}", flush=True)
    model.save(MODEL_PATH)

    with open(REPORT_OUT, "w") as f:
        json.dump(
            {"classification_report": report, "confusion_matrix": cm,
             "class_order": CLASS_NAMES, "img_size": list(IMG_SIZE)},
            f, indent=2,
        )

    print("[HighRes] Done.", flush=True)


if __name__ == "__main__":
    main()
