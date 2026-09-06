"""
Retrain on the combined dataset: the original Kaggle Brain Tumor MRI Dataset
(masoudnickparvar) + BRISC2025 (physician-validated, multi-plane, published
academic dataset), merged into one train/test split.

Warm-starts from the current model/brain_tumor_model.keras (round 3, 300x300,
full backbone) since the combined data should refine rather than replace what
it already learned.
"""

import json
import os
import shutil

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (300, 300)
BATCH_SIZE = 16

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NICKPARVAR_DIR = os.path.join(BASE_DIR, "data", "brain-tumor-mri-dataset")
BRISC_DIR = os.path.join(BASE_DIR, "data", "brisc2025")
MERGED_DIR = os.path.join(BASE_DIR, "data", "merged")
MODEL_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_model.keras")
REPORT_OUT = os.path.join(BASE_DIR, "model", "training_report.json")

AUTOTUNE = tf.data.AUTOTUNE

# BRISC2025 uses "no_tumor" -- map to our "notumor" naming.
BRISC_CLASS_MAP = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "no_tumor": "notumor",
}


def build_merged_dataset():
    """Symlink both sources' images into data/merged/{Training,Testing}/<class>/
    so tf.keras.utils.image_dataset_from_directory can read them as one set.
    Keeps each source's own train/test split (no leakage across the split)."""
    if os.path.isdir(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)

    for split, nick_split, brisc_split in [
        ("Training", "Training", "train"),
        ("Testing", "Testing", "test"),
    ]:
        for class_name in CLASS_NAMES:
            out_dir = os.path.join(MERGED_DIR, split, class_name)
            os.makedirs(out_dir, exist_ok=True)

        nick_split_dir = os.path.join(NICKPARVAR_DIR, nick_split)
        for class_name in CLASS_NAMES:
            src_dir = os.path.join(nick_split_dir, class_name)
            out_dir = os.path.join(MERGED_DIR, split, class_name)
            for fname in os.listdir(src_dir):
                os.symlink(
                    os.path.join(src_dir, fname),
                    os.path.join(out_dir, f"nick_{fname}"),
                )

        for brisc_class, our_class in BRISC_CLASS_MAP.items():
            src_dir = os.path.join(
                BRISC_DIR, "brisc2025", "classification_task", brisc_split, brisc_class
            )
            if not os.path.isdir(src_dir):
                continue
            out_dir = os.path.join(MERGED_DIR, split, our_class)
            for fname in os.listdir(src_dir):
                os.symlink(
                    os.path.join(src_dir, fname),
                    os.path.join(out_dir, f"brisc_{fname}"),
                )

    for split in ["Training", "Testing"]:
        for class_name in CLASS_NAMES:
            n = len(os.listdir(os.path.join(MERGED_DIR, split, class_name)))
            print(f"[Merged] {split}/{class_name}: {n} images", flush=True)


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
    print("[Merged] Building merged dataset directory...", flush=True)
    build_merged_dataset()

    print("[Merged] Loading datasets...", flush=True)
    train_ds, val_ds, test_ds = load_datasets()
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    print("[Merged] Loading current model...", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)

    print("[Merged] Baseline test evaluation (before merged-data fine-tune):", flush=True)
    evaluate(model, test_ds)

    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    frozen_bn = 0
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
    print(f"[Merged] Unfroze full backbone, kept {frozen_bn} BatchNorm layers frozen", flush=True)

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
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    print("[Merged] Fine-tuning on merged dataset...", flush=True)
    model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks)

    print("[Merged] Final test evaluation:", flush=True)
    report, cm = evaluate(model, test_ds)

    print(f"[Merged] Saving model to {MODEL_PATH}", flush=True)
    model.save(MODEL_PATH)

    with open(REPORT_OUT, "w") as f:
        json.dump(
            {"classification_report": report, "confusion_matrix": cm,
             "class_order": CLASS_NAMES, "img_size": list(IMG_SIZE),
             "data_sources": ["masoudnickparvar/brain-tumor-mri-dataset", "briscdataset/brisc2025"]},
            f, indent=2,
        )

    print("[Merged] Done.", flush=True)


if __name__ == "__main__":
    main()
