import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model/brain_tumor_model.keras")

efficientnet = model.layers[0]

feature_model = tf.keras.models.Model(
    inputs=efficientnet.input,
    outputs=efficientnet.get_layer("top_conv").output,
)

converter = tf.lite.TFLiteConverter.from_keras_model(feature_model)
tflite_features = converter.convert()

with open("model/feature_extractor.tflite", "wb") as f:
    f.write(tflite_features)

dense_layer = model.get_layer("dense")
dense_1_layer = model.get_layer("dense_1")

dense_w, dense_b = dense_layer.get_weights()
dense1_w, dense1_b = dense_1_layer.get_weights()

np.save("model/dense_w.npy", dense_w)
np.save("model/dense_b.npy", dense_b)
np.save("model/dense1_w.npy", dense1_w)
np.save("model/dense1_b.npy", dense1_b)

print("Exported: feature_extractor.tflite, dense_w.npy, dense_b.npy, dense1_w.npy, dense1_b.npy")