import tensorflow as tf

model = tf.keras.models.load_model("model/brain_tumor_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model/brain_tumor_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Converted successfully: model/brain_tumor_model.tflite")