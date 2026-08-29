import json
import os
import tensorflow as tf

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "brain_tumor_model.keras")
model = tf.keras.models.load_model(model_path)
output_layer = model.layers[-1]
print(json.dumps({
    "input_shape": model.input_shape,
    "output_shape": model.output_shape,
    "layer_names": [layer.name for layer in model.layers],
    "final_layer": output_layer.name,
    "final_activation": getattr(getattr(output_layer, "activation", None), "__name__", None),
}, indent=2))
