import json
import os
import tensorflow as tf

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "brain_tumor_model.keras")
model = tf.keras.models.load_model(model_path)
backbone = model.layers[0]
print(json.dumps({
    "backbone": backbone.name,
    "backbone_config": {
        "include_preprocessing": backbone.get_config().get("include_preprocessing"),
        "weights": backbone.get_config().get("weights"),
    },
    "first_backbone_layers": [
        {"name": layer.name, "class": layer.__class__.__name__}
        for layer in backbone.layers[:5]
    ],
}, indent=2))
