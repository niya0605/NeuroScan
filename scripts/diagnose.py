import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

interpreter = Interpreter(model_path="model/brain_tumor_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("INPUT DETAILS:", input_details)
print("OUTPUT DETAILS:", output_details)

for path in ["PATH_TO_IMAGE_1", "PATH_TO_IMAGE_2"]:
    image = Image.open(path).convert("RGB").resize((300, 300))
    array = np.asarray(image, dtype=np.float32)
    input_data = np.expand_dims(array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    print(f"\n{path}")
    print("Raw probabilities:", probs)