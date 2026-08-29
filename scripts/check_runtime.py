import importlib.util
print({name: bool(importlib.util.find_spec(name)) for name in ["tensorflow", "PIL", "numpy", "matplotlib"]})
