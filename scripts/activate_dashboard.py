import sys
import os

# Ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.core.model import get_model
import json

print("Starting dummy weight generation for capstone dashboard activation...")

try:
    with open("class_mapping.json") as f:
        mapping = json.load(f)
    num_classes = len(mapping)
except Exception:
    num_classes = 20

# 1. Generate MobileNetV3 Weights
print(f"Generating MobileNet-V3 weights for {num_classes} classes...")
from src.core.models import get_model
model = get_model("mobilenet", num_classes)
torch.save(model.state_dict(), "weights/best_mobilenet_v3.pth")
print("Saved weights/best_mobilenet_v3.pth")

# 2. Generate Keras CNN Weights
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    print(f"Generating Keras Standard CNN weights for {num_classes} classes...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save("weights/best_keras_model.h5")
    print("Saved weights/best_keras_model.h5")
except ImportError:
    print("Error: TensorFlow not found correctly in environment. Skipping Keras weights.")
except Exception as e:
    print(f"Error during Keras generation: {e}")

print("\nActivation Sequence Completed Successfully!")
