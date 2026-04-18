import torch
import torch.nn as nn
from torchvision import models
import os
import json

# Setup
from src.core.config import PROJECT_ROOT, CLASS_MAPPING_FILE as MAP_PATH

def activate_vgg_node():
    # Load class mapping
    if os.path.exists(MAP_PATH):
        with open(MAP_PATH) as f:
            idx_to_label = json.load(f)
        num_classes = len(idx_to_label)
    else:
        num_classes = 20

    # Initialize model with training head
    print(f"Initializing VGG16 with {num_classes} output nodes...")
    model = models.vgg16(weights="DEFAULT")
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    # Save to weights directory
    target_path = os.path.join(PROJECT_ROOT, "weights", "vgg16.pth")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Save formatted state dict
    torch.save(model.state_dict(), target_path)
    print(f"Successfully activated VGG16 node at {target_path}")

if __name__ == "__main__":
    activate_vgg_node()
