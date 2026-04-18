import torch
import os
import json
from .models import get_model
from .checkpoint_compat import load_checkpoint_strict

# Local configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = ["resnet50", "efficientnet", "mobilenet"]

def load_models(num_classes):
    """Loading and initializing candidate architectural nodes."""
    models = []
    
    # Check weight availability
    for name in MODEL_NAMES:
        weight_path = f"weights/{name}.pth"
        if not os.path.exists(weight_path):
            print(f"[ERROR] Weight Node {name} not found at {weight_path}. Retrain requested.")
            continue
            
        try:
            model = get_model(name, num_classes)
            # Robust mapping for half precision or CPU context
            state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=False)
            load_checkpoint_strict(model, state_dict, source=weight_path)
            model.to(DEVICE)
            model.eval()
            models.append((name, model))
        except Exception as e:
            print(f"[ERROR] Failed to initialize node {name}: {e}")

    return models


def ensemble_predict(models, image_tensor):
    """Synchronized Multi-Model Architectural Consensus Predictor."""
    outputs = []
    
    if not models:
        return None, 0.0, None

    # Aggregate weighted probability mappings
    for name, model in models:
        with torch.no_grad():
            # Softmax to standardize confidence levels across heads
            out = torch.softmax(model(image_tensor), dim=1)
            outputs.append(out)

    # Average Consensus
    avg = torch.mean(torch.stack(outputs), dim=0)
    conf, pred = torch.max(avg, 1)

    return pred.item(), conf.item(), outputs
