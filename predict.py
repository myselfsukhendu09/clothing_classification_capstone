import argparse
import json
import os

import torch
from PIL import Image
from torchvision import transforms

from checkpoint_compat import load_checkpoint_strict
from config import CLASS_MAPPING_FILE, IMAGE_SIZE, MODEL_SAVE_PATH
from model import get_model


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if necessary files exists
    if not os.path.exists(CLASS_MAPPING_FILE):
        print(
            f"Error: class mapping not found at {CLASS_MAPPING_FILE}. Please run the training script first."
        )
        return

    if not os.path.exists(MODEL_SAVE_PATH):
        print(
            f"Error: model checkpoint not found at {MODEL_SAVE_PATH}. Please train the model first."
        )
        return

    if not os.path.exists(image_path):
        print(f"Error: image path not found: {image_path}")
        return

    # Load class mappings
    with open(CLASS_MAPPING_FILE, "r") as f:
        idx_to_label = json.load(f)

    num_classes = len(idx_to_label)

    # Initialize the model
    model = get_model("resnet50", num_classes)
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    load_checkpoint_strict(model, checkpoint, source=MODEL_SAVE_PATH)
    model.to(device)
    model.eval()

    # Preprocess image
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to load image at {image_path}: {e}")
        return

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Perform Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probabilities, 1)
        topk_probs, topk_classes = torch.topk(
            probabilities, min(3, probabilities.size(1))
        )

    class_idx = str(top_class.item())
    class_name = idx_to_label.get(class_idx, "Unknown")
    confidence = top_prob.item() * 100

    if (confidence / 100) < 0.6:
        class_name = f"Uncertain prediction ({class_name})"

    print("\n--- Inference Result ---")
    print(f"File: {image_path}")
    print(f"Predicted Class: {class_name}")
    print(f"Confidence: {confidence:.2f}%\n")
    print("--- Top-K Confidence Spread ---")
    for i in range(topk_probs.size(1)):
        c_idx = str(topk_classes[0, i].item())
        print(
            f"{i + 1}. {idx_to_label.get(c_idx, 'Unknown')}: {topk_probs[0, i].item() * 100:.2f}%"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict clothing class from a single image."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Absolute or relative path to the image.",
    )
    args = parser.parse_args()

    predict(args.image)
