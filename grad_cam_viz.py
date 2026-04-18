import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms

# CONFIG
MODEL_PATH = "best_model_run_0.pth"
IMAGE_PATH = r"C:\Users\mysel\Downloads\images_test\shirt_test.jpg"  # Example
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Load model
    model = models.efficientnet_b0(weights=None)
    # Re-build classifier to match training
    num_classes = 20  # Assuming same number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(0.5), torch.nn.Linear(in_features, num_classes)
    )

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)

    # Target layer for Grad-CAM (usually the last convolutional layer)
    target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    # Process image
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        return

    rgb_img = np.array(Image.open(IMAGE_PATH).convert("RGB")) / 255.0
    input_tensor = (
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )(Image.open(IMAGE_PATH).convert("RGB"))
        .unsqueeze(0)
        .to(DEVICE)
    )

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # Overlay CAM on original image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save or show
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM Explainability")

    plt.savefig("grad_cam_output.png")
    print("Grad-CAM result saved at grad_cam_output.png")


if __name__ == "__main__":
    main()
