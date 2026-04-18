import torch.nn as nn
from torchvision import models

# ==============================================================================
# SENIOR CV ENGINEER NOTES:
# This file contains various architectural definitions.
# We implement a Simple Artificial Neural Network (ANN), a Custom Convolutional
# Neural Network (CNN), and leverage a pre-trained ResNet-50.
# YOLOv8 classification is handled natively via the 'ultralytics' library in the
# master training script.
# ==============================================================================


class SimpleANN(nn.Module):
    """
    A basic Artificial Neural Network (Multi-Layer Perceptron).
    This flattens the image and passes it through dense layers.
    Note: ANNs lose spatial hierarchy, so its performance on CV tasks will be poor.
    """

    def __init__(self, num_classes=20, image_size=224):
        super(SimpleANN, self).__init__()
        # Flattened spatial dimensions: 3 channels * width * height
        input_dim = 3 * image_size * image_size

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # Regularization
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class CustomCNN(nn.Module):
    """
    A Custom Convolutional Neural Network built from scratch.
    Learns spatial features using Convolutional filters.
    """

    def __init__(self, num_classes=20):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces spatial dimensions by 2 (224 -> 112)
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56 -> 28
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_resnet50(num_classes):
    """
    Leverages heavily optimized Transfer Learning via ResNet-50.
    This effectively uses pre-learned Imagenet representations.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    # We replace the final layer to adapt to our specific top layer classification.
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def get_vgg16(num_classes):
    """
    Classic VGG-16 architecture. 
    Known for simplicity and deep representations, though more parameter-heavy than ResNet.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # VGG16 uses 'classifier' block instead of 'fc'
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model
