import torchvision.models as models
import torch.nn as nn

def get_model(name, num_classes):
    """Factory function for CV model architectures."""
    if name == "resnet50":
        # Professional ResNet50 with Ensemble MLP head
        model = models.resnet50(weights="DEFAULT")
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    elif name == "efficientnet":
        # Switched to B0 to resolve 32 vs 40 channel mismatch in checkpoint
        model = models.efficientnet_b0(weights="DEFAULT")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_features, num_classes)
        )

    elif name == "mobilenet":
        # Simplified head to match checkpoint keys (classifier.3.weight instead of classifier.3.1.weight)
        model = models.mobilenet_v3_large(weights="DEFAULT")
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif name == "ann":
        from .models_collection import SimpleANN
        model = SimpleANN(num_classes)
        
    elif name == "custom_cnn":
        from .models_collection import CustomCNN
        model = CustomCNN(num_classes)

    elif name == "vgg16":
        # Professional VGG16 with transfer learning head
        model = models.vgg16(weights="DEFAULT")
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    else:
        raise ValueError(f"Unknown model: {name}")

    return model
