import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# External local imports
from src.core.config import *
from src.core.dataset import get_dataloaders
from src.core.models_collection import CustomCNN, SimpleANN, get_resnet50

# Senior CV Engineer Note: Efficiently assigning GPU hardware if CUDA is enabled implicitly
# PyTorch automatically assigns ops to CUDA without breaking the CPU pipeline seamlessly.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_pytorch_model(model_name, model, train_loader, val_loader, num_epochs=3):
    """
    Trains a given PyTorch model and logs the highest training and validation accuracies achieved.
    The process safely falls back to CPU if a discrete GPU card isn't accessible without failing.
    """
    print(f"\n{'=' * 50}\nTraining {model_name} on {DEVICE}...\n{'=' * 50}")
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0
    final_train_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_corrects = 0
        total_train = 0

        for inputs, labels in tqdm(
            train_loader, desc=f"{model_name} Epoch {epoch + 1}/{num_epochs} [TRAIN]"
        ):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_corrects += torch.sum(preds == labels.data)
            total_train += inputs.size(0)

        epoch_train_acc = (train_corrects.double() / total_train).item()

        # Validation Phase
        model.eval()
        val_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader, desc=f"{model_name} Epoch {epoch + 1}/{num_epochs} [VAL]"
            ):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                total_val += inputs.size(0)

        epoch_val_acc = (val_corrects.double() / total_val).item()

        print(
            f"[{model_name}] Epoch {epoch + 1} -> Train Accuracy: {epoch_train_acc:.4f} | Validation Accuracy: {epoch_val_acc:.4f}"
        )

        # Track the peak validation accuracy for robust benchmarking
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            final_train_acc = epoch_train_acc
            # Note: We safely checkpoint the weights to disk avoiding data serialization faults
            torch.save(
                model.state_dict(),
                os.path.join(PROJECT_ROOT, f"best_{model_name.lower()}.pth"),
            )

    return final_train_acc, best_val_acc


def synthesize_mock_metrics():
    """
    Provides simulated telemetry to plot the framework without waiting 100 hours on a CPU.
    Professional environments test charting components with synthesized scalars before expensive cluster runs.
    """
    print(
        "\n[INFO] Simulating pipeline completion for immediate demonstration of the architecture and dashboard."
    )
    return {
        "Model Architecture": [
            "ANN (Multi-Layer Perceptron)",
            "Custom CNN",
            "ResNet-50 (Transfer Learning)",
        ],
        "Training Accuracy": [0.45, 0.78, 0.92],
        "Testing/Val Accuracy": [0.38, 0.72, 0.88],
    }


def plot_metrics(metrics_dict):
    """
    Plots a consolidated bar chart utilizing seaborn to compare the performance matrices.
    """
    df_metrics = pd.DataFrame(metrics_dict)

    # Melt the dataframe strictly for grouped bar chart compatibility
    df_melted = df_metrics.melt(
        id_vars="Model Architecture", var_name="Metric", value_name="Accuracy"
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x="Model Architecture",
        y="Accuracy",
        hue="Metric",
        data=df_melted,
        palette="viridis",
    )
    plt.title(
        "Computer Vision Model Architectures Capability Comparison Matrix", fontsize=16
    )
    plt.ylabel("Accuracy Profile", fontsize=13)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)

    # Render percentage annotations dynamically
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1%}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            xytext=(0, 10),
            textcoords="offset points",
        )

    # Save to rigid artifact natively seamlessly
    plt.tight_layout()
    chart_path = os.path.join(PROJECT_ROOT, "model_comparison_chart.png")
    plt.savefig(chart_path, dpi=300)
    print(
        f"\n[Artifact Output] Bar chart reliably populated and saved at: {chart_path}"
    )

    # Save the exact metrics structure
    df_metrics.to_csv(os.path.join(PROJECT_ROOT, "detailed_metrics.csv"), index=False)


def run_project_pipeline(demo_mode, ann_epochs, cnn_epochs, resnet_epochs):
    print("Initiating Enterprise Computer Vision Model Benchmark Engine...\n")

    # For quick demo / chart-only evaluation, keep demo_mode=True.
    if demo_mode:
        metrics = synthesize_mock_metrics()
        plot_metrics(metrics)
    else:
        # Load production dataloaders
        train_loader, val_loader, test_loader, idx_to_label = get_dataloaders(
            CSV_FILE, IMAGES_DIR, BATCH_SIZE, IMAGE_SIZE
        )
        num_classes = len(idx_to_label)

        # Keep label<->index mapping consistent for inference.
        with open(CLASS_MAPPING_FILE, "w") as f:
            json.dump(idx_to_label, f)

        # 1. Simple ANN
        ann_model = SimpleANN(num_classes=num_classes)
        ann_t, ann_v = train_pytorch_model(
            "ANN", ann_model, train_loader, val_loader, num_epochs=ann_epochs
        )

        # 2. Custom CNN
        cnn_model = CustomCNN(num_classes=num_classes)
        cnn_t, cnn_v = train_pytorch_model(
            "Custom_CNN", cnn_model, train_loader, val_loader, num_epochs=cnn_epochs
        )

        # 3. ResNet50
        resnet = get_resnet50(num_classes)
        res_t, res_v = train_pytorch_model(
            "ResNet50", resnet, train_loader, val_loader, num_epochs=resnet_epochs
        )

        actual_metrics = {
            "Model Architecture": ["ANN", "Custom CNN", "ResNet-50"],
            "Training Accuracy": [ann_t, cnn_t, res_t],
            "Testing/Val Accuracy": [ann_v, cnn_v, res_v],
        }
        plot_metrics(actual_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--demo", action="store_true", help="Generate metrics/chart using mock values."
    )
    group.add_argument(
        "--train",
        action="store_true",
        help="Actually train ANN/CNN/ResNet and generate metrics/chart.",
    )

    # Submission-friendly defaults:
    # - With no flags, we run REAL training (not mock metrics).
    # - Keep default epochs small for CPU environments; override as needed.
    parser.add_argument("--ann-epochs", type=int, default=1)
    parser.add_argument("--cnn-epochs", type=int, default=2)
    parser.add_argument("--resnet-epochs", type=int, default=2)
    args = parser.parse_args()

    demo_mode = bool(args.demo)

    run_project_pipeline(
        demo_mode=demo_mode,
        ann_epochs=args.ann_epochs,
        cnn_epochs=args.cnn_epochs,
        resnet_epochs=args.resnet_epochs,
    )
