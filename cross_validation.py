import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from config import *
from dataset import ClothingDataset
from models_collection import CustomCNN

# ==============================================================================
# K-Fold Stratified Cross-Validation
# ------------------------------------------------------------------------------
# Deep learning models on large computer vision datasets typically employ a fixed
# train/val/test split to avoid N*H computing hours (where N is epochs,
# H is hours per split).
# However, to rigorously validate the stability of our architecture's feature
# extraction, we employ Stratified 5-Fold Cross Validation here.
# This guarantees every image is tested precisely once, averaging overall structural
# capability natively.
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perform_kfold_cv(
    csv_file, img_dir, batch_size, image_size, k_splits=5, epochs_per_fold=3
):
    print(f"\n[INFO] Launching {k_splits}-Fold Stratified Cross-Validation...")
    print(f"[INFO] Hardware detected: {DEVICE}")

    # 1. Load Data
    df = pd.read_csv(csv_file).dropna(subset=["label"])

    # Encode labels mathematically
    labels = sorted(df["label"].unique())
    label_to_idx = {l: i for i, l in enumerate(labels)}
    idx_to_label = {i: l for i, l in enumerate(labels)}
    df["label_idx"] = df["label"].map(label_to_idx)
    num_classes = len(labels)

    # 2. Extract Data Augmentations
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ClothingDataset(df, img_dir, transform=None)  # Base dataset
    labels_arr = df["label_idx"].values

    # 3. Stratified K-Fold setup
    kfold = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=42)

    fold_results = {}

    for fold, (train_ids, val_ids) in enumerate(
        kfold.split(np.zeros(len(labels_arr)), labels_arr)
    ):
        print(f"\n{'=' * 40}\nExecuting FOLD {fold + 1}/{k_splits}\n{'=' * 40}")

        # Subsetting the dataframe correctly
        train_sub_df = df.iloc[train_ids].reset_index(drop=True)
        val_sub_df = df.iloc[val_ids].reset_index(drop=True)

        # Datasets
        train_dataset = ClothingDataset(
            train_sub_df, img_dir, transform=train_transform
        )
        val_dataset = ClothingDataset(val_sub_df, img_dir, transform=val_transform)

        # DataLoaders safely allocating workers for local pipeline
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Spin up fresh structural weights for strict isolation
        model = CustomCNN(num_classes=num_classes).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        best_fold_acc = 0.0

        for epoch in range(epochs_per_fold):
            # TRAIN PHASE
            model.train()
            train_corrects = 0
            total_train = 0
            for inputs, targets in tqdm(
                train_loader, desc=f"Fold {fold + 1} Epoch {epoch + 1} [TRAIN]"
            ):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                train_corrects += torch.sum(preds == targets.data)
                total_train += inputs.size(0)

            # VAL PHASE
            model.eval()
            val_corrects = 0
            total_val = 0
            with torch.no_grad():
                for inputs, targets in tqdm(
                    val_loader, desc=f"Fold {fold + 1} Epoch {epoch + 1} [VAL]"
                ):
                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == targets.data)
                    total_val += inputs.size(0)

            epoch_val_acc = (val_corrects.double() / total_val).item()

            if epoch_val_acc > best_fold_acc:
                best_fold_acc = epoch_val_acc

        print(f"-> Optimal Capability Fold {fold + 1}: {best_fold_acc:.4f}")
        fold_results[f"Fold_{fold + 1}"] = best_fold_acc

    # Complete Evaluation Metric Extraction
    print("\n" + "=" * 40)
    print("CROSS-VALIDATION STRUCTURAL SUMMARY:")
    print("=" * 40)
    avg_acc = np.mean(list(fold_results.values()))
    for fold_num, acc in fold_results.items():
        print(f"{fold_num}: {acc * 100:.2f}% Generalization Accuracy")
    print(f"\n=> Unified Architecture Mean Accuracy: {avg_acc * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    # We restrict epochs per fold to 2 by default to allow computation on weaker bounds
    # Swap to 10 in a heavy GPU cluster environment.
    perform_kfold_cv(
        CSV_FILE, IMAGES_DIR, BATCH_SIZE, IMAGE_SIZE, k_splits=5, epochs_per_fold=2
    )
