import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ========================
# ADVANCED CONFIG (RTX 3050 4GB OPTIMIZED)
# ========================
DATASET_PATH = r"C:\Users\mysel\Downloads\images_original"
NUM_EPOCHS = 10
NUM_RUNS = 10
BATCH_SIZE = 48  # Maximizing throughput for 4GB VRAM
LR = 3e-4
ACCUMULATION_STEPS = 1
NUM_WORKERS = 8
SAVE_DIR = "runs"

# GPU Optimizations
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# REPRODUCIBILITY
# ========================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic benchmark if reproducibility is priority (though benchmark=True is faster)
    # torch.backends.cudnn.deterministic = True


# ========================
# DATA AUGMENTATION (STRONG)
# ========================
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ========================
# DATASET PREPARATION (deferred until runtime)
# ========================
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def prepare_data(
    dataset_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, val_split=0.15
):
    full_dataset = datasets.ImageFolder(dataset_path)
    num_classes = len(full_dataset.classes)
    print(f"Detected {num_classes} classes: {full_dataset.classes}")

    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = None

    train_loader = DataLoader(
        TransformDataset(train_dataset, train_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        TransformDataset(val_dataset, val_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader, num_classes


# ========================
# MODEL (EFFICIENTNET WITH DROPOUT)
# ========================
def get_model():
    model = models.efficientnet_b0(weights="DEFAULT")
    in_features = model.classifier[1].in_features
    # Add Dropout for better regularization
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(in_features, NUM_CLASSES)
    )
    return model


# ========================
# TRAIN FUNCTION
# ========================
def train_one_run(run_id):
    print(f"\n[STARTING RUN {run_id + 1}/{NUM_RUNS}]")
    set_seed(42 + run_id)

    model = get_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0

    # ----------------------------
    # PROGRESSIVE UNFREEZING
    # ----------------------------
    # Initially freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    print("Pre-training classifier for 2 epochs...")

    # prepare per-run CSV log
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, f"run_{run_id + 1:02d}_log.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_acc", "val_acc", "train_loss"])

    for epoch in range(NUM_EPOCHS):
        # Unfreeze after 2 epochs
        if epoch == 2:
            print("Unfreezing backbone for full fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True

        model.train()
        correct, total, train_loss = 0, 0, 0

        # tqdm can also have encoding issues, let's keep it simple or set ascii=True
        pbar = tqdm(
            train_loader, desc=f"Run {run_id + 1} Epoch {epoch + 1}", ascii=True
        )
        for step, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # gradient accumulation support
            loss_value = loss.item()
            loss = loss / ACCUMULATION_STEPS
            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss_value
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            pbar.set_postfix(
                {"Loss": f"{loss_value:.4f}", "Acc": f"{100 * correct / total:.2f}%"}
            )

        train_acc = 100 * correct / total

        # VALIDATION
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        val_acc = 100 * correct / total
        scheduler.step()

        print(f"Epoch {epoch + 1}: Train {train_acc:.2f}% | Val {val_acc:.2f}%")

        # append CSV
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [epoch + 1, f"{train_acc:.4f}", f"{val_acc:.4f}", f"{train_loss:.6f}"]
            )

        # SAVE BEST MODEL PERFORMANCE
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, f"best_model_run_{run_id}.pth"),
            )

    return best_acc


# ========================
# MAIN LOOP
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-run training pipeline")
    parser.add_argument(
        "--dataset",
        "--data",
        dest="dataset",
        default=DATASET_PATH,
        help="Path to dataset root",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--accum-steps", dest="accum_steps", type=int, default=ACCUMULATION_STEPS
    )
    parser.add_argument(
        "--num-workers", dest="num_workers", type=int, default=NUM_WORKERS
    )
    parser.add_argument("--num-epochs", dest="num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--num-runs", dest="num_runs", type=int, default=NUM_RUNS)
    parser.add_argument("--lr", dest="lr", type=float, default=LR)
    parser.add_argument("--save-dir", dest="save_dir", default=SAVE_DIR)
    parser.add_argument("--val-split", dest="val_split", type=float, default=0.15)
    parser.add_argument(
        "--disable-cudnn-benchmark", dest="disable_benchmark", action="store_true"
    )

    args = parser.parse_args()

    # apply CLI overrides
    if args.disable_benchmark:
        torch.backends.cudnn.benchmark = False

    ACCUMULATION_STEPS = args.accum_steps
    NUM_WORKERS = args.num_workers
    NUM_EPOCHS = args.num_epochs
    NUM_RUNS = args.num_runs
    LR = args.lr
    SAVE_DIR = args.save_dir

    # prepare data loaders now that args are known
    train_loader, val_loader, detected_classes = prepare_data(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        val_split=args.val_split,
    )

    # set global NUM_CLASSES used by get_model()
    NUM_CLASSES = detected_classes

    all_results = []
    start_time = time.time()

    for run in range(NUM_RUNS):
        acc = train_one_run(run)
        all_results.append(acc)

    end_time = time.time()
    total_duration = (end_time - start_time) / 60

    # ========================
    # FINAL REPORT
    # ========================
    print("\n[FINAL PRODUCTION REPORT]")
    print(f"Total Time: {total_duration:.2f} mins")
    print("All Runs:", [f"{a:.2f}%" for a in all_results])
    print("Best Accuracy:", f"{max(all_results):.2f}%")
    print("Average Accuracy:", f"{sum(all_results) / len(all_results):.2f}%")
