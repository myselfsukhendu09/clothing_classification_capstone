import argparse
import json
import os

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from checkpoint_compat import load_checkpoint_strict
from config import (
    BATCH_SIZE,
    CLASS_MAPPING_FILE,
    CSV_FILE,
    IMAGE_SIZE,
    IMAGES_DIR,
    MODEL_SAVE_PATH,
    REPORTS_DIR,
    SEED,
)
from dataset import get_dataloaders
from model import get_model
from utils import ensure_dir, set_seed


def evaluate(model_path: str, reports_dir: str) -> int:
    set_seed(SEED)
    ensure_dir(reports_dir)

    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found at {CSV_FILE}")
        return 2
    if not os.path.isdir(IMAGES_DIR):
        print(f"Error: image directory not found at {IMAGES_DIR}")
        return 2
    if not os.path.exists(CLASS_MAPPING_FILE):
        print(
            f"Error: class mapping not found at {CLASS_MAPPING_FILE}. Run training first."
        )
        return 2
    if not os.path.exists(model_path):
        print(f"Error: model checkpoint not found at {model_path}. Run training first.")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, idx_to_label = get_dataloaders(
        CSV_FILE, IMAGES_DIR, BATCH_SIZE, IMAGE_SIZE
    )
    num_classes = len(idx_to_label)

    model = get_model("resnet50", num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    load_checkpoint_strict(model, checkpoint, source=model_path)
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    labels_sorted = list(range(num_classes))
    target_names = [idx_to_label[i] for i in labels_sorted]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_sorted,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

    report_path = os.path.join(reports_dir, "test_classification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm_path = os.path.join(reports_dir, "test_confusion_matrix.csv")
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(cm_path, index=True)

    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Test accuracy: {report.get('accuracy', 0.0):.4f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the held-out test set."
    )
    parser.add_argument("--model-path", type=str, default=MODEL_SAVE_PATH)
    parser.add_argument("--reports-dir", type=str, default=REPORTS_DIR)
    args = parser.parse_args()
    raise SystemExit(evaluate(args.model_path, args.reports_dir))
