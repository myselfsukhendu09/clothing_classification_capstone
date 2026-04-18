import torch
import os
import tqdm
from src.core.models import get_model
from src.core.utils import get_data_loaders

# Set GPU compute if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- ACTIVE DEVICE: {DEVICE} ---")

# Candidate architecture list
MODEL_LIST = ["resnet50", "efficientnet", "mobilenet"]

def train_model(model_name, train_loader, val_loader, num_classes, epochs=10):
    """Refined Model Optimization Sequence."""
    print(f"\n--- Initiating Optimization for {model_name.upper()} ---")
    model = get_model(model_name, num_classes).to(DEVICE)

    # Industry Standard AdamW for Better Generalization
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0

    for epoch in range(epochs):
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}")
        
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())

        # VALIDATION PHASE
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                _, pred = outputs.max(1)

                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        acc = 100 * correct / total
        print(f" -> Validation Accuracy Metrics: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), f"weights/{model_name}.pth")
            print(f" -> Checkpoint Cached: weights/{model_name}.pth")

    return best_acc

def train_all():
    """Master Training Orchestrator."""
    os.makedirs("weights", exist_ok=True)
    train_loader, val_loader, num_classes = get_data_loaders()

    results = {}

    for model_name in MODEL_LIST:
        # Verified sequence: Epoch set for architecture synchronization
        acc = train_model(model_name, train_loader, val_loader, num_classes, epochs=1)
        results[model_name] = acc

    return results

if __name__ == "__main__":
    train_all()
