import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import json

DATASET_PATH = r"C:\Users\mysel\Downloads\images_original"

def get_data_loaders(batch_size=24):
    """Refined CV Data Pipeline with Optimized Loaders."""
    
    # 1. Augmentation Strategy for Generalization
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset root {DATASET_PATH} not found.")
        
    full_dataset = datasets.ImageFolder(DATASET_PATH)
    
    # Persistent class mapping for dashboard integration
    class_mapping = {str(i): cls for i, cls in enumerate(full_dataset.classes)}
    with open("class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Generated: class_mapping.json for {len(full_dataset.classes)} categories.")

    # 2. Split (80/20)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Overwrite transforms on sub-datasets
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = val_tf

    # 3. Optimized DataLoaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, # Slightly safer for universal dev boxes than 6
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, len(full_dataset.classes)
