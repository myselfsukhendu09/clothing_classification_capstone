#!/usr/bin/env python3
"""
Dataset Download & Setup Script
================================
This script provides instructions and utilities for setting up the
clothing classification dataset used in this project.

The dataset contains ~5,763 images across 20 clothing categories.
Total size: ~6.4 GB

Usage:
    python setup_dataset.py
"""

import os
import json

# Expected class structure
CLASSES = [
    "Blazer", "Blouse", "Body", "Dress", "Hat",
    "Hoodie", "Longsleeve", "Not sure", "Other", "Outwear",
    "Pants", "Polo", "Shirt", "Shoes", "Shorts",
    "Skip", "Skirt", "T-Shirt", "Top", "Undershirt"
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def check_dataset():
    """Check if the dataset is properly set up."""
    if not os.path.exists(DATA_DIR):
        print("❌ Dataset directory 'data/' not found.")
        print()
        print("=" * 60)
        print("  DATASET SETUP INSTRUCTIONS")
        print("=" * 60)
        print()
        print("This project uses a clothing image dataset with 20 categories.")
        print(f"Total: ~5,763 images | Size: ~6.4 GB")
        print()
        print("Option 1: Manual Setup")
        print("-" * 40)
        print("1. Download clothing images from your preferred source")
        print("   (e.g., Kaggle, custom dataset, web scraping)")
        print("2. Organize images into class-based subfolders:")
        print()
        print("   data/")
        for cls in CLASSES[:5]:
            print(f"   ├── {cls}/")
            print(f"   │   ├── image_001.jpg")
            print(f"   │   ├── image_002.jpg")
            print(f"   │   └── ...")
        print(f"   ├── ...")
        for cls in CLASSES[-2:]:
            print(f"   ├── {cls}/")
            print(f"   │   └── ...")
        print()
        print("Option 2: Use restructure_dataset.py")
        print("-" * 40)
        print("If you have a flat directory of images with a CSV mapping,")
        print("use the provided script to auto-organize:")
        print()
        print("   python restructure_dataset.py")
        print()
        print("=" * 60)
        return False

    # Check class folders
    found_classes = []
    total_images = 0

    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.exists(cls_dir):
            n_images = len([f for f in os.listdir(cls_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            found_classes.append(cls)
            total_images += n_images

    print("=" * 60)
    print("  DATASET STATUS REPORT")
    print("=" * 60)
    print(f"  Classes found: {len(found_classes)}/{len(CLASSES)}")
    print(f"  Total images:  {total_images:,}")
    print()

    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.exists(cls_dir):
            n = len([f for f in os.listdir(cls_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            status = "✅" if n > 0 else "⚠️"
            print(f"  {status} {cls:20s} → {n:5d} images")
        else:
            print(f"  ❌ {cls:20s} → MISSING")

    print("=" * 60)

    if len(found_classes) == len(CLASSES) and total_images > 0:
        print("✅ Dataset is properly configured!")
        return True
    else:
        missing = set(CLASSES) - set(found_classes)
        print(f"⚠️  Missing classes: {', '.join(missing)}")
        return False


if __name__ == "__main__":
    check_dataset()
