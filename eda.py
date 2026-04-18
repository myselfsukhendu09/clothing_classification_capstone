import os
import random
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image

# Global Configuration
DATASET_PATH = r"C:\Users\mysel\Downloads\images_original"

def get_class_distribution(dataset_path):
    """Scan the dataset directory and count images per class folder."""
    class_counts = {}
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} not found.")
        return {}
        
    for cls in os.listdir(dataset_path):
        cls_path = os.path.join(dataset_path, cls)
        if os.path.isdir(cls_path):
            # Filtering for common image extensions
            images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            class_counts[cls] = len(images)
    return class_counts

def plot_class_distribution(class_counts):
    """Generate a bar chart of class distribution."""
    if not class_counts:
        return
        
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue', edgecolor='navy')
    plt.xticks(rotation=45, ha='right')
    plt.title("Clothing Class Distribution - Dataset Intelligence")
    plt.ylabel("Number of Samples")
    plt.xlabel("Category")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/class_distribution.png")
    print("Saved: outputs/class_distribution.png")
    plt.close()

def show_sample_images(dataset_path, num_samples=5):
    """Pick random images from random classes and display them."""
    classes = [c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))]
    if not classes:
        return
        
    plt.figure(figsize=(15, 7))
    
    for i in range(num_samples):
        cls = random.choice(classes)
        cls_dir = os.path.join(dataset_path, cls)
        all_imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not all_imgs:
            continue
            
        img_name = random.choice(all_imgs)
        img_path = os.path.join(cls_dir, img_name)

        try:
            img = Image.open(img_path)
            plt.subplot(1, num_samples, i+1)
            plt.imshow(img)
            plt.title(f"Class: {cls}")
            plt.axis("off")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/sample_images.png")
    print("Saved: outputs/sample_images.png")
    plt.close()

def run_eda():
    """Main EDA execution pipeline."""
    os.makedirs("outputs", exist_ok=True)

    print("--- Initiating Dataset Intelligence Sequence ---")
    class_counts = get_class_distribution(DATASET_PATH)
    
    if not class_counts:
        print("EDA Aborted: No data found.")
        return

    print(f"Detected Classes: {len(class_counts)}")
    for cls, count in class_counts.items():
        print(f" -> {cls}: {count}")

    plot_class_distribution(class_counts)
    show_sample_images(DATASET_PATH)

    # Detect imbalance metrics
    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)

    print(f"\nStatistical Summary:")
    print(f" - Max Samples: {max_count}")
    print(f" - Min Samples: {min_count}")
    print(f" - Imbalance Ratio: {max_count/min_count:.2f}")
    print("------------------------------------------------")

if __name__ == "__main__":
    run_eda()
