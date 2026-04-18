import os

import torch

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global Hardware Controls & Device Overrides natively executing CUDA boundaries
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True
NUM_WORKERS = 4
PIN_MEMORY = True

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = DATA_DIR
CSV_FILE = os.path.join(DATA_DIR, "images.csv")

# Reports and logs
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Model checkpoints structure
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_resnet50.pth")
BEST_ANN_PATH = os.path.join(BASE_DIR, "best_ann.pth")
BEST_CUSTOM_CNN_PATH = os.path.join(BASE_DIR, "best_custom_cnn.pth")
CLASS_MAPPING_FILE = os.path.join(BASE_DIR, "class_mapping.json")

# Core Hyperparameters (Tuned natively against VRAM scale via iteration testing limits)
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
