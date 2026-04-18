import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json

import argparse

# ======================== 
# CONFIGURATION
# ========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_FILE = os.path.join(DATA_DIR, "images.csv")
CLASS_MAPPING_FILE = os.path.join(PROJECT_ROOT, "class_mapping.json")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def build_keras_model(num_classes):
    """
    Builds a CNN model using Keras as requested.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Classification Head
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_keras(images_dir):
    if not os.path.exists(CSV_FILE):
        print(f"Error: Dataset not found at {CSV_FILE}")
        return

    print(f"Loading data with images from: {images_dir}")
    # Load data
    df = pd.read_csv(CSV_FILE)
    
    # Keras ImageDataGenerator expects string labels for flow_from_dataframe
    df['label'] = df['label'].astype(str)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    # Data Generators (Augmentation included)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=images_dir,
        x_col='image',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=images_dir,
        x_col='image',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Save class mapping if not already present
    class_indices = train_generator.class_indices
    # Invert mapping to match our project's format {idx: label}
    idx_to_label = {int(v): k for k, v in class_indices.items()}
    with open(CLASS_MAPPING_FILE, "w") as f:
        json.dump(idx_to_label, f)

    num_classes = len(class_indices)
    print(f"Detected {num_classes} classes.")

    # Build and Train
    model = build_keras_model(num_classes)
    model.summary()

    print("\nStarting Keras Training Session...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # Save the model
    save_path = os.path.join(PROJECT_ROOT, "best_keras_model.h5")
    model.save(save_path)
    print(f"\nModel saved successfully to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, default=DATA_DIR)
    args = parser.parse_args()
    
    train_keras(args.images_dir)
