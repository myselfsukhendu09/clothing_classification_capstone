import os
import shutil

import pandas as pd
from tqdm import tqdm

# CONFIG
DATASET_PATH = r"C:\Users\mysel\Downloads\images_original"
CSV_PATH = r"C:\Users\mysel\Downloads\images.csv"


def restructure_dataset():
    print(f"Checking dataset structure at {DATASET_PATH}...")

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["label"])

    # Create class directories
    classes = df["label"].unique()
    for cls in classes:
        cls_dir = os.path.join(DATASET_PATH, str(cls))
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
            print(f"Created directory: {cls_dir}")

    # Move files
    print("Moving files to class directories...")
    moved_count = 0
    skipped_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row["image"] + ".jpg"
        label = str(row["label"])

        src_path = os.path.join(DATASET_PATH, img_name)
        dst_path = os.path.join(DATASET_PATH, label, img_name)

        if os.path.exists(src_path):
            # Check if it's already in a subdirectory (though our check said it's flat)
            shutil.move(src_path, dst_path)
            moved_count += 1
        elif os.path.exists(dst_path):
            skipped_count += 1
        else:
            # Maybe it's in another folder or missing
            pass

    print(f"Done! Moved: {moved_count}, Skipped (already moved): {skipped_count}")


if __name__ == "__main__":
    restructure_dataset()
