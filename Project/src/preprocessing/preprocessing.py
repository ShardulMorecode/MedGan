import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Define paths
RAW_DATA_PATH = "D:/Final Year/Mega project/Project/data/raw"
PROCESSED_DATA_PATH = "D:/Final Year/Mega project/Project/data/processed"

STARE_RAW = os.path.join(RAW_DATA_PATH, "stare")
STARE_PROCESSED = os.path.join(PROCESSED_DATA_PATH, "stare")

MRI_RAW = os.path.join(RAW_DATA_PATH, "brain_mri")
MRI_PROCESSED = os.path.join(PROCESSED_DATA_PATH, "brain_mri")

IMG_SIZE = (256, 256)  # Standard image size


def preprocess_stare():
    """Convert .ppm to .png and resize."""
    os.makedirs(STARE_PROCESSED, exist_ok=True)
    for file in tqdm(os.listdir(STARE_RAW), desc="Processing STARE Dataset"):
        if file.endswith(".ppm"):
            img_path = os.path.join(STARE_RAW, file)
            img = Image.open(img_path)
            img = img.resize(IMG_SIZE)
            save_path = os.path.join(STARE_PROCESSED, file.replace(".ppm", ".png"))
            img.save(save_path)
    print("✅ STARE dataset preprocessing complete.")


def preprocess_mri():
    """Resize MRI images and normalize pixel values."""
    for split in ["train", "test"]:
        for category in ["glioma", "meningioma", "notumor", "pituitary"]:
            input_path = os.path.join(MRI_RAW, split, category)
            output_path = os.path.join(MRI_PROCESSED, split, category)
            os.makedirs(output_path, exist_ok=True)

            for file in tqdm(os.listdir(input_path), desc=f"Processing {split}/{category}"):
                if file.endswith(".jpg") or file.endswith(".png"):
                    img_path = os.path.join(input_path, file)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read as color image
                    img = cv2.resize(img, IMG_SIZE)  # Resize to standard size
                    img = img / 255.0  # Normalize pixel values (0-1)
                    save_path = os.path.join(output_path, file)
                    cv2.imwrite(save_path, (img * 255).astype(np.uint8))  # Save as uint8
    print("✅ MRI dataset preprocessing complete.")


if __name__ == "__main__":
    preprocess_stare()
    preprocess_mri()
