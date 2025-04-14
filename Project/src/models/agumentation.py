import os
import cv2
import numpy as np
import random
from glob import glob
from tqdm import tqdm

# Paths
input_dir = r"D:\Final Year\Mega project\Project\data\processed\stare"
output_dir = r"D:\Final Year\Mega project\Project\data\processed\stare_aug"
os.makedirs(output_dir, exist_ok=True)

# Augmentation functions
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

def adjust_brightness(image, factor):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def add_gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Load images
image_paths = glob(os.path.join(input_dir, "*.png"))
print(f"Found {len(image_paths)} images")

# Perform augmentation
for img_path in tqdm(image_paths, desc="Augmenting images"):
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    
    # Original image copy
    cv2.imwrite(os.path.join(output_dir, filename), img)
    
    # Rotations
    for angle in [90, 180, 270]:
        rotated = rotate_image(img, angle)
        cv2.imwrite(os.path.join(output_dir, f"{name}_rot{angle}{ext}"), rotated)
    
    # Flipping
    flipped_h = flip_image(img, 1)
    flipped_v = flip_image(img, 0)
    cv2.imwrite(os.path.join(output_dir, f"{name}_flipH{ext}"), flipped_h)
    cv2.imwrite(os.path.join(output_dir, f"{name}_flipV{ext}"), flipped_v)
    
    # Brightness adjustments
    for factor in [0.7, 1.3]:
        bright_img = adjust_brightness(img, factor)
        cv2.imwrite(os.path.join(output_dir, f"{name}_bright{factor}{ext}"), bright_img)
    
    # Gaussian blur
    blurred = add_gaussian_blur(img, 5)
    cv2.imwrite(os.path.join(output_dir, f"{name}_blur{ext}"), blurred)

print("Data augmentation complete. Augmented images saved in stare_aug.")
