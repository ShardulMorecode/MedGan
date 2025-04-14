import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from generator import Generator

# Paths
MODEL_PATH = "D:/Final Year/Mega project/Project/saved_models/generator_v11.pth"  # Ensure this is the latest version
TEST_IMAGE_PATH = "D:/Final Year/Mega project/Project/data/test/download.jpeg"  
OUTPUT_PATH = "D:/Final Year/Mega project/Project/results/enhanced_sample.jpg"

# Load Trained Generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))  
generator.eval()  # ✅ Set to evaluation mode

# Load and Preprocess Image
def load_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ Error: Test image not found at {img_path}")

    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"❌ Error: OpenCV failed to read image. Check file format: {img_path}")

    image = cv2.resize(image, (256, 256))  
    image = image / 255.0  # Normalize
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  
    return image

# Enhance Image
def enhance_image(image_tensor):
    with torch.no_grad():
        enhanced_image = generator(image_tensor)
    return enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  

# Load Image
try:
    original = load_image(TEST_IMAGE_PATH)

    # Generate Enhanced Image
    enhanced = enhance_image(original)

    # ✅ Convert to 0-255 scale, ensuring values are properly clamped
    original = (original.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    enhanced = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)

    # ✅ Apply a sharpening filter to the enhanced image (optional)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)

    # ✅ Save Enhanced Image
    cv2.imwrite(OUTPUT_PATH, enhanced)

    # ✅ Display Before-After
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.title("Enhanced Image")
    plt.axis("off")

    plt.show()

    print(f"✅ Enhanced image saved at: {OUTPUT_PATH}")

except Exception as e:
    print(f"❌ Error: {e}")
