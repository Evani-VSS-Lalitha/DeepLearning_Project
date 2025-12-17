import os
import cv2
import numpy as np
from skimage.restoration import wiener

# === Parameters ===
INPUT_DIR = "Data/train"  # Folder with original images
OUTPUT_DIR = "Preprocess_outputs\preprocess_img"
CATEGORIES = ['MEN_Coats', 'MEN_Hood', 'MEN_Suits', 'WOMEN_Dress', 'WOMEN_Hood']

# === Functions ===

def apply_brightness(img):
    return cv2.convertScaleAbs(img, alpha=1.2, beta=30)

def apply_flip(img):
    return cv2.flip(img, 1)  # Horizontal flip

def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def apply_wiener_filter(img):
    # Convert to float32 for processing
    img_float = img.astype(np.float32) / 255.0
    
    # Process each channel separately
    channels = []
    for i in range(3):  # BGR channels
        channel = img_float[:, :, i]
        
        # Define a simple PSF (Point Spread Function)
        psf = np.ones((5, 5)) / 25
        
        # Apply Wiener filter
        filtered = wiener(channel, psf, balance=0.1)
        
        # Clip values to valid range
        filtered = np.clip(filtered, 0, 1)
        channels.append(filtered)
    
    # Combine channels and convert back to uint8
    filtered_img = np.stack(channels, axis=-1)
    filtered_img = (filtered_img * 255).astype(np.uint8)
    
    return filtered_img

def save_image(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)

# === Main Loop ===

for category in CATEGORIES:
    input_path = os.path.join(INPUT_DIR, category)
    output_path = os.path.join(OUTPUT_DIR, f"{category}_processed")

    for file in os.listdir(input_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_path, file)
            img = cv2.imread(img_path)
            base_name = os.path.splitext(file)[0]

            save_image(os.path.join(output_path, f"{base_name}_brightened.jpg"), apply_brightness(img))
            save_image(os.path.join(output_path, f"{base_name}_flipped.jpg"), apply_flip(img))
            save_image(os.path.join(output_path, f"{base_name}_gaussian.jpg"), apply_gaussian_blur(img))
            save_image(os.path.join(output_path, f"{base_name}_wiener.jpg"), apply_wiener_filter(img))

print("Preprocessing completed for all categories.")