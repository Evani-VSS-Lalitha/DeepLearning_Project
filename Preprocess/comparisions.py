import os
import cv2
import numpy as np
from skimage.restoration import wiener
import matplotlib.pyplot as plt

# === Parameters ===
INPUT_DIR = "Data/train"
OUTPUT_DIR = "Preprocess_outputs"  # New output directory
CATEGORIES = ['MEN_Hood', 'WOMEN_Dress', 'MEN_Coats']
SAMPLE_IMAGES = 1  # Number of sample images per category

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Updated Wiener Filter Function ===
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

# === Other Processing Functions ===
def apply_brightness(img):
    return cv2.convertScaleAbs(img, alpha=1.2, beta=30)

def apply_flip(img):
    return cv2.flip(img, 1)  # Horizontal flip

def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

# === Function to create comparison grid ===
def create_comparison_grid(original, processed_images, titles):
    plt.figure(figsize=(15, 8))
    
    # Convert BGR to RGB for matplotlib display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis('off')
    
    # Processed images
    for i, (img, title) in enumerate(zip(processed_images, titles)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 3, i+2)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    return plt

# === Main Processing ===
for category in CATEGORIES:
    input_path = os.path.join(INPUT_DIR, category)
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process only the first image for each category
    if image_files:
        img_path = os.path.join(input_path, image_files[0])
        img = cv2.imread(img_path)
        
        # Apply all processing techniques
        flipped = apply_flip(img)
        brightened = apply_brightness(img)
        blurred = apply_gaussian_blur(img)
        wiener_filtered = apply_wiener_filter(img)
        
        # Create comparison grid
        processed_images = [flipped, brightened, blurred, wiener_filtered]
        titles = ["Flipped", "Brightened", "Gaussian Blur", "Wiener Filter"]
        
        plt = create_comparison_grid(img, processed_images, titles)
        plt.suptitle(f"Preprocessing Comparison - {category}", y=1.02)
        
        # Save to Preprocess_outputs folder
        output_path = os.path.join(OUTPUT_DIR, f"{category}_preprocessing_comparison.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison for {category} at {output_path}")

print("Preprocessing comparison completed for all categories.")