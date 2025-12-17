import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gabor

# Set paths
image_dir = "Preprocess_outputs/preprocess_img"
output_dir = "fe_outputs/clothes"
os.makedirs(output_dir, exist_ok=True)

IMAGE_SIZE = (128, 128)

# Output files
feature_file = os.path.join(output_dir, "image_features.npy")
name_file = os.path.join(output_dir, "image_names.npy")

# Load existing data if available
if os.path.exists(feature_file) and os.path.exists(name_file):
    existing_features = list(np.load(feature_file, allow_pickle=True))
    existing_names = list(np.load(name_file, allow_pickle=True))
else:
    existing_features = []
    existing_names = []

# Feature extractors
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_canny_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.array([np.mean(edges), np.std(edges)])

def extract_harris_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    return np.array([np.mean(dst), np.std(dst)])

def extract_gabor_features(image):
    gray = rgb2gray(image)
    filt_real, filt_imag = gabor(gray, frequency=0.6)
    return np.array([np.mean(filt_real), np.std(filt_real)])

# Process images
new_features = []
new_names = []

for root, _, files in os.walk(image_dir):
    for img_name in files:
        if img_name in existing_names:
            print(f"Skipping already processed image: {img_name}")
            continue

        img_path = os.path.join(root, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping invalid image: {img_path}")
            continue

        try:
            image = cv2.resize(image, IMAGE_SIZE)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            print(f"Processing image: {img_name}")
            color_hist = extract_color_histogram(image)
            canny_feat = extract_canny_edges(image)
            harris_feat = extract_harris_corners(image)
            gabor_feat = extract_gabor_features(image_rgb)

            feature_vector = np.hstack((color_hist, canny_feat, harris_feat, gabor_feat))
            new_features.append(feature_vector)
            new_names.append(img_name)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

# Combine and save
all_features = np.array(existing_features + new_features)
all_names = np.array(existing_names + new_names)
np.save(feature_file, all_features)
np.save(name_file, all_names)

print("Feature extraction completed and saved.")

# Visualize on first valid new image
if new_names:
    sample_path = None
    for root, _, files in os.walk(image_dir):
        if new_names[0] in files:
            sample_path = os.path.join(root, new_names[0])
            break

    if sample_path:
        sample_image = cv2.imread(sample_path)
        sample_image = cv2.resize(sample_image, IMAGE_SIZE)
        sample_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        gabor_real, _ = gabor(rgb2gray(sample_rgb), frequency=0.6)

        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        axes[0].imshow(sample_rgb)
        axes[0].set_title("Original Image")
        axes[1].imshow(canny, cmap="gray")
        axes[1].set_title("Canny Edges")
        axes[2].imshow(dst, cmap="gray")
        axes[2].set_title("Harris Corners")
        axes[3].imshow(gabor_real, cmap="gray")
        axes[3].set_title("Gabor Filter")
        axes[4].plot(extract_color_histogram(sample_image))
        axes[4].set_title("Color Histogram")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_comparison.png"))
        plt.close()
        print("Feature comparison saved as feature_comparison.png")
else:
    print("No new images processed, skipping visualization.")
