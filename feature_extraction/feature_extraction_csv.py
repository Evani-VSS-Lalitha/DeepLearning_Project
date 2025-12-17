import os
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.filters import gabor

# === Paths ===
CSV_PATH = "Preprocess_outputs/Preprocess_clothing_dataset.csv"
IMAGE_DIR = "Preprocess_outputs/preprocess_img"
OUTPUT_DIR = "fe_outputs/csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output files
FEATURES_FILE = os.path.join(OUTPUT_DIR, "image_features.npy")
LABELS_FILE = os.path.join(OUTPUT_DIR, "image_labels.npy")
NAMES_FILE = os.path.join(OUTPUT_DIR, "image_names.npy")

# Load preprocessed CSV
df = pd.read_csv(CSV_PATH)
print(f" Loaded CSV with {len(df)} rows")
print(" Categories available:", df["Category"].unique())

# === Feature extractors ===
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
    filt_real, _ = gabor(gray, frequency=0.6)
    return np.array([np.mean(filt_real), np.std(filt_real)])

# === Process images based on CSV ===
features = []
labels = []
names = []

IMAGE_SIZE = (128, 128)

for _, row in df.iterrows():
    img_name = row["Image"]
    category = row["Category"]

    # Match original + augmented versions
    base_name = os.path.splitext(img_name)[0]
    for root, _, files in os.walk(IMAGE_DIR):
        for f in files:
            if f.startswith(base_name):  # original + augmented match
                img_path = os.path.join(root, f)
                image = cv2.imread(img_path)

                if image is None:
                    print(f" Skipping invalid image: {img_path}")
                    continue

                try:
                    image = cv2.resize(image, IMAGE_SIZE)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Extract features
                    color_hist = extract_color_histogram(image)
                    canny_feat = extract_canny_edges(image)
                    harris_feat = extract_harris_corners(image)
                    gabor_feat = extract_gabor_features(image_rgb)

                    feature_vector = np.hstack((color_hist, canny_feat, harris_feat, gabor_feat))

                    features.append(feature_vector)
                    labels.append(category)
                    names.append(f)

                except Exception as e:
                    print(f" Error processing {f}: {e}")
                    continue

print(f"\n Extracted features for {len(features)} images")

# Save outputs
np.save(FEATURES_FILE, np.array(features))
np.save(LABELS_FILE, np.array(labels))
np.save(NAMES_FILE, np.array(names))

print(f" Features saved to {FEATURES_FILE}")
print(f" Labels saved to {LABELS_FILE}")
print(f" Names saved to {NAMES_FILE}")
