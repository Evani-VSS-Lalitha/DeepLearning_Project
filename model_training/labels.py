import os
import pandas as pd

# Paths
original_csv_path = "Data/unique_clothing_dataset.csv"
preprocessed_img_dir = "Preprocess_outputs/preprocess_img"
output_csv_path = "training_outputs/csv/labels.csv"

# Load CSV
df = pd.read_csv(original_csv_path)
print("CSV Columns found:", df.columns.tolist())

# Validate necessary columns
if 'Image' not in df.columns or 'Clothing Type' not in df.columns:
    raise KeyError("CSV must contain 'Image' and 'Clothing Type' columns")

# Create mapping: base image name ➜ label
name_to_label = {
    row['Image'].split('_')[0]: row['Clothing Type'] for _, row in df.iterrows()
}
print("Sample keys from CSV:", list(name_to_label.keys())[:5])

# Recursively list all image files
image_files = []
for root, _, files in os.walk(preprocessed_img_dir):
    for file in files:
        if file.endswith(".jpg"):
            image_files.append(os.path.join(root, file))

print(f"Total preprocessed images found: {len(image_files)}")
print("Sample image paths:", image_files[:3])

# Match images with labels
new_rows = []
for full_path in image_files:
    filename = os.path.basename(full_path)
    base_name = filename.split('_')[0]  # Extract '29058502' from '29058502_4pl_crop.jpg'
    label = name_to_label.get(base_name)
    if label:
        new_rows.append({'img_name': filename, 'label': label})
    else:
        print(f"[Skipping] No label found for: {filename}")

# Save matched entries
new_df = pd.DataFrame(new_rows)
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
new_df.to_csv(output_csv_path, index=False)

print(f"[] New CSV saved to: {output_csv_path} with {len(new_df)} entries")
print(f"[] Matched {len(new_rows)} out of {len(image_files)} images")
