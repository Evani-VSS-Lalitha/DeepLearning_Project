import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# === Base Directory (Corrected) ===
# This resolves to the main FASHION_FORESIGHT folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# === Paths to Features ===
FEATURES_PATH = os.path.join(BASE_DIR, "fe_outputs", "csv", "image_features.npy")
LABELS_PATH = os.path.join(BASE_DIR, "fe_outputs", "csv", "image_labels.npy")
NAMES_PATH = os.path.join(BASE_DIR, "fe_outputs", "csv", "image_names.npy")

print("Using BASE_DIR:", BASE_DIR)
print("Looking for features at:", FEATURES_PATH)

# === Load Data ===
features = np.load(FEATURES_PATH, allow_pickle=True)
labels = np.load(LABELS_PATH, allow_pickle=True)
names = np.load(NAMES_PATH, allow_pickle=True)

print(f"\n Loaded features: {features.shape}")
print(f" Loaded labels: {len(labels)}")
print("Categories available:", np.unique(labels))

# === Encode Labels ===
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# === Feature Scaling ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Handle Class Imbalance ===
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_encoded),
    y=labels_encoded
)
class_weights = dict(zip(np.unique(labels_encoded), class_weights))

# === Train Model ===
clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight=class_weights,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# === Predictions & Evaluation ===
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print("\n Training Performance")
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))

print("\n Testing Performance")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

print("\nClassification Report (Test Data):\n", classification_report(y_test, y_test_pred, target_names=le.classes_))
print("\nConfusion Matrix (Test Data):\n", confusion_matrix(y_test, y_test_pred))

# === Save Model Artifacts ===
joblib.dump(clf, os.path.join(BASE_DIR, "rf_fashion_model.pkl"))
joblib.dump(le, os.path.join(BASE_DIR, "label_encoder.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

print("\n Model, Label Encoder & Scaler saved successfully!")
