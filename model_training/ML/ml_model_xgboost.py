import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# === Base Directory ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# === Paths ===
FEATURES_PATH = os.path.join(BASE_DIR, "fe_outputs", "csv", "image_features.npy")
LABELS_PATH = os.path.join(BASE_DIR, "fe_outputs", "csv", "image_labels.npy")
NAMES_PATH = os.path.join(BASE_DIR, "fe_outputs", "csv", "image_names.npy")

print("Loading data from:", FEATURES_PATH)

# === Load Data ===
features = np.load(FEATURES_PATH, allow_pickle=True)
labels = np.load(LABELS_PATH, allow_pickle=True)

# === Encode Labels ===
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# === Scaling ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Compute Class Weights ===
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_encoded),
    y=labels_encoded
)
class_weights = dict(zip(np.unique(labels_encoded), class_weights))

# === Train XGBoost Model ===
model = XGBClassifier(
    n_estimators=600,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=len(np.unique(labels_encoded)),
    eval_metric="mlogloss",
    tree_method="hist",  # fast GPU/CPU training
    scale_pos_weight=None
)

model.fit(X_train, y_train)

# === Predictions ===
y_train_pred = np.argmax(model.predict_proba(X_train), axis=1)
y_test_pred = np.argmax(model.predict_proba(X_test), axis=1)

# === Evaluation ===
print("\n Training Accuracy:", accuracy_score(y_train, y_train_pred))
print(" Testing Accuracy:", accuracy_score(y_test, y_test_pred))

print("\n Classification Report:\n", classification_report(y_test, y_test_pred, target_names=le.classes_))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# === Save Artifacts ===
joblib.dump(model, os.path.join(BASE_DIR, "xgb_fashion_model.pkl"))
joblib.dump(le, os.path.join(BASE_DIR, "label_encoder.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

print("\n XGBoost Model, Label Encoder & Scaler saved successfully!")
