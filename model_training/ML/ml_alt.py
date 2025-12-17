import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# === Base Directory ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Paths ===
FEATURES_PATH = os.path.join(BASE_DIR, "fe_outputs", "csv", "image_features.npy")
LABELS_PATH = os.path.join(BASE_DIR, "fe_outputs", "csv", "image_labels.npy")

# === Load Data ===
features = np.load(FEATURES_PATH, allow_pickle=True)
labels = np.load(LABELS_PATH, allow_pickle=True)

print(f" Loaded features: {features.shape}")
print(f" Loaded labels: {len(labels)}")
print(" Categories available:", np.unique(labels))

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

# === Class Weights (for imbalance) ===
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_encoded),
    y=labels_encoded
)
class_weights_dict = dict(zip(np.unique(labels_encoded), class_weights))

# === Define Models ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, multi_class="multinomial"),
    "SVM (Linear)": SVC(kernel="linear", class_weight="balanced"),
    "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced"),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "XGBoost": XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(labels_encoded)),
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

# === Train & Evaluate Models ===
results = {}

for name, model in models.items():
    print(f"\n Training {name} ...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f" {name} Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save Best Model ===
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

joblib.dump(best_model, os.path.join(BASE_DIR, "ml_best_model.pkl"))
joblib.dump(le, os.path.join(BASE_DIR, "label_encoder.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

print(f"\n Best Model: {best_model_name} with Accuracy {results[best_model_name]:.4f}")
print(" Model, Encoder, Scaler saved successfully!")
