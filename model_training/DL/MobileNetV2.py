import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix

# === Paths ===
# Go three levels up: /Fashion_Foresight/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(BASE_DIR, "Data", "train")
TEST_DIR = os.path.join(BASE_DIR, "Data", "test")

print("BASE_DIR:", BASE_DIR)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)

# === Parameters ===
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 5           # stage 1 (frozen base)
FINE_TUNE_EPOCHS = 10  # stage 2 (fine-tune last layers)
LR_FINE_TUNE = 1e-5

# === Data Generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training"
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation"
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

# === Load Pretrained MobileNetV2 ===
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(160, 160, 3))
base_model.trainable = False  # Stage 1: freeze base

# === Build Model ===
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("\n Stage 1: Training classifier head only...")
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# === Fine-Tuning Stage ===
print("\n Stage 2: Fine-tuning last layers...")
base_model.trainable = True
for layer in base_model.layers[:-30]:  # freeze all except last 30 layers
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINE_TUNE),
              loss="categorical_crossentropy", metrics=["accuracy"])
history_finetune = model.fit(train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS)

# === Evaluate on Test Set ===
test_loss, test_acc = model.evaluate(test_gen)
print(f"\n Fine-Tuned Test Accuracy: {test_acc:.4f}")

# === Predictions ===
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

print("\nClassification Report:\n", classification_report(y_true, y_pred_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.title("Confusion Matrix - Fine-Tuned MobileNetV2")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(BASE_DIR, "mobilenetv2_finetuned_confusion_matrix.png"))
plt.show()

# === Save Model ===
MODEL_PATH = os.path.join(BASE_DIR, "mobilenetv2_finetuned.keras")
model.save(MODEL_PATH)
print(f"\n Fine-tuned MobileNetV2 saved to {MODEL_PATH}")
