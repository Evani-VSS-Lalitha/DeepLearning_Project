import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report, confusion_matrix

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(BASE_DIR, "Data", "train")
TEST_DIR = os.path.join(BASE_DIR, "Data", "test")

# === Parameters ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5            # Stage 1: frozen base
FINE_TUNE_EPOCHS = 10 # Stage 2: fine-tuning
LR_FINE_TUNE = 1e-5

# === Data Generators ===
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                   rotation_range=20, width_shift_range=0.1,
                                   height_shift_range=0.1, shear_range=0.1,
                                   zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE, class_mode="categorical", subset="training")
val_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                                            batch_size=BATCH_SIZE, class_mode="categorical", subset="validation")
test_gen = test_datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE,
                                            batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

# === Load Pretrained ResNet50 ===
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# === Build Model ===
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("\nStage 1: Training classifier head...")
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# === Fine-Tuning ===
print("\nStage 2: Fine-tuning last layers...")
base_model.trainable = True
for layer in base_model.layers[:-50]:  # freeze all except last 50 layers
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINE_TUNE),
              loss="categorical_crossentropy", metrics=["accuracy"])
history_finetune = model.fit(train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS)

# === Evaluate ===
test_loss, test_acc = model.evaluate(test_gen)
print(f"\nFine-Tuned ResNet50 Test Accuracy: {test_acc:.4f}")

# === Predictions ===
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

print("\nClassification Report:\n", classification_report(y_true, y_pred_classes, target_names=class_labels))
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.title("Confusion Matrix - Fine-Tuned ResNet50")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# === Save Model ===
MODEL_PATH = os.path.join(BASE_DIR, "resnet50_finetuned.keras")
model.save(MODEL_PATH)
print(f"\nFine-Tuned ResNet50 saved to {MODEL_PATH}")
