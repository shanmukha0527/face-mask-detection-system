# ─────────────────────────────────────────────
#  Face Mask Detector — Model Training
#  Model : MobileNetV2 (transfer learning)
#  Author: Your Name
# ─────────────────────────────────────────────

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ── Config ────────────────────────────────────
DATASET_DIR = "../DATA SET"
MODEL_PATH = "../MOdel/mask_detector.h5"
PLOT_PATH = "../MOdel/training_plot.png"

INIT_LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

CLASSES = ["with_mask", "with_out mask"]


# ── 1. Load images ────────────────────────────
print("[INFO] Loading images …")
data, labels = [], []

for class_name in CLASSES:
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(class_dir):
        print(f"  [WARN] Folder not found: {class_dir} — skipping")
        continue
    print(f"  Loading from: {class_dir}")
    for img_path in paths.list_images(class_dir):
        img = load_img(img_path, target_size=IMAGE_SIZE)
        img = img_to_array(img)
        img = preprocess_input(img)
        data.append(img)
        labels.append(class_name)

data = np.array(data, dtype="float32")
labels = np.array(labels)
print(f"  Loaded {len(data)} images across {len(CLASSES)} classes")


# ── 2. Encode labels ──────────────────────────
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)  # one-hot


# ── 3. Train / test split ─────────────────────
(X_train, X_test, y_train, y_test) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)


# ── 4. Data augmentation ──────────────────────
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)


# ── 5. Build model (MobileNetV2 + custom head) ─
print("[INFO] Building model …")
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Custom classification head
head = base_model.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten(name="flatten")(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(len(CLASSES), activation="softmax")(head)

model = Model(inputs=base_model.input, outputs=head)


# ── 6. Compile ────────────────────────────────
print("[INFO] Compiling model …")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# ── 7. Train ──────────────────────────────────
print("[INFO] Training model …")
H = model.fit(
    aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // BATCH_SIZE,
    epochs=EPOCHS,
)


# ── 8. Evaluate ───────────────────────────────
print("[INFO] Evaluating model …")
pred_idxs = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE), axis=1)
print(classification_report(y_test.argmax(axis=1), pred_idxs, target_names=lb.classes_))

# ── 9. Save model ─────────────────────────────
os.makedirs("../MOdel", exist_ok=True)
model.save(MODEL_PATH)
print(f"[INFO] Model saved → {MODEL_PATH}")

# ── 10. Plot training curves ──────────────────
plt.style.use("ggplot")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(H.history["loss"], label="train loss")
axes[0].plot(H.history["val_loss"], label="val loss")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(H.history["accuracy"], label="train acc")
axes[1].plot(H.history["val_accuracy"], label="val acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"[INFO] Training plot saved → {PLOT_PATH}")
plt.show()
