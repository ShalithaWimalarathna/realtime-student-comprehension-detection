"""
KIU Comprehension System — Model Trainer
=========================================
Trains a new emotion recognition model on combined datasets
(FER-2013 + RAF-DB + any Asian dataset you add).

Usage:
    python train_model.py

Output:
    models/resnet50_fer2013.h5   (replaces old model)
    models/training_history.png  (accuracy/loss chart)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Add, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import cv2

# ─────────────────────────────────────────────
#  CONFIGURATION  — edit paths here if needed
# ─────────────────────────────────────────────
TRAIN_DIR     = "datasets/train"   # subfolders = emotion names
TEST_DIR      = "datasets/test"
MODEL_OUT     = "models/resnet50_fer2013.h5"
IMG_SIZE      = 64                 # must match CONFIG in main.py
BATCH_SIZE    = 64
EPOCHS        = 50                 # early stopping will stop sooner if needed
LEARNING_RATE = 0.001

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# ─────────────────────────────────────────────
#  DATA GENERATORS  (with heavy augmentation)
# ─────────────────────────────────────────────
def make_generators():
    # Augmentation helps a lot when training on mixed ethnic datasets
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],   # helps with different lighting
        fill_mode="nearest",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTION_LABELS,
        shuffle=True,
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTION_LABELS,
        shuffle=False,
    )

    return train_gen, test_gen


# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE  (ResNet-style mini CNN)
#  Efficient enough to train on a laptop CPU/GPU
#  but accurate enough for real-time use
# ─────────────────────────────────────────────
def residual_block(x, filters, stride=1):
    shortcut = x

    x = Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    # Match dimensions for shortcut if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def build_model(num_classes=7):
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Stem
    x = Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)

    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    # Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model


# ─────────────────────────────────────────────
#  CLASS WEIGHT CALCULATOR
#  Handles imbalanced datasets (FER-2013 has
#  far more Happy images than Disgust)
# ─────────────────────────────────────────────
def get_class_weights(train_gen):
    labels = train_gen.classes
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    return dict(enumerate(weights))


# ─────────────────────────────────────────────
#  PLOT TRAINING HISTORY
# ─────────────────────────────────────────────
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ax1.plot(history.history["accuracy"],     color="#22c55e", linewidth=2, label="Train")
    ax1.plot(history.history["val_accuracy"], color="#6366f1", linewidth=2, label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white")
    ax1.grid(alpha=0.2)

    ax2.plot(history.history["loss"],     color="#ef4444", linewidth=2, label="Train")
    ax2.plot(history.history["val_loss"], color="#f59e0b", linewidth=2, label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    out_path = "models/training_history.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[INFO] Training chart saved → {out_path}")
    plt.close()


# ─────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def train():
    print("\n" + "=" * 60)
    print("  KIU Emotion Model Trainer")
    print("=" * 60)

    # Validate dataset folders exist
    for folder in [TRAIN_DIR, TEST_DIR]:
        if not os.path.exists(folder):
            print(f"\n[ERROR] Folder not found: {folder}")
            print("  Make sure you have downloaded and placed the datasets.")
            print("  See README.md for folder structure.")
            return

    os.makedirs("models", exist_ok=True)

    print("\n[Step 1] Loading datasets...")
    train_gen, test_gen = make_generators()

    total_train = train_gen.samples
    total_test  = test_gen.samples
    print(f"  Training images : {total_train:,}")
    print(f"  Test images     : {total_test:,}")
    print(f"  Classes found   : {list(train_gen.class_indices.keys())}")

    # Warn if class mapping doesn't match expected
    found = sorted(train_gen.class_indices.keys())
    expected = sorted(EMOTION_LABELS)
    if found != expected:
        print(f"\n[WARN] Expected folders: {expected}")
        print(f"       Found folders   : {found}")
        print("       Make sure folder names are lowercase.")

    print("\n[Step 2] Computing class weights...")
    class_weights = get_class_weights(train_gen)
    for i, (cls, w) in enumerate(class_weights.items()):
        label = EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else str(i)
        print(f"  {label:<10} weight = {w:.3f}")

    print("\n[Step 3] Building model...")
    model = build_model(num_classes=len(EMOTION_LABELS))
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n[Step 4] Setting up training callbacks...")
    callbacks = [
        ModelCheckpoint(
            MODEL_OUT,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        CSVLogger("models/training_log.csv"),
    ]

    print("\n[Step 5] Training...\n")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[Step 6] Evaluating on test set...")
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"\n  Final Test Accuracy : {acc * 100:.2f}%")
    print(f"  Final Test Loss     : {loss:.4f}")

    print("\n[Step 7] Saving training chart...")
    plot_history(history)

    print("\n" + "=" * 60)
    print(f"  Done! Model saved → {MODEL_OUT}")
    print(f"  Test accuracy     : {acc * 100:.2f}%")
    print("=" * 60 + "\n")

    if acc < 0.60:
        print("[TIP] Accuracy below 60% — try adding more Asian face images")
        print("      to datasets/train/ and running train_model.py again.\n")
    elif acc < 0.75:
        print("[TIP] Good start! Adding more diverse data will push this higher.\n")
    else:
        print("[TIP] Great accuracy! Run main.py to use your new model.\n")


if __name__ == "__main__":
    train()
