"""
03_train_cnn.py  —  Train a 1D CNN classifier on hand-landmark features
=========================================================================
Loads landmarks.csv, reuses the existing label_encoder.joblib,
builds a Conv1D model, and saves it as classifier_cnn.keras.

Usage:
    python isl_sign2text/03_train_cnn.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import joblib

# ── TensorFlow / Keras ──────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # suppress INFO logs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data_npz"
LE_PATH     = SCRIPT_DIR / "models" / "label_encoder.joblib"
MODEL_OUT   = SCRIPT_DIR / "models" / "classifier_cnn.keras"

# ── 1. Load data ────────────────────────────────────────────────────
print("=" * 60)
print("  CNN Training Pipeline — ISL Sign2Text")
print("=" * 60)

if not LE_PATH.exists():
    sys.exit(f"[ERROR] label_encoder.joblib not found at {LE_PATH}")

def load_all_data(data_dir):
    X_list, y_list = [], []
    files = list(data_dir.glob("*.npz"))
    if not files:
        sys.exit(f"[ERROR] No .npz files found in {data_dir}")
    
    print(f"[DATA]  Loading {len(files)} dataset files...")
    for f in files:
        data = np.load(f)
        X_list.append(data["X"])
        y_list.append(data["y"])
    
    return np.vstack(X_list), np.concatenate(y_list)

X, y_raw = load_all_data(DATA_DIR)
X = X.astype(np.float32)

print(f"[DATA]  Total samples : {X.shape[0]}")
print(f"[DATA]  Feature shape : {X.shape[1]}")
print(f"[DATA]  Unique classes: {len(np.unique(y_raw))}")

# ── 2. Encode labels using the EXISTING label encoder ───────────────
le = joblib.load(str(LE_PATH))

# Filter out labels not in the encoder (e.g., 'FUCK YOU')
valid_mask = np.isin(y_raw, le.classes_)
X = X[valid_mask]
y_filtered = y_raw[valid_mask]

if len(X) < len(y_raw):
    print(f"[CLEAN] Removed {len(y_raw) - len(X)} samples with unknown labels.")

y_encoded = le.transform(y_filtered)
num_classes = len(le.classes_)
print(f"[DATA]  Using label_encoder with {num_classes} classes")

# One-hot encode for categorical_crossentropy
y_onehot = keras.utils.to_categorical(y_encoded, num_classes=num_classes)

# ── 3. Train / validation split ────────────────────────────────────
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"[SPLIT] Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}")

# Reshape for Conv1D: (samples, 126) → (samples, 126, 1)
X_train = X_train.reshape(-1, 126, 1)
X_val   = X_val.reshape(-1, 126, 1)

# ── 4. Build model ──────────────────────────────────────────────────
model = keras.Sequential([
    layers.Input(shape=(126, 1)),

    layers.Conv1D(64,  kernel_size=3, activation="relu", padding="same"),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
    layers.GlobalAveragePooling1D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── 5. Train ────────────────────────────────────────────────────────
print("\n[TRAIN] Starting training …")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1,
)

# ── 6. Report ───────────────────────────────────────────────────────
train_acc = history.history["accuracy"][-1]
val_acc   = history.history["val_accuracy"][-1]

print("\n" + "=" * 60)
print(f"  ✅  Training   accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
print(f"  ✅  Validation accuracy : {val_acc:.4f}  ({val_acc*100:.2f}%)")
print("=" * 60)

# ── 7. Save ─────────────────────────────────────────────────────────
model.save(str(MODEL_OUT))
print(f"\n[SAVE] Model saved → {MODEL_OUT}")
print(f"[SAVE] Size: {MODEL_OUT.stat().st_size / 1024:.1f} KB")
print("[DONE] CNN training complete.\n")
