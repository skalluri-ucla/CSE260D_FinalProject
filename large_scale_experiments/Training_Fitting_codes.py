"""
MWPS Big Classifier Training
- Train on a single NPZ (e.g., GOLDEN or ALL)
- Uses a lighter classifier + distance-aware custom loss
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import io
import os
import sys

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
from datetime import datetime
import shutil

import hashlib
import base64

import hashlib
import base64
import pandas as pd
import asyncio


# Determine the current operating system
is_windows = os.name == 'nt'

# Define paths based on the operating system
if is_windows:
    standards_path = "D:\\ReViAI\\Github\\ReViStandards"
    database_path = "D:\\ReViAI\\Github\\DataBaseAI"
    management_path = "D:\\ReViAI\\Github\\Management"
    ReVi_API_path = "D:\\ReViAI\\Github\\ReVi_API"
else:
    standards_path = "/mnt/3090D/ReViAI/Github/ReViStandards"
    database_path = "/mnt/3090D/ReViAI/Github/DataBaseAI"
    management_path = "/mnt/3090D/ReViAI/Github/Management"
    ReVi_API_path = "/mnt/3090D/ReViAI/Github/ReVi_API"

# Adding the path to the directory containing Standards for Ai-Cores.py
sys.path.append(standards_path)
sys.path.append(database_path)
sys.path.append(management_path)
sys.path.append(ReVi_API_path)



# Path configuration based on the operating system
base_path = "D:\\ReViAI\\Github" if os.name == 'nt' else "/mnt/3090D/ReViAI/Github"
standards_path = os.path.join(base_path, "ReViStandards")
database_path = os.path.join(base_path, "DataBaseAI")
management_path = os.path.join(base_path, "Management")
ReVi_API_path = os.path.join(base_path, "ReVi_API")





# We do NOT need build_big_mwps_classifier anymore for this script.
# from Ai_Model_codes import build_big_mwps_classifier

from ReViKeyWords import (
    get_table_name,
    connect_to_database,
    load_sqlalchemy_database_credentials,
    Normalize_Digits,
    generate_model_file_name,
    generate_model_Version,
    generate_main_folder,
    generate_plot_file_name,
    get_plot_Path,
    load_REVIAI_database_remote_credentials,
    load_sqlalchemy_database_credentials,
    frequency_mapping,
    cross_os_path,
    to_windows_path
)



# =============================================================================
# CONFIG
# =============================================================================

if os.name == "nt":
    # Windows machine (e.g., DELLR15 / ReVi4090)
    BASE_DIR = r"D:\ReViAI\Trained models\MWPS\epochs 500\EURUSD\M15\MWPS EURUSD M15 e.500 V.2025-07-03-04-25"
else:
    # Linux machine (RTX3090 VPS mount)
    BASE_DIR = "/mnt/3090D/ReViAI/Trained models/MWPS/epochs 500/EURUSD/M15/MWPS EURUSD M15 e.500 V.2025-07-03-04-25"

print("BASE_DIR =", BASE_DIR)

GOLDEN_NPZ_PATH = os.path.join(BASE_DIR, "classification_sequences_GOLDEN.npz")
ALL_NPZ_PATH    = os.path.join(BASE_DIR, "MWPSDATA_EURUSD_M15_O21uv1_classification_sequences.npz")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "MWPS_3M_Classifier_Light.h5")
LABEL_MAP_PATH  = os.path.join(BASE_DIR, "MWPS_3M_Classifier_label_map.npy")  # saves label2idx




def build_mwps_light_classifier(
    time_steps: int,
    n_features: int,
    num_classes: int,
    l2_reg: float = 1e-5,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Input:  (batch, time_steps, n_features)
    Output: class probabilities (num_classes) via softmax
    """

    reg = regularizers.l2(l2_reg)

    inp = layers.Input(shape=(time_steps, n_features), name="X_seq")

    # Optional masking if you pad with zeros at some point
    x = layers.Masking(mask_value=0.0, name="mask")(inp)

    # 1D Conv for local temporal patterns
    x = layers.Conv1D(
        filters=128,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_regularizer=reg,
        name="conv1",
    )(x)
    x = layers.BatchNormalization(name="bn_conv1")(x)

    # BiLSTM stack
    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            kernel_regularizer=reg,
            name="lstm_1",
        ),
        name="bilstm_1",
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=False,
            kernel_regularizer=reg,
            name="lstm_2",
        ),
        name="bilstm_2",
    )(x)

    # Dense head
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_1",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_2",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    out = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_output",
    )(x)

    model = models.Model(inputs=inp, outputs=out, name="MWPS_LightClassifier")
    return model


# =============================================================================
# CUSTOM LOSS: distance-aware
# =============================================================================

def make_distance_aware_loss(label_values, alpha: float = 1.0):
    """
    label_values: list/np.array of numeric label values (e.g. [-10, -9, ..., 10])
                  indexed by class index (0..num_classes-1)

    alpha: weight for distance penalty term.
           Larger alpha => stronger punishment for far misclassifications.

    Penalty = CE + alpha * |y_true_numeric - y_pred_numeric|
    """

    label_values = tf.constant(label_values, dtype=tf.float32)  # (C,)

    def loss_fn(y_true, y_pred):
        """
        y_true: integer class indices, shape (batch,) or (batch,1)
        y_pred: softmax probabilities, shape (batch, C)
        """
        # Ensure correct shape
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)  # (batch,)

        # Standard sparse categorical cross-entropy
        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=False,
        )  # (batch,)

        # Numeric value of true label: label_values[y_true]
        true_vals = tf.gather(label_values, y_true)  # (batch,)

        # Expected numeric value under predicted distribution:
        # pred_vals[i] = sum_c p[i,c] * label_values[c]
        pred_vals = tf.tensordot(y_pred, label_values, axes=1)  # (batch,)

        # Distance penalty: |true_numeric - pred_numeric|
        dist = tf.abs(true_vals - pred_vals)  # (batch,)

        return ce + alpha * dist

    return loss_fn


# =============================================================================
# DATA HELPERS
# =============================================================================

def load_npz_sequences(path):
    """Load X, y, t_seq from an npz (t_seq optional)."""
    data = np.load(path, allow_pickle=True)
    X = data["X_low"]   # (N, T, 1, F, 1)
    y = data["y"]
    t_seq = data["t_seq"] if "t_seq" in data.files else None

    print(f"[load_npz] {path}")
    print("  X.shape =", X.shape)
    print("  y.shape =", y.shape)
    if t_seq is not None:
        print("  t_seq.shape =", t_seq.shape)
    else:
        print("  t_seq not present in file.")

    return X, y, t_seq


def build_label_mapping(y_all):
    """
    Build label2idx mapping from numeric labels in y_all.
    Example labels: [-10, -9, ..., -1, 0, 1, ..., 10]
    """
    unique_labels = np.unique(y_all)        # sorted
    unique_labels = unique_labels.tolist()

    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}

    print("Label2Idx:", label2idx)
    print("num_classes =", len(unique_labels))

    return label2idx, idx2label, len(unique_labels)


def encode_labels(y, label2idx):
    """Map raw numeric labels → integer class indices."""
    return np.array([label2idx[lab] for lab in y], dtype=np.int32)


def make_class_weights(y_idx, num_classes):
    """
    Compute class weights safely:
    - Only use classes that actually appear in y_idx for sklearn.
    - Ensure every class 0..num_classes-1 has a weight (default 1.0 if missing).
    """
    classes_present = np.unique(y_idx)
    print("  classes_present in this split:", classes_present)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_present,
        y=y_idx
    )
    class_weights = {int(c): float(w) for c, w in zip(classes_present, weights)}

    # Ensure weights exist for all classes
    for c in range(num_classes):
        class_weights.setdefault(c, 1.0)

    return class_weights


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_mwps_classifier(training_data_path):
    """
    Train on a single NPZ file (e.g., GOLDEN or ALL).
    Uses the light model + distance-aware loss.
    """
    # ---------------- Load data ----------------
    X_5d, y_raw, _ = load_npz_sequences(training_data_path)  # X_5d: (N, T, 1, F, 1)

    # Reshape to 3D for the new model: (N, T, F)
    X = X_5d[:, :, 0, :, 0]
    print("[reshape] X 5D → 3D:", X.shape)

    time_steps = X.shape[1]
    n_features = X.shape[2]

    # ---------------- Label mapping ----------------
    label2idx, idx2label, num_classes = build_label_mapping(y_raw)

    # Encode labels to 0..num_classes-1
    y_idx = encode_labels(y_raw, label2idx)

    # Save label2idx for later inference
    np.save(LABEL_MAP_PATH, label2idx)
    print(f"[save] label2idx → {LABEL_MAP_PATH}")

    # label_values aligned with class indices: [v_0, v_1, ..., v_C-1]
    label_values = np.array([idx2label[i] for i in range(num_classes)], dtype=np.float32)
    print("label_values (per class index):", label_values)

    # ---------------- Build model ----------------
    model = build_mwps_light_classifier(
        time_steps=time_steps,
        n_features=n_features,
        num_classes=num_classes,
        l2_reg=1e-5,
        dropout_rate=0.3,
    )

    loss_fn = make_distance_aware_loss(label_values, alpha=1.0)  # tune alpha if needed

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    model.summary()

    # ---------------- Train/validation split ----------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_idx,
        test_size=0.2,
        random_state=42,
        stratify=y_idx,
    )

    # Class weights for imbalance
    class_weights = make_class_weights(y_train, num_classes=num_classes)
    print("class_weights:", class_weights)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    # ---------------- Train model ----------------
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,   # you can likely go higher on 4090
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    # ---------------- Final evaluation ----------------
    print("\n=== FINAL EVALUATION on FULL DATASET ===")
    all_loss, all_acc = model.evaluate(X, y_idx, verbose=0)
    print(f"ALL → loss={all_loss:.4f}, acc={all_acc:.4f}")

    # ---------------- Save model ----------------
    model.save(MODEL_SAVE_PATH)
    print(f"\n[save] model → {MODEL_SAVE_PATH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=== MWPS Big Classifier Trainer (Light) ===")

    try:
        print("TensorFlow version:", tf.__version__)
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    except Exception as e:
        print("TensorFlow not fully available:", e)

    # For now: train on GOLDEN only
    train_mwps_classifier(GOLDEN_NPZ_PATH)

    # Later you can switch to ALL:
    # train_mwps_classifier(ALL_NPZ_PATH)

    print("=== DONE ===")
