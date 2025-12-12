
import io
import os
import sys
import logging
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyodbc
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


# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")






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






"""
MWPS Big Classifier Training
- Pretrain on GOLDEN sequences
- Then continue training on ALL sequences
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


from model_generation.Model import build_big_mwps_classifier
# Adding the path to the directory containing ReViKeyWords script
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

BASE_DIR = cross_os_path(r"D:\ReViAI\Trained models\MWPS\epochs 500\EURUSD\M15\MWPS EURUSD M15 e.500 V.2025-07-03-04-25")

GOLDEN_NPZ_PATH = os.path.join(BASE_DIR, "classification_sequences_GOLDEN.npz")
ALL_NPZ_PATH    = os.path.join(BASE_DIR, "MWPSDATA_EURUSD_M15_O21uv1_classification_sequences.npz")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "MWPS_3M_Classifier.h5")
LABEL_MAP_PATH  = os.path.join(BASE_DIR, "MWPS_3M_Classifier_label_map.npy")  # saves label2idx

# =============================================================================
# MODEL
# =============================================================================












# =============================================================================
# DATA HELPERS
# =============================================================================

def load_npz_sequences(path):
    """Load X, y, t_seq from an npz (t_seq optional)."""
    data = np.load(path, allow_pickle=True)
    X = data["X_low"]
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
    Build label2idx mapping from ALL labels.
    Example labels: [-10, -9, ..., -1, 1, ..., 10]
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







def train_mwps_classifier(training_data_path):
    # ---------------- Load data ----------------
    # Expecting load_npz_sequences to return (X, y_raw, meta)
    X, y_raw, _ = load_npz_sequences(training_data_path)

    # X shape: (N, T, 1, F, 1)
    time_steps = X.shape[1]
    n_features = X.shape[3]

    # ---------------- Label mapping ----------------
    label2idx, idx2label, num_classes = build_label_mapping(y_raw)

    # Encode labels
    y_idx = encode_labels(y_raw, label2idx)

    # Save label2idx for later inference
    np.save(LABEL_MAP_PATH, label2idx)
    print(f"[save] label2idx → {LABEL_MAP_PATH}")

    # ---------------- Build model ----------------
    model = build_big_mwps_classifier(
        time_steps=time_steps,
        n_features=n_features,
        num_classes=num_classes,
        l2_reg=1e-5,
        dropout_rate=0.3,
    )
    model.summary()

    # ---------------- Train/validation split ----------------
    from sklearn.model_selection import train_test_split

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
        batch_size=32,
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






if __name__ == "__main__":
    print("=== MWPS Big Classifier Trainer ===")

    try:
        print("TensorFlow version:", tf.__version__)
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    except Exception as e:
        print("TensorFlow not fully available:", e)

    train_mwps_classifier(GOLDEN_NPZ_PATH)
    print("=== DONE ===")




