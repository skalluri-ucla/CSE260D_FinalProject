"""
MWPS Big Classifier Training (Light 3D, 2-phase test, Indexer-based)
- Phase 1 (train1): train on INDEXER_NPZ (e.g., Sidhi_dataframe...npz)
    * Forgettability on GOLDEN          → forgettability_train1_{indexer}.xlsx
    * Model                             → Model_train_1_{indexer}.keras
    * History                           → hist_train_1_{indexer}.xlsx
- Phase 2 (train2): load Model_train_1 and continue training on ALL (streaming)
    * Forgettability on GOLDEN          → forgettability_train2_{indexer}.xlsx
    * Model                             → Model_train_2_{indexer}.keras
    * History                           → hist_train_2_{indexer}.xlsx
"""

import os
import gc
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import Sequence
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# import your models
from model_generation.Model import build_mwps_light_classifier   # <<< use LIGHT version here



# =============================================================================
# PATH / CONFIG
# =============================================================================

# Original BASE_DIR (where NPZs live)
if os.name == "nt":
    BASE_DIR = r"D:\ReViAI\Trained models\MWPS\epochs 500\EURUSD\M15\MWPS EURUSD M15 e.500 V.2025-07-03-04-25"
else:
    BASE_DIR = "/mnt/3090D/ReViAI/Trained models/MWPS/epochs 500\\EURUSD\\M15\\MWPS EURUSD M15 e.500 V.2025-07-03-04-25".replace(
        "\\", "/"
    )

GOLDEN_NPZ_PATH   = os.path.join(BASE_DIR, "classification_sequences_GOLDEN.npz")
# Indexer_NPZ_PATH  = os.path.join(BASE_DIR, "Sidhi_dataframe_w_perp_and_div_v2.npz")
Indexer_NPZ_PATH  = os.path.join(BASE_DIR, "classification_sequences_GOLDEN.npz")
ALL_NPZ_PATH      = os.path.join(BASE_DIR, "MWPSDATA_EURUSD_M15_O21uv1_classification_sequences.npz")

print("BASE_DIR =", BASE_DIR)

# Derive indexer core name (used to name models/artifacts)
INDEXER_CORE_NAME = os.path.splitext(os.path.basename(Indexer_NPZ_PATH))[0]

# ------------------ LOCAL ROOT + EXPERIMENT DIR ------------------ #
if os.name == "nt":
    LOCAL_CACHE_ROOT = r"C:\MWPS_cache"      # root for all experiments
else:
    LOCAL_CACHE_ROOT = "/home/revi/MWPS_cache"

RUN_PREFIX = f"EXP_Light_{INDEXER_CORE_NAME}"     # experiment folder per indexer
EXPERIMENT_DIR = os.path.join(LOCAL_CACHE_ROOT, RUN_PREFIX)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

print("EXPERIMENT_DIR =", EXPERIMENT_DIR)


# =============================================================================
# FORGETTABILITY CALLBACK
# =============================================================================

class ForgettabilityCallback(tf.keras.callbacks.Callback):
    """
    After each epoch, evaluate the model on ALL given samples (X_full, y_full)
    and write a 0/1 correctness column to an Excel file.

    - index: t_seq (DateTimeIndex or simple index)
    - columns: epoch_001, epoch_002, ...
    """

    def __init__(self, X_full, y_full, t_seq, out_xlsx_path, batch_size=1024):
        super().__init__()
        self.X_full = X_full              # (N, T, F)
        self.y_full = y_full              # int class indices (N,)
        self.t_seq = t_seq                # datetime64 or similar (N,)
        self.out_xlsx_path = out_xlsx_path
        self.batch_size = batch_size

        # Prepare index as pandas DateTimeIndex (or simple RangeIndex fallback)
        try:
            self.index = pd.to_datetime(self.t_seq)
        except Exception:
            self.index = pd.RangeIndex(len(self.t_seq))

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        col_name = f"epoch_{epoch_num:03d}"
        print(f"\n[Forgettability] Evaluating dataset on epoch {epoch_num}...")

        # Predict on all samples
        y_pred = self.model.predict(
            self.X_full,
            batch_size=self.batch_size,
            verbose=0,
        )
        y_pred_idx = np.argmax(y_pred, axis=1)

        correct = (y_pred_idx == self.y_full).astype(int)

        # Either load existing Excel or start a new DataFrame
        if os.path.exists(self.out_xlsx_path):
            df = pd.read_excel(self.out_xlsx_path, index_col=0)
            if len(df) != len(self.index):
                print("[Forgettability] Warning: length mismatch; recreating DataFrame.")
                df = pd.DataFrame(index=self.index)
        else:
            df = pd.DataFrame(index=self.index)

        df[col_name] = correct

        df.to_excel(self.out_xlsx_path)
        print(f"[Forgettability] Saved epoch {epoch_num} correctness → {self.out_xlsx_path}")


# =============================================================================
# CUSTOM LOSS: DISTANCE-AWARE
# =============================================================================

def make_distance_aware_loss(label_values, alpha: float = 1.0):
    """
    label_values: list/np.array of numeric label values (e.g. [-10, -9, ..., 10])
                  indexed by class index (0..num_classes-1)

    alpha: weight for distance penalty term.
           Larger alpha => stronger punishment for far misclassifications.
    """
    label_values = tf.constant(label_values, dtype=tf.float32)  # (C,)

    base_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction="none",  # per-sample loss
    )

    def loss_fn(y_true, y_pred):
        """
        y_true: integer class indices, shape (batch,) or (batch,1)
        y_pred: softmax probabilities, shape (batch, C)
        """
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)  # (batch,)

        ce = base_loss_obj(y_true, y_pred)  # (batch,)

        true_vals = tf.gather(label_values, y_true)  # (batch,)
        pred_vals = tf.reduce_sum(y_pred * label_values, axis=-1)  # (batch,)

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


def extract_labels(y_raw):
    """
    Ensure we get a 1D label vector out of y_raw.
    If y_raw has extra columns (e.g., index+label), we keep only the label.
    """
    print("=== DEBUG y_raw ===")
    print("  y_raw.shape:", y_raw.shape)
    print("  y_raw.dtype:", y_raw.dtype)

    # Case 1: structured dtype
    if isinstance(y_raw, np.ndarray) and y_raw.dtype.names is not None:
        print("  y_raw has named fields:", y_raw.dtype.names)
        labels = y_raw["label"]  # adjust name if needed
    # Case 2: 2D array (N, k) -> last column is label
    elif y_raw.ndim == 2:
        print("  y_raw is 2D, assuming last column is label")
        labels = y_raw[:, -1]
    # Case 3: already 1D
    else:
        labels = y_raw

    print("First 20 labels:", labels[:20])
    return labels.astype(np.float32)


def build_label_mapping(y_all):
    """
    Build label2idx mapping from numeric labels in y_all.
    Example labels: [-10, -9, ..., -1, 1, ..., 10]  (0s removed)
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
    - Only use classes that appear in y_idx for sklearn.
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

    for c in range(num_classes):
        class_weights.setdefault(c, 1.0)

    return class_weights


# =============================================================================
# PREPARE DATASET (RAM version – used for INDEXER / GOLDEN)
# =============================================================================

def prepare_dataset(npz_path, label2idx=None, idx2label=None):
    """
    Load NPZ, drop label==0, reshape to (N, T, F), and encode labels.

    If label2idx/idx2label are provided, reuse them (for GOLDEN / ALL).
    Otherwise create them (for INDEXER).
    """
    X_5d, y_raw, t_seq = load_npz_sequences(npz_path)
    y_raw = extract_labels(y_raw)

    # Drop numeric 0 labels
    mask = (y_raw != 0.0)
    num_dropped = np.count_nonzero(~mask)
    if num_dropped > 0:
        print(f"[filter] dropping {num_dropped} / {len(y_raw)} samples with label == 0.0")
        X_5d = X_5d[mask]
        y_raw = y_raw[mask]
        if t_seq is not None:
            t_seq = t_seq[mask]
    else:
        print(f"[filter] no label == 0.0 found, using all {len(y_raw)} samples")

    # Reshape to 3D
    X = X_5d[:, :, 0, :, 0].astype(np.float32)
    print("[reshape] X 5D → 3D:", X.shape)

    time_steps = X.shape[1]
    n_features = X.shape[2]

    # Build or reuse label mapping
    if label2idx is None or idx2label is None:
        label2idx, idx2label, num_classes = build_label_mapping(y_raw)
    else:
        # Reuse existing mapping; assume other datasets use same labels
        unique_labels = np.unique(y_raw)
        missing = [lab for lab in unique_labels if lab not in label2idx]
        if missing:
            print("⚠️ Warning: some labels in this dataset not in provided label2idx:", missing)
        num_classes = len(label2idx)

    y_idx = encode_labels(y_raw, label2idx)

    label_values = np.array(
        [idx2label[i] for i in range(num_classes)],
        dtype=np.float32
    )

    class_weights = make_class_weights(y_idx, num_classes=num_classes)

    return {
        "X": X,
        "y_idx": y_idx,
        "t_seq": t_seq if t_seq is not None else np.arange(len(X)),
        "time_steps": time_steps,
        "n_features": n_features,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "num_classes": num_classes,
        "label_values": label_values,
        "class_weights": class_weights,
    }


# =============================================================================
# PREPARE ALL FOR STREAMING (memmap)
# =============================================================================

def prepare_all_for_streaming(npz_path, label2idx, idx2label):
    """
    Prepare the ALL dataset in streaming mode:
    - X is kept as a memmap (no full load to RAM).
    - We filter out label==0.0 by using an index list.
    - y and t_seq for the *used* samples are kept in normal arrays.
    """
    data = np.load(npz_path, allow_pickle=True, mmap_mode="r")

    X_5d = data["X_low"]          # memmap: shape (N, T, 1, F, 1)
    y_raw_all = extract_labels(data["y"])
    t_seq_all = data["t_seq"] if "t_seq" in data.files else np.arange(len(y_raw_all))

    print(f"[stream ALL] X_5d.shape = {X_5d.shape}")
    print(f"[stream ALL] y_raw_all.shape = {y_raw_all.shape}")

    # Drop label==0.0 via index mask
    mask = (y_raw_all != 0.0)
    idx_source = np.nonzero(mask)[0]       # indices into original memmap
    y_raw_used = y_raw_all[mask]
    t_seq_used = t_seq_all[mask]

    print(f"[stream ALL] dropping {np.count_nonzero(~mask)} / {len(y_raw_all)} (label==0.0)")
    print(f"[stream ALL] using {len(y_raw_used)} samples")

    # Shapes
    time_steps = X_5d.shape[1]
    n_features = X_5d.shape[3]

    # Encode labels for the *used* samples
    y_idx_used = encode_labels(y_raw_used, label2idx)

    # Class weights on used samples
    num_classes = len(label2idx)
    class_weights = make_class_weights(y_idx_used, num_classes=num_classes)

    return {
        "X_memmap": X_5d,
        "source_indices": idx_source,   # original row indices in X_5d
        "y_idx_used": y_idx_used,       # labels aligned with filtered positions
        "t_seq_used": t_seq_used,
        "time_steps": time_steps,
        "n_features": n_features,
        "class_weights": class_weights,
    }


# =============================================================================
# STREAMING SEQUENCE FOR KERAS
# =============================================================================

class NPZStreamingSequence(Sequence):
    """
    Streams batches from a big NPZ memmap.

    - X_memmap:   memmap, shape (N, T, 1, F, 1)
    - y_idx_filtered: labels array for *filtered* samples (label!=0.0), len = N_used
    - source_indices: mapping from filtered position -> original row in X_memmap
    - positions:  subset of filtered positions used for this Sequence
    """

    def __init__(
        self,
        X_memmap,
        y_idx_filtered,
        source_indices,
        positions,
        batch_size=64,
        shuffle=True,
    ):
        self.X_memmap = X_memmap
        self.y_idx_filtered = y_idx_filtered
        self.source_indices = source_indices    # shape (N_used,)
        self.positions = np.array(positions)    # shape (N_subset,)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.positions) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.positions))

        batch_pos = self.positions[start:end]          # positions in filtered arrays
        orig_idx = self.source_indices[batch_pos]      # indices into X_memmap

        X_batch_5d = self.X_memmap[orig_idx, ...]      # (B, T, 1, F, 1)
        X_batch = X_batch_5d[:, :, 0, :, 0].astype(np.float32)  # (B, T, F)

        y_batch = self.y_idx_filtered[batch_pos]

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.positions)


# =============================================================================
# BUILD + COMPILE LIGHT MODEL
# =============================================================================

def build_and_compile_model(time_steps, n_features, num_classes, label_values, alpha=1.0, lr=1e-4):
    # LIGHT MWPS classifier
    model = build_mwps_light_classifier(
        time_steps=time_steps,
        n_features=n_features,
        num_classes=num_classes,
        l2_reg=1e-5,
        dropout_rate=0.3,
    )

    loss_fn = make_distance_aware_loss(label_values, alpha=alpha)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    model.summary()
    return model, loss_fn


# =============================================================================
# COPY NPZs INTO EXPERIMENT FOLDER
# =============================================================================

def cache_npz_to_local(npz_path: str, dest_dir: str) -> str:
    """
    Copy a big NPZ from network to a *given destination directory* (e.g. EXPERIMENT_DIR).

    If the local copy already exists with the same size and is newer or same mtime,
    it will be reused.

    Returns the path to the *local* NPZ inside dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)

    npz_path = os.path.abspath(npz_path)
    local_path = os.path.join(dest_dir, os.path.basename(npz_path))

    # If source already in dest_dir, no need to copy
    if os.path.abspath(npz_path) == os.path.abspath(local_path):
        print(f"[cache] NPZ already local in dest_dir: {npz_path}")
        return npz_path

    need_copy = True
    if os.path.exists(local_path):
        try:
            src_stat = os.stat(npz_path)
            dst_stat = os.stat(local_path)
            same_size = (src_stat.st_size == dst_stat.st_size)
            not_older = (dst_stat.st_mtime >= src_stat.st_mtime)
            if same_size and not_older:
                need_copy = False
        except OSError:
            need_copy = True

    if need_copy:
        print(f"[cache] Copying NPZ → dest_dir\n  from: {npz_path}\n  to:   {local_path}")
        shutil.copy2(npz_path, local_path)
        print("[cache] Copy complete.")
    else:
        print(f"[cache] Using existing local copy in dest_dir: {local_path}")

    return local_path


# =============================================================================
# PHASE 1: train1 on INDEXER (Sidhi...), forgettability on GOLDEN
# =============================================================================

def run_train1(indexer_npz_local, golden_npz_local, epochs=200, batch_size=64, alpha=1.0):
    print("\n=== PHASE 1: train1 on INDEXER (LIGHT, forgettability on GOLDEN) ===")

    # Prepare INDEXER dataset (training)
    indexer = prepare_dataset(indexer_npz_local)
    X_idx = indexer["X"]
    y_idx_idx = indexer["y_idx"]
    t_idx = indexer["t_seq"]
    time_steps = indexer["time_steps"]
    n_features = indexer["n_features"]
    label2idx = indexer["label2idx"]
    idx2label = indexer["idx2label"]
    num_classes = indexer["num_classes"]
    label_values = indexer["label_values"]
    class_weights = indexer["class_weights"]

    # Prepare GOLDEN dataset only for forgettability (reuse label mapping)
    golden = prepare_dataset(golden_npz_local, label2idx=label2idx, idx2label=idx2label)
    X_g = golden["X"]
    y_g = golden["y_idx"]
    t_g = golden["t_seq"]

    # Save label_map for reference (per indexer)
    label_map_path = os.path.join(EXPERIMENT_DIR, f"label_map_{INDEXER_CORE_NAME}.npy")
    np.save(label_map_path, label2idx)
    print(f"[save] label2idx → {label_map_path}")

    # Build & compile LIGHT model
    model, loss_fn = build_and_compile_model(
        time_steps=time_steps,
        n_features=n_features,
        num_classes=num_classes,
        label_values=label_values,
        alpha=alpha,
        lr=1e-4,
    )

    # Split INDEXER into train/val
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_idx,
        y_idx_idx,
        t_idx,
        test_size=0.2,
        random_state=42,
        stratify=y_idx_idx,
    )

    # Callbacks for PHASE 1
    forget1_path = os.path.join(EXPERIMENT_DIR, f"forgettability_train1_{INDEXER_CORE_NAME}.xlsx")
    model_train1_path = os.path.join(EXPERIMENT_DIR, f"Model_train_1_{INDEXER_CORE_NAME}.keras")
    hist_train1_path = os.path.join(EXPERIMENT_DIR, f"hist_train_1_{INDEXER_CORE_NAME}.xlsx")

    # Forgettability measured on GOLDEN
    forget_cb_1 = ForgettabilityCallback(
        X_full=X_g,
        y_full=y_g,
        t_seq=t_g,
        out_xlsx_path=forget1_path,
        batch_size=1024,
    )

    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )

    callbacks = [es_cb, forget_cb_1]

    # Train on INDEXER
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    # Final eval on full INDEXER
    print("\n=== FINAL EVAL on INDEXER (train1, LIGHT) ===")
    all_loss, all_acc = model.evaluate(X_idx, y_idx_idx, verbose=0)
    print(f"train1 (INDEXER, LIGHT) → loss={all_loss:.4f}, acc={all_acc:.4f}")

    # Save model and history
    model.save(model_train1_path)
    print(f"[save] Model_train_1 → {model_train1_path}")

    hist_df = pd.DataFrame(history.history)
    hist_df.to_excel(hist_train1_path, index=False)
    print(f"[save] hist_train_1 → {hist_train1_path}")

    return {
        "model_path": model_train1_path,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "label_values": label_values,
        "indexer": indexer,
        "golden": golden,
    }


# =============================================================================
# PHASE 2: train2 on ALL (streaming), starting from Model_train_1,
#          forgettability measured on GOLDEN
# =============================================================================

def run_train2(all_npz_local, phase1_info, epochs=200, batch_size=64, alpha=1.0):
    print("\n=== PHASE 2: train2 on ALL (LIGHT, start from Model_train_1) ===")

    model_train1_path = phase1_info["model_path"]
    label2idx = phase1_info["label2idx"]
    idx2label = phase1_info["idx2label"]
    label_values = phase1_info["label_values"]
    golden = phase1_info["golden"]

    # ---- Prepare ALL dataset in streaming mode ----
    all_data = prepare_all_for_streaming(all_npz_local, label2idx=label2idx, idx2label=idx2label)

    X_memmap       = all_data["X_memmap"]
    source_indices = all_data["source_indices"]
    y_all_idx_used = all_data["y_idx_used"]
    t_all_used     = all_data["t_seq_used"]
    time_steps     = all_data["time_steps"]
    n_features     = all_data["n_features"]
    print(f"[train2] ALL (filtered) samples = {len(y_all_idx_used)}")

    n_used = len(y_all_idx_used)
    positions = np.arange(n_used)  # positions in filtered arrays

    # ---- Train/val split on positions ----
    pos_train, pos_val = train_test_split(
        positions,
        test_size=0.2,
        random_state=42,
        stratify=y_all_idx_used,
    )

    # Class weights using train positions
    num_classes = len(label2idx)
    class_weights_all = make_class_weights(y_all_idx_used[pos_train], num_classes=num_classes)
    print("[train2] class_weights:", class_weights_all)

    # ---- Load Model_train_1 (LIGHT) and recompile ----
    model = tf.keras.models.load_model(
        model_train1_path,
        compile=False,
        safe_mode=False,
    )
    loss_fn = make_distance_aware_loss(label_values, alpha=alpha)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    model.summary()

    # ---- Build streaming sequences for ALL ----
    train_seq = NPZStreamingSequence(
        X_memmap=X_memmap,
        y_idx_filtered=y_all_idx_used,
        source_indices=source_indices,
        positions=pos_train,
        batch_size=batch_size,
        shuffle=True,
    )

    val_seq = NPZStreamingSequence(
        X_memmap=X_memmap,
        y_idx_filtered=y_all_idx_used,
        source_indices=source_indices,
        positions=pos_val,
        batch_size=batch_size,
        shuffle=False,
    )

    # ---- GOLDEN data for forgettability in PHASE 2 ----
    X_g = golden["X"]
    y_g = golden["y_idx"]
    t_g = golden["t_seq"]

    # Callbacks for PHASE 2
    forget2_path      = os.path.join(EXPERIMENT_DIR, f"forgettability_train2_{INDEXER_CORE_NAME}.xlsx")
    model_train2_path = os.path.join(EXPERIMENT_DIR, f"Model_train_2_{INDEXER_CORE_NAME}.keras")
    hist_train2_path  = os.path.join(EXPERIMENT_DIR, f"hist_train_2_{INDEXER_CORE_NAME}.xlsx")

    forget_cb_2 = ForgettabilityCallback(
        X_full=X_g,
        y_full=y_g,
        t_seq=t_g,
        out_xlsx_path=forget2_path,
        batch_size=1024,
    )

    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )

    callbacks = [es_cb, forget_cb_2]

    # ---- Train on ALL (streaming) ----
    history2 = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        class_weight=class_weights_all,
        callbacks=callbacks,
        verbose=2,
    )

    # ---- Final eval on ALL using streaming as well ----
    print("\n=== FINAL EVAL on ALL (train2, streaming, LIGHT) ===")
    eval_positions = np.arange(n_used)
    eval_seq = NPZStreamingSequence(
        X_memmap=X_memmap,
        y_idx_filtered=y_all_idx_used,
        source_indices=source_indices,
        positions=eval_positions,
        batch_size=batch_size,
        shuffle=False,
    )
    all_loss2, all_acc2 = model.evaluate(eval_seq, verbose=0)
    print(f"train2 (ALL, LIGHT) → loss={all_loss2:.4f}, acc={all_acc2:.4f}")

    # ---- Save model and history ----
    model.save(model_train2_path)
    print(f"[save] Model_train_2 → {model_train2_path}")

    hist2_df = pd.DataFrame(history2.history)
    hist2_df.to_excel(hist_train2_path, index=False)
    print(f"[save] hist_train_2 → {hist_train2_path}")

    return {
        "model_path": model_train2_path,
        "all_loss": float(all_loss2),
        "all_acc": float(all_acc2),
    }


# =============================================================================
# MAIN (2-phase test, LIGHT)
# =============================================================================

if __name__ == "__main__":
    print("=== MWPS 2-phase LIGHT test (INDEXER-based): INDEXER → ALL, forgettability on GOLDEN ===")
    print("BASE_DIR       =", BASE_DIR)
    print("EXPERIMENT_DIR =", EXPERIMENT_DIR)
    print("INDEXER_CORE   =", INDEXER_CORE_NAME)

    try:
        print("TensorFlow version:", tf.__version__)
        print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
    except Exception as e:
        print("TensorFlow not fully available:", e)

    # Cache NPZs directly into EXPERIMENT_DIR (copy all important data locally)
    GOLDEN_NPZ_LOCAL  = cache_npz_to_local(GOLDEN_NPZ_PATH,  EXPERIMENT_DIR)
    INDEXER_NPZ_LOCAL = cache_npz_to_local(Indexer_NPZ_PATH, EXPERIMENT_DIR)
    ALL_NPZ_LOCAL     = cache_npz_to_local(ALL_NPZ_PATH,     EXPERIMENT_DIR)

    # Phase 1: train on INDEXER, forgettability on GOLDEN
    phase1_info = run_train1(
        indexer_npz_local=INDEXER_NPZ_LOCAL,
        golden_npz_local=GOLDEN_NPZ_LOCAL,
        epochs=200,
        batch_size=64,
        alpha=1.0,
    )

    # Phase 2: continue on ALL, still measuring forgettability on GOLDEN
    phase2_info = run_train2(
        all_npz_local=ALL_NPZ_LOCAL,
        phase1_info=phase1_info,
        epochs=200,
        batch_size=64,
        alpha=1.0,
    )

    print("\n=== DONE ===")
    print("Phase 2 final loss:", phase2_info["all_loss"])
    print("Phase 2 final acc :", phase2_info["all_acc"])
