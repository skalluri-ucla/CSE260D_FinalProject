"""
MWPS Heavy Classifier Training (ONE-PHASE, ALL vs GOLDEN, streaming)

- Train on ALL NPZ (streaming from memmap)
    * ALL dataset: MWPSDATA_EURUSD_M15_O21uv1_classification_sequences.npz
- Evaluate forgettability on GOLDEN after every epoch
    * GOLDEN: classification_sequences_GOLDEN.npz

Artifacts:
    * Forgettability on GOLDEN          → forgettability_ALL_{core}.xlsx
    * Model                             → Model_train_ALL_{core}.keras
    * History                           → hist_train_ALL_{core}.xlsx
    * label2idx map                     → label_map_ALL_{core}.npy

Also:
    * Copies this script + Ai_Model_codes.py into /tmp/mwps_cache for reproducibility
"""

import os
import shutil
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.utils import Sequence
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from model_generation.Model import build_mwps_heavy_classifier


# =============================================================================
# PATH / CONFIG
# =============================================================================

# BASE_DIR where NPZs live (same pattern as your 2-phase script)
if os.name == "nt":
    BASE_DIR = r"D:\ReViAI\Trained models\MWPS\epochs 500\EURUSD\M15\MWPS EURUSD M15 e.500 V.2025-07-03-04-25"
else:
    BASE_DIR = "/mnt/3090D/ReViAI/Trained models/MWPS/epochs 500\\EURUSD\\M15\\MWPS EURUSD M15 e.500 V.2025-07-03-04-25".replace(
        "\\", "/"
    )

GOLDEN_NPZ_PATH = os.path.join(BASE_DIR, "classification_sequences_GOLDEN.npz")
ALL_NPZ_PATH    = os.path.join(BASE_DIR, "MWPSDATA_EURUSD_M15_O21uv1_classification_sequences.npz")

print("BASE_DIR =", BASE_DIR)
print("GOLDEN_NPZ_PATH =", GOLDEN_NPZ_PATH)
print("ALL_NPZ_PATH    =", ALL_NPZ_PATH)

# Core name from ALL npz
ALL_CORE_NAME = os.path.splitext(os.path.basename(ALL_NPZ_PATH))[0]

# --------- LOCAL CACHE ROOT FOR EXPERIMENT ARTIFACTS ---------
if os.name == "nt":
    LOCAL_CACHE_ROOT = r"C:\MWPS_cache"
else:
    LOCAL_CACHE_ROOT = "/home/revi/MWPS_cache"

RUN_PREFIX = f"EXP_Heavy_ALL_{ALL_CORE_NAME}"
EXPERIMENT_DIR = os.path.join(LOCAL_CACHE_ROOT, RUN_PREFIX)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

print("EXPERIMENT_DIR =", EXPERIMENT_DIR)

# --------- /tmp script cache for reproducibility ---------
CACHE_ROOT = "/tmp/mwps_cache"
os.makedirs(CACHE_ROOT, exist_ok=True)

RUN_NAME = RUN_PREFIX  # reuse


# =============================================================================
# SMALL UTILS
# =============================================================================

def get_timestamp_str() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def copy_script_to_cache(extra_files=None):
    """
    Copy this training script (and optional extra files)
    into a timestamped folder under /tmp/mwps_cache/RUN_NAME/.
    """
    extra_files = extra_files or []
    this_file = os.path.abspath(__file__)
    ts = get_timestamp_str()

    run_cache_dir = os.path.join(CACHE_ROOT, RUN_NAME, f"run_{ts}")
    os.makedirs(run_cache_dir, exist_ok=True)

    # This script
    dst_script = os.path.join(run_cache_dir, os.path.basename(this_file))
    shutil.copy2(this_file, dst_script)

    # Extra files (e.g., Ai_Model_codes.py)
    for f in extra_files:
        if not f:
            continue
        f = os.path.abspath(f)
        if os.path.isfile(f):
            dst = os.path.join(run_cache_dir, os.path.basename(f))
            shutil.copy2(f, dst)

    print(f"[CACHE] Copied script(s) to: {run_cache_dir}")


# =============================================================================
# CUSTOM LOSS: DISTANCE-AWARE
# =============================================================================

def make_distance_aware_loss(label_values, alpha: float = 1.0):
    """
    label_values: numeric label values indexed by class index (0..C-1).
    alpha: weight of distance penalty term.
    """
    label_values = tf.constant(label_values, dtype=tf.float32)  # (C,)

    base_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction="none",
    )

    def loss_fn(y_true, y_pred):
        # y_true: int indices, y_pred: softmax probs
        y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        ce = base_loss_obj(y_true_flat, y_pred)  # (batch,)

        true_vals = tf.gather(label_values, y_true_flat)           # (batch,)
        pred_vals = tf.reduce_sum(y_pred * label_values, axis=-1)  # (batch,)
        dist = tf.abs(true_vals - pred_vals)

        return ce + alpha * dist

    return loss_fn


# =============================================================================
# FORGETTABILITY CALLBACK
# =============================================================================

class ForgettabilityCallback(tf.keras.callbacks.Callback):
    """
    After each epoch, evaluate model on full GOLDEN set (X_full, y_full)
    and write correctness (0/1) to an Excel file:

        index: t_seq (DateTimeIndex or RangeIndex)
        columns: epoch_001, epoch_002, ...
    """

    def __init__(self, X_full, y_full, t_seq, out_xlsx_path, batch_size=1024):
        super().__init__()
        self.X_full = X_full
        self.y_full = y_full
        self.t_seq = t_seq
        self.out_xlsx_path = out_xlsx_path
        self.batch_size = batch_size

        try:
            self.index = pd.to_datetime(self.t_seq)
        except Exception:
            self.index = pd.RangeIndex(len(self.t_seq))

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        col_name = f"epoch_{epoch_num:03d}"
        print(f"\n[Forgettability] Evaluating GOLDEN on epoch {epoch_num}...")

        y_pred = self.model.predict(
            self.X_full,
            batch_size=self.batch_size,
            verbose=0,
        )
        y_pred_idx = np.argmax(y_pred, axis=1)

        correct = (y_pred_idx == self.y_full).astype(int)

        if os.path.exists(self.out_xlsx_path):
            df = pd.read_excel(self.out_xlsx_path, index_col=0)
            if len(df) != len(self.index):
                print("[Forgettability] index length mismatch; recreating DataFrame.")
                df = pd.DataFrame(index=self.index)
        else:
            df = pd.DataFrame(index=self.index)

        df[col_name] = correct
        df.to_excel(self.out_xlsx_path)
        print(f"[Forgettability] Saved correctness for epoch {epoch_num} → {self.out_xlsx_path}")


# =============================================================================
# DATA HELPERS
# =============================================================================

def extract_labels(y_raw):
    """
    Ensure we get a 1D numeric label vector out of y_raw.
    Mirrors your previous helper.
    """
    print("=== DEBUG y_raw ===")
    print("  y_raw.shape:", y_raw.shape)
    print("  y_raw.dtype:", y_raw.dtype)

    # Structured dtype
    if isinstance(y_raw, np.ndarray) and y_raw.dtype.names is not None:
        print("  y_raw has named fields:", y_raw.dtype.names)
        labels = y_raw["label"]  # adjust if field name is different
    # 2D array -> last column as label
    elif y_raw.ndim == 2:
        print("  y_raw is 2D, assuming last column is label")
        labels = y_raw[:, -1]
    else:
        labels = y_raw

    print("First 20 labels:", labels[:20])
    return labels.astype(np.float32)


def build_label_mapping(y_all):
    """
    Build label2idx / idx2label from numeric labels (no 0s).
    """
    unique_labels = np.unique(y_all)
    unique_labels = unique_labels.tolist()

    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}

    print("Label2Idx:", label2idx)
    print("num_classes =", len(unique_labels))

    return label2idx, idx2label, len(unique_labels)


def encode_labels(y, label2idx):
    return np.array([label2idx[lab] for lab in y], dtype=np.int32)


def make_class_weights(y_idx, num_classes):
    """
    Balanced class weights; ensure every class index has a value.
    """
    classes_present = np.unique(y_idx)
    print("  classes_present for weights:", classes_present)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_present,
        y=y_idx,
    )
    class_weights = {int(c): float(w) for c, w in zip(classes_present, weights)}

    for c in range(num_classes):
        class_weights.setdefault(c, 1.0)

    return class_weights


def load_npz_sequences(npz_path):
    """
    For GOLDEN only (RAM): X_low 5D, y, t_seq.
    """
    data = np.load(npz_path, allow_pickle=True)
    X_5d = data["X_low"]
    y_raw = data["y"]
    t_seq = data["t_seq"] if "t_seq" in data.files else np.arange(len(y_raw))

    print(f"[load_npz] {npz_path}")
    print("  X_5d.shape:", X_5d.shape)
    print("  y_raw.shape:", y_raw.shape)
    print("  t_seq.shape:", t_seq.shape)

    return X_5d, y_raw, t_seq


def prepare_dataset(npz_path, label2idx, idx2label):
    """
    Prepare GOLDEN dataset in RAM, using existing label2idx / idx2label mapping.
    - Drop label==0.0
    - Reshape to (N, T, F)
    - Encode labels
    """
    X_5d, y_raw, t_seq = load_npz_sequences(npz_path)
    y_raw = extract_labels(y_raw)

    mask = (y_raw != 0.0)
    num_dropped = np.count_nonzero(~mask)
    if num_dropped > 0:
        print(f"[GOLDEN] dropping {num_dropped} / {len(y_raw)} samples with label==0.0")
        X_5d = X_5d[mask]
        y_raw = y_raw[mask]
        t_seq = t_seq[mask]
    else:
        print(f"[GOLDEN] no label==0.0; using all {len(y_raw)} samples")

    X = X_5d[:, :, 0, :, 0].astype(np.float32)
    print("[GOLDEN reshape] X 5D → 3D:", X.shape)

    # encode labels
    y_idx = encode_labels(y_raw, label2idx)

    return {
        "X": X,
        "y_idx": y_idx,
        "t_seq": t_seq,
    }


# =============================================================================
# PREPARE ALL FOR STREAMING (build mapping directly from ALL)
# =============================================================================

def prepare_all_for_streaming_build_mapping(npz_path):
    """
    For ALL dataset:
    - Load via memmap
    - Extract y_raw_all, t_seq_all
    - Drop label==0.0 using a mask → source_indices
    - Build label2idx / idx2label / label_values from used labels
    - Encode y_idx_used
    - Compute class_weights
    """
    data = np.load(npz_path, allow_pickle=True, mmap_mode="r")

    X_5d = data["X_low"]  # memmap
    y_raw_all = extract_labels(data["y"])
    t_seq_all = data["t_seq"] if "t_seq" in data.files else np.arange(len(y_raw_all))

    print(f"[ALL stream] X_5d.shape = {X_5d.shape}")
    print(f"[ALL stream] y_raw_all.shape = {y_raw_all.shape}")

    mask = (y_raw_all != 0.0)
    idx_source = np.nonzero(mask)[0]  # indices into X_5d
    y_raw_used = y_raw_all[mask]
    t_seq_used = t_seq_all[mask]

    print(f"[ALL stream] dropping {np.count_nonzero(~mask)} / {len(y_raw_all)} with label==0.0")
    print(f"[ALL stream] using {len(y_raw_used)} samples")

    # Build label mapping from ALL (used) labels
    label2idx, idx2label, num_classes = build_label_mapping(y_raw_used)
    y_idx_used = encode_labels(y_raw_used, label2idx)

    # class weights
    class_weights = make_class_weights(y_idx_used, num_classes=num_classes)

    # label_values in index order
    label_values = np.array(
        [idx2label[i] for i in range(num_classes)],
        dtype=np.float32,
    )

    time_steps = X_5d.shape[1]
    n_features = X_5d.shape[3]

    return {
        "X_memmap": X_5d,
        "source_indices": idx_source,
        "y_idx_used": y_idx_used,
        "t_seq_used": t_seq_used,
        "time_steps": time_steps,
        "n_features": n_features,
        "class_weights": class_weights,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "label_values": label_values,
        "num_classes": num_classes,
    }


# =============================================================================
# STREAMING SEQUENCE
# =============================================================================

class NPZStreamingSequence(Sequence):
    """
    Streams batches from ALL memmap.

    - X_memmap:   memmap (N, T, 1, F, 1)
    - y_idx_filtered: labels for used samples (label!=0)
    - source_indices: mapping from used positions -> original rows
    - positions: subset of used positions for this sequence
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
        self.source_indices = source_indices
        self.positions = np.array(positions)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.positions) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.positions))

        batch_pos = self.positions[start:end]
        orig_idx = self.source_indices[batch_pos]

        X_batch_5d = self.X_memmap[orig_idx, ...]  # (B, T, 1, F, 1)
        X_batch = X_batch_5d[:, :, 0, :, 0].astype(np.float32)

        y_batch = self.y_idx_filtered[batch_pos]
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.positions)


# =============================================================================
# BUILD + COMPILE HEAVY MODEL
# =============================================================================

def build_and_compile_model(time_steps, n_features, num_classes, label_values, alpha=1.0, lr=1e-4):
    model = build_mwps_heavy_classifier(
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
# CACHE NPZs TO EXPERIMENT DIR
# =============================================================================

def cache_npz_to_local(npz_path: str, dest_dir: str) -> str:
    """
    Copy NPZ into dest_dir if needed; reuse if already up-to-date.
    """
    os.makedirs(dest_dir, exist_ok=True)

    npz_path = os.path.abspath(npz_path)
    local_path = os.path.join(dest_dir, os.path.basename(npz_path))

    if os.path.abspath(npz_path) == os.path.abspath(local_path):
        print(f"[cache] NPZ already in dest_dir: {npz_path}")
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
        print(f"[cache] Copying NPZ\n  from: {npz_path}\n  to:   {local_path}")
        shutil.copy2(npz_path, local_path)
        print("[cache] Copy complete.")
    else:
        print(f"[cache] Using existing local copy: {local_path}")

    return local_path


# =============================================================================
# ONE-PHASE TRAINING: ALL (streaming) vs GOLDEN (forgettability)
# =============================================================================

def run_train_all(all_npz_local, golden_npz_local, epochs=200, batch_size=64, alpha=1.0):
    print("\n=== ONE-PHASE: Train HEAVY on ALL (streaming), forgettability on GOLDEN ===")

    # ---- Prepare ALL (build mapping + streaming) ----
    all_data = prepare_all_for_streaming_build_mapping(all_npz_local)

    X_memmap       = all_data["X_memmap"]
    source_indices = all_data["source_indices"]
    y_all_idx_used = all_data["y_idx_used"]
    t_all_used     = all_data["t_seq_used"]
    time_steps     = all_data["time_steps"]
    n_features     = all_data["n_features"]
    class_weights  = all_data["class_weights"]
    label2idx      = all_data["label2idx"]
    idx2label      = all_data["idx2label"]
    label_values   = all_data["label_values"]
    num_classes    = all_data["num_classes"]

    print(f"[ALL] filtered samples = {len(y_all_idx_used)}")

    # Save label map
    label_map_path = os.path.join(EXPERIMENT_DIR, f"label_map_ALL_{ALL_CORE_NAME}.npy")
    np.save(label_map_path, label2idx)
    print(f"[save] label2idx → {label_map_path}")

    # ---- Prepare GOLDEN in RAM using same mapping ----
    golden = prepare_dataset(golden_npz_local, label2idx=label2idx, idx2label=idx2label)
    X_g = golden["X"]
    y_g = golden["y_idx"]
    t_g = golden["t_seq"]

    # ---- Build & compile HEAVY model ----
    model, _ = build_and_compile_model(
        time_steps=time_steps,
        n_features=n_features,
        num_classes=num_classes,
        label_values=label_values,
        alpha=alpha,
        lr=1e-4,
    )

    # ---- Positions for ALL ----
    n_used = len(y_all_idx_used)
    positions = np.arange(n_used)

    # Train/val split over positions
    pos_train, pos_val = train_test_split(
        positions,
        test_size=0.2,
        random_state=42,
        stratify=y_all_idx_used,
    )

    # Class weights based on train split
    class_weights_train = make_class_weights(y_all_idx_used[pos_train], num_classes=num_classes)
    print("[ALL] class_weights_train:", class_weights_train)

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

    # ---- Callbacks ----
    forget_all_path = os.path.join(EXPERIMENT_DIR, f"forgettability_ALL_{ALL_CORE_NAME}.xlsx")
    model_all_path  = os.path.join(EXPERIMENT_DIR, f"Model_train_ALL_{ALL_CORE_NAME}.keras")
    hist_all_path   = os.path.join(EXPERIMENT_DIR, f"hist_train_ALL_{ALL_CORE_NAME}.xlsx")

    forget_cb = ForgettabilityCallback(
        X_full=X_g,
        y_full=y_g,
        t_seq=t_g,
        out_xlsx_path=forget_all_path,
        batch_size=1024,
    )

    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )

    callbacks = [es_cb, forget_cb]

    # ---- TRAIN ----
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        class_weight=class_weights_train,
        callbacks=callbacks,
        verbose=2,
    )

    # ---- Final eval on ALL (streaming) ----
    print("\n=== FINAL EVAL on ALL (streaming) ===")
    eval_positions = np.arange(n_used)
    eval_seq = NPZStreamingSequence(
        X_memmap=X_memmap,
        y_idx_filtered=y_all_idx_used,
        source_indices=source_indices,
        positions=eval_positions,
        batch_size=batch_size,
        shuffle=False,
    )
    all_loss, all_acc = model.evaluate(eval_seq, verbose=0)
    print(f"ALL → loss={all_loss:.4f}, acc={all_acc:.4f}")

    # ---- Save model + history ----
    model.save(model_all_path)
    print(f"[save] Model_train_ALL → {model_all_path}")

    hist_df = pd.DataFrame(history.history)
    hist_df.to_excel(hist_all_path, index=False)
    print(f"[save] hist_train_ALL → {hist_all_path}")

    return {
        "model_path": model_all_path,
        "all_loss": float(all_loss),
        "all_acc": float(all_acc),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=== MWPS ONE-PHASE HEAVY: ALL (streaming) vs GOLDEN (forgettability) ===")
    print("BASE_DIR       =", BASE_DIR)
    print("EXPERIMENT_DIR =", EXPERIMENT_DIR)
    print("ALL_CORE_NAME  =", ALL_CORE_NAME)

    try:
        print("TensorFlow version:", tf.__version__)
        print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
    except Exception as e:
        print("TensorFlow not fully available:", e)

    # Copy this script + Ai_Model_codes to /tmp/mwps_cache
    ai_model_codes_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Ai_Model_codes.py",
    )
    copy_script_to_cache(extra_files=[ai_model_codes_path])

    # Cache NPZs into EXPERIMENT_DIR
    GOLDEN_NPZ_LOCAL = cache_npz_to_local(GOLDEN_NPZ_PATH, EXPERIMENT_DIR)
    ALL_NPZ_LOCAL    = cache_npz_to_local(ALL_NPZ_PATH,    EXPERIMENT_DIR)

    # Run training
    phase_info = run_train_all(
        all_npz_local=ALL_NPZ_LOCAL,
        golden_npz_local=GOLDEN_NPZ_LOCAL,
        epochs=200,
        batch_size=64,
        alpha=1.0,
    )

    print("\n=== DONE (ONE-PHASE HEAVY ALL vs GOLDEN) ===")
    print("Final ALL loss:", phase_info["all_loss"])
    print("Final ALL acc :", phase_info["all_acc"])
