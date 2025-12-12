# MWPS_streaming.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# If you prefer, you can move these helpers into a shared utils file instead
# and import from there. For now I keep them here to avoid circular imports.

def encode_labels(y, label2idx):
    """Map raw numeric labels → integer class indices."""
    return np.array([label2idx[lab] for lab in y], dtype=np.int32)


def make_class_weights(y_idx, num_classes):
    """
    Compute class weights safely:
    - Only use classes that appear in y_idx for sklearn.
    - Ensure every class 0..num_classes-1 has a weight (default 1.0 if missing).
    """
    from sklearn.utils.class_weight import compute_class_weight

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


def make_distance_aware_loss(label_values, alpha: float = 1.0):
    label_values = tf.constant(label_values, dtype=tf.float32)  # (C,)

    base_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction="none",  # per-sample loss
    )

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)  # (batch,)
        ce = base_loss_obj(y_true, y_pred)  # (batch,)

        true_vals = tf.gather(label_values, y_true)  # (batch,)
        pred_vals = tf.reduce_sum(y_pred * label_values, axis=-1)  # (batch,)

        dist = tf.abs(true_vals - pred_vals)  # (batch,)

        return ce + alpha * dist

    return loss_fn


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


class NPZBatchSequence(tf.keras.utils.Sequence):
    """
    Streams batches from a huge 5D NPZ array without loading all into RAM.

    X_5d: memmapped array (N, T, 1, F, 1)
    valid_idx: indices into original NPZ where label != 0
    y_idx_valid: class indices aligned with valid_idx
    sample_ids: indices into valid_idx (logical sample IDs)
    """

    def __init__(self, X_5d, valid_idx, y_idx_valid, sample_ids, batch_size=64, shuffle=True):
        self.X_5d = X_5d                  # memmap or ndarray
        self.valid_idx = valid_idx        # shape (N_valid,)
        self.y_idx_valid = y_idx_valid    # shape (N_valid,)
        self.sample_ids = np.array(sample_ids, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.sample_ids) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end   = min((idx + 1) * self.batch_size, len(self.sample_ids))

        batch_ids = self.sample_ids[start:end]        # indices into valid_idx
        rows = self.valid_idx[batch_ids]              # original NPZ row indices

        # (B, T, 1, F, 1) → (B, T, F)
        X_batch = self.X_5d[rows, :, 0, :, 0].astype(np.float32)
        y_batch = self.y_idx_valid[batch_ids]

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sample_ids)


def continue_training_on_all_data(
    model_path,
    all_npz_path,
    label_map_path,
    save_path_all,
    epochs=80,
    alpha=1.0,

    batch_size=64,
):
    """
    1) Load GOLDEN-trained model.
    2) Evaluate it on ALL data (baseline) using a streaming Sequence.
    3) Continue training on ALL data (with val split) using streaming Sequences.
    4) Evaluate again.
    5) Save updated model to NEW file (save_path_all).

    NOTE: never loads the full X array into RAM.
    """

    print("\n=== CONTINUE TRAINING ON ALL DATA ===")
    print(f"[load] model → {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False,
    )

    # ---- Load NPZ in memmap mode (no 11 GB in RAM) ----
    data = np.load(all_npz_path, allow_pickle=True, mmap_mode="r")
    X_5d = data["X_low"]   # memmapped (N, 200, 1, 132, 1)
    y_raw = data["y"]
    print("[load_npz]", all_npz_path)
    print("  X.shape =", X_5d.shape)
    print("  y.shape =", y_raw.shape)

    # Extract numeric labels
    y_raw = extract_labels(y_raw)

    # Drop label 0 *only via indices*, no X_5d slicing
    mask = (y_raw != 0.0)
    num_dropped = np.count_nonzero(~mask)
    if num_dropped > 0:
        print(f"[filter] dropping {num_dropped} / {len(y_raw)} samples with label == 0.0")
    else:
        print("[filter] no label == 0.0 found.")

    # valid_idx maps "logical valid samples" → original NPZ rows
    valid_idx = np.where(mask)[0]
    y_raw_valid = y_raw[valid_idx]

    # Load label map and encode labels for valid samples
    print(f"[load] label2idx → {label_map_path}")
    label2idx = np.load(label_map_path, allow_pickle=True).item()

    y_idx_valid = encode_labels(y_raw_valid, label2idx)

    idx2label = {i: lab for lab, i in label2idx.items()}
    label_values = np.array(
        [idx2label[i] for i in range(len(label2idx))],
        dtype=np.float32,
    )

    loss_fn = make_distance_aware_loss(label_values, alpha=alpha)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_acc")],
    )

    # Time/feature dims for sanity check (no big array)
    _, time_steps, _, n_features, _ = X_5d.shape
    print(f"[info] time_steps={time_steps}, n_features={n_features}, num_classes={len(label2idx)}")

    # ---- Build "logical" sample IDs and train/val split ----
    sample_ids = np.arange(len(valid_idx), dtype=np.int64)

    train_ids, val_ids, y_train, y_val = train_test_split(
        sample_ids,
        y_idx_valid,
        test_size=0.2,
        random_state=42,
        stratify=y_idx_valid,
    )

    # Class weights from y_train
    class_weights = make_class_weights(y_train, num_classes=len(label2idx))
    print("class_weights:", class_weights)

    # ---- Build Sequences ----
    train_seq = NPZBatchSequence(
        X_5d=X_5d,
        valid_idx=valid_idx,
        y_idx_valid=y_idx_valid,
        sample_ids=train_ids,
        batch_size=batch_size,
        shuffle=True,
    )

    val_seq = NPZBatchSequence(
        X_5d=X_5d,
        valid_idx=valid_idx,
        y_idx_valid=y_idx_valid,
        sample_ids=val_ids,
        batch_size=batch_size,
        shuffle=False,
    )

    full_seq = NPZBatchSequence(
        X_5d=X_5d,
        valid_idx=valid_idx,
        y_idx_valid=y_idx_valid,
        sample_ids=sample_ids,   # all valid samples
        batch_size=batch_size,
        shuffle=False,
    )

    # ---- Baseline evaluation on ALL data (before continuation) ----
    print("\n--- BASELINE EVALUATION ON ALL DATA (BEFORE CONTINUATION) ---")
    base_loss, base_acc = model.evaluate(full_seq, verbose=0)
    print(f"BASELINE ALL → loss={base_loss:.4f}, acc={base_acc:.4f}")

    # ---- Callbacks ----
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    print("\n=== TRAINING ON ALL DATA (STREAMING) ===")
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    print("\n=== FINAL EVALUATION ON ALL DATA (AFTER CONTINUATION) ===")
    all_loss, all_acc = model.evaluate(full_seq, verbose=0)
    print(f"ALL (AFTER) → loss={all_loss:.4f}, acc={all_acc:.4f}")

    model.save(save_path_all)
    print(f"[save] Updated ALL-trained model → {save_path_all}")

    return float(all_loss), float(all_acc)
