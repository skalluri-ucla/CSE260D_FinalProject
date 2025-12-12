

import os
import numpy as np
import pandas as pd












'''
    # ------------------ PATHS -----------------------
    core1_path = (
        r"D:\ReViAI\Trained models\MWPS\epochs 500\EURUSD\M15"
        r"\MWPS EURUSD M15 e.500 V.2025-07-03-04-25"
        r"\MWPSDATA_EURUSD_M15_O21uv1_Core_1_df.csv"
    )
    label_path = (
        r"D:\ReViAI\Trained models\MWPS\epochs 500\EURUSD\M15"
        r"\MWPS EURUSD M15 e.500 V.2025-07-03-04-25"
        r"\MWPSDATA_EURUSD_M15_O21uv1_Core_target.csv"
    )
'''

core1_path = r"D:\ReViAI\Trained models\MWPS\epochs 500\EURUSD\M15\MWPS EURUSD M15 e.500 V.2025-07-03-04-25\Unseen Input_scaled_core_M5.csv"
label_path = r"D:\ReViAI\Trained models\MWPS\epochs 500\EURUSD\M15\MWPS EURUSD M15 e.500 V.2025-07-03-04-25\Unseen_scaled_Core_target.csv"
indexer_file_path = r"D:\ReViAI\Trained models\MWPS\epochs 500\EURUSD\M15\MWPS EURUSD M15 e.500 V.2025-07-03-04-25\Unseen Indexer data.csv"







# =============================================================================
# WINDOW FUNCTION
# =============================================================================

def _windows_2d(X, window):
    if len(X) < window:
        return np.empty((0, window, X.shape[1]))
    return np.squeeze(
        np.lib.stride_tricks.sliding_window_view(X, (window, X.shape[1])),
        axis=1
    )









def create_classification_sequences(
        core_df,
        labels_full_df,
        keep_times_df=None,        # <== NEW: DataFrame that defines which DateTimes to keep
        time_step=200,
        check_zero_cols=True,
        max_label_shift_bars=4,    # <= 4 bars difference allowed
        bar_minutes=5,             # 1 bar = 5 minutes
):
    """
    core_df        : M5 input (DateTime index)
    labels_full_df : M15 labels (DateTime index, 1 col)
    keep_times_df  : DataFrame whose DateTimes define which label times to keep.
                     - If None  -> keep ALL valid sequences
                     - If not None:
                        * If its index is DateTimeIndex -> use index
                        * Else if it has 'DateTime' column -> use that column
                        * Other columns are ignored

    Returns:
        X_5d      : (N, time_step, 1, F, 1)
        y_labels  : (N,)
        t_seq     : (N,) np.datetime64 end time of each INPUT window
    """

    # ---- 0) Resolve keep_times ----
    keep_times = None
    if keep_times_df is not None:
        df_keep = keep_times_df.copy()

        if isinstance(df_keep.index, pd.DatetimeIndex):
            keep_times = df_keep.index.values.astype('datetime64[ns]')
        elif 'DateTime' in df_keep.columns:
            dt_col = pd.to_datetime(df_keep['DateTime'])
            keep_times = dt_col.values.astype('datetime64[ns]')
        else:
            raise ValueError(
                "keep_times_df must have a DatetimeIndex or a 'DateTime' column."
            )

    # ---- 1) basic cleanup for labels ----
    if isinstance(labels_full_df, pd.Series):
        labels_full_df = labels_full_df.to_frame()

    label_col = labels_full_df.columns[0]

    c1     = core_df.dropna()
    labels = labels_full_df.dropna()

    idx = c1.index.values              # core times (M5)
    idy = labels.index.values          # label times (M15)

    X = c1.to_numpy()                  # (Nc, F)
    Y = labels[label_col].to_numpy()   # (Nl,)

    tL = int(time_step)

    # ---- 2) sliding windows on CORE ----
    low_windows = _windows_2d(X, tL)             # (Nwin, tL, F)
    n_win       = low_windows.shape[0]

    # end index for each window in core index
    end_pos   = np.arange(tL - 1, tL - 1 + n_win)
    end_times = idx[end_pos]                     # (Nwin,) end time of each window (M5 index)

    # ---- 3) no-NaN mask ----
    ok_nan = ~np.isnan(low_windows).any(axis=(1, 2))   # (Nwin,)

    # ---- 4) zero-column mask (if needed) ----
    if check_zero_cols:
        cols_keep = [
            i for i, col in enumerate(c1.columns)
            if ("RSI" not in col and "ZigZagFlag" not in col)
        ]
        if cols_keep:
            ok_zero = (low_windows[:, :, cols_keep] != 0).all(axis=(1, 2))
        else:
            ok_zero = np.ones(n_win, dtype=bool)
    else:
        ok_zero = np.ones(n_win, dtype=bool)

    # ---- 5) LABEL ALIGNMENT with ±4-bar tolerance ----
    max_delta = np.timedelta64(max_label_shift_bars * bar_minutes, 'm')

    pos_right = np.searchsorted(idy, end_times, side="right")
    prev_idx  = pos_right - 1
    next_idx  = pos_right

    prev_valid = (prev_idx >= 0)
    next_valid = (next_idx < len(idy))

    big_td = np.timedelta64(10**9, 'm')

    dt_prev = np.full(n_win, big_td, dtype='timedelta64[m]')
    dt_next = np.full(n_win, big_td, dtype='timedelta64[m]')

    dt_prev[prev_valid] = np.abs(end_times[prev_valid] - idy[prev_idx[prev_valid]])
    dt_next[next_valid] = np.abs(end_times[next_valid] - idy[next_idx[next_valid]])

    use_prev   = dt_prev <= dt_next
    chosen_idx = np.where(use_prev, prev_idx, next_idx)
    chosen_dt  = np.where(use_prev, dt_prev, dt_next)

    chosen_valid = (prev_valid | next_valid) & (chosen_dt <= max_delta)





    # label for each window (0 if no valid label)
    y_seq = np.zeros((n_win,), dtype=Y.dtype)
    y_seq[chosen_valid] = Y[chosen_idx[chosen_valid]]

    # label time for each window
    label_times_for_windows = np.full(n_win, np.datetime64('NaT'), dtype='datetime64[ns]')
    label_times_for_windows[chosen_valid] = idy[chosen_idx[chosen_valid]]

    ok_y = chosen_valid

    # ---- 6) FILTER BY keep_times_df (if provided) ----
    if keep_times is not None:
        in_keep = np.isin(label_times_for_windows, keep_times)
    else:
        in_keep = np.ones(n_win, dtype=bool)

    # 🔴 NEW: drop any window whose final label == 0
    non_zero_label = (y_seq != 0)

    # ---- 7) combine all masks ----
    ok_all = ok_nan & ok_zero & ok_y & in_keep & non_zero_label






    low_final = low_windows[ok_all]        # (N_keep, tL, F)
    y_final   = y_seq[ok_all]              # (N_keep,)
    t_final   = end_times[ok_all]          # (N_keep,) from INPUT index

    if len(low_final) == 0:
        print("[classification] No valid sequences after alignment.")
        return (
            np.empty((0, tL, 1, X.shape[1], 1)),
            np.empty((0,), dtype=Y.dtype),
            np.empty((0,), dtype='datetime64[ns]'),
        )

    # ---- 8) reshape for ConvLSTM: (N, tL, 1, F, 1) ----
    low_5d = np.expand_dims(low_final[:, :, :, None], axis=2)

    y_labels = y_final.squeeze()
    if y_labels.ndim != 1:
        y_labels = y_labels.reshape(-1)

    print(f"[classification] Candidates={low_windows.shape[0]}  Kept={len(low_final)}")
    print("[classification] Unique labels:", np.unique(y_labels))

    # Return INPUT-based time index (for matching)
    return low_5d, y_labels, t_final











import os
import numpy as np
import pandas as pd


def generate_classification_sequences_main(indexer_file_path):
    """
    Creates classification sequences for ANY list of DateTimes provided
    inside the indexer CSV.

    Output NPZ will be saved with the SAME BASE NAME as the indexer file.

    Example:
        indexer_file_path = ".../GoldenPoints.csv"
        → output NPZ = ".../GoldenPoints.npz"
    """

    # ------------------ PREP -----------------------
    # Extract indexer base name without extension
    seq_name = os.path.splitext(os.path.basename(indexer_file_path))[0]

    print(f"\n=== Generating sequences for: {seq_name} ===")

    # Load indexer CSV (must contain a DateTime column or be a simple list)
    indexer_df = pd.read_csv(indexer_file_path)
    print(f"indexer_df = \n{indexer_df.head()}")
    print(f"[info] indexer_df rows: {len(indexer_df)}")

    # Normalize times based on structure
    if "DateTime" in indexer_df.columns:
        index_times = pd.to_datetime(indexer_df["DateTime"])
    else:
        # If no DateTime column, assume index contains the timestamps
        index_times = pd.to_datetime(indexer_df.index)

    # Clean: drop duplicates and sort
    index_times = index_times.drop_duplicates().sort_values()
    print(f"[info] unique requested DateTimes: {len(index_times)}")



    # Extract folder
    main_folder = os.path.dirname(label_path)

    # ---------------- LOAD CORE AND LABELS ----------------
    labels_df_raw = pd.read_csv(label_path, parse_dates=["DateTime"])
    label_col = [c for c in labels_df_raw.columns if "ZigZagNumericLabel" in c][0]

    labels_df = labels_df_raw.set_index("DateTime")[[label_col]]
    core_df = pd.read_csv(core1_path, parse_dates=["DateTime"], index_col="DateTime")

    print(f"[info] labels_df rows: {len(labels_df)}")
    print(f"[info] core_df rows:   {len(core_df)}")

    # ---------------- FILTER LABELS BY INDEXER (ROBUST) ----------------
    # Use reindex so missing times don't crash the code.
    selected_labels = labels_df.reindex(index_times)

    # Track missing vs found
    missing_mask = selected_labels[label_col].isna()
    num_missing = missing_mask.sum()
    num_found = len(selected_labels) - num_missing

    print(f"[info] requested times: {len(index_times)}")
    print(f"[info] matched in labels: {num_found}")
    print(f"[info] missing in labels: {num_missing}")

    if num_missing > 0:
        # Optional: show a few missing times for debugging
        missing_times_preview = index_times[missing_mask.values][:10]
        print("[warn] Example missing times (up to 10):")
        print(missing_times_preview)

    # Drop rows where no matching label exists
    selected_labels = selected_labels.dropna()

    if selected_labels.empty:
        raise ValueError(
            "No DateTimes from the indexer exist in the labels_df index. "
            "Nothing to build sequences from."
        )

    print("\nDetected Selected Points (first 5):")
    print(selected_labels.head())

    # Save selected points for reference
    excel_path = os.path.join(main_folder, f"{seq_name}_SelectedPoints.xlsx")
    selected_labels.to_excel(excel_path)
    print(f"\nSaved selected points → {excel_path}")

    # ---------------- CREATE SEQUENCES ----------------
    X_seq, y_seq, t_seq = create_classification_sequences(
        core_df=core_df,
        labels_full_df=labels_df,
        keep_times_df=selected_labels,
        time_step=200,
        check_zero_cols=True,
    )

    # ---------------- SAVE NPZ WITH INDEXER NAME ----------------
    out_npz_path = os.path.join(main_folder, f"{seq_name}.npz")

    np.savez_compressed(
        out_npz_path,
        X_low=X_seq,
        y=y_seq,
        t_seq=t_seq,
    )

    print(f"\nSaved sequence NPZ → {out_npz_path}")

    # ---------------- SUMMARY ----------------
    print("\n=== SUMMARY ===")
    print("X_low.shape =", X_seq.shape)
    print("y.shape     =", y_seq.shape)
    print("t_seq.shape =", t_seq.shape)
    print("\nDone.\n")









# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    print("=== Classification Sequence Generator ===")

    # ---------------- GOLDEN POINT DETECTION ----------------
    # golden_times = Detect_Golden_Representative_Data(labels_df)



    try:
        import tensorflow as tf
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    except:
        print("TensorFlow not available.")

    try:
        generate_classification_sequences_main(indexer_file_path)
        print("=== DONE: Classification sequences generated successfully ===")
    except Exception as e:
        print("ERROR while generating classification sequences:")
        print(e)


