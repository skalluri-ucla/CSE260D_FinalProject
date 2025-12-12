


















def Detect_Golden_Representative_Data(
        label_df,
        label_col=None,
        min_rallies=5,
        min_drops=5
):
    """
    Simple 'golden' data detector:

    - Start from the full label series (with DateTime index).
    - Split into blocks of consecutive NON-ZERO labels (zeros are separators).
    - A block is kept only if:
        * it has at least `min_rallies` positive labels (> 0)
        * AND at least `min_drops` negative labels (< 0)
    - Inside each kept block:
        * keep only the FIRST occurrence of each label value
          (so you don't get many -1 or -3 in a row; just one of each).
    - Zeros are always dropped.
    - Result keeps global time order.
    """

    if label_col is None:
        label_col = label_df.columns[0]

    labels = label_df[label_col].to_numpy()
    N = len(labels)

    kept_indices = []

    i = 0
    while i < N:
        v = labels[i]

        # skip zeros and NaNs – they are always "bad"
        if v == 0 or np.isnan(v):
            i += 1
            continue

        # start of a non-zero block
        block_start = i
        while i < N and labels[i] != 0 and not np.isnan(labels[i]):
            i += 1
        block_end = i - 1

        # this is one contiguous non-zero block [block_start, block_end]
        block_vals = labels[block_start:block_end + 1]

        # count rallies/drops in this block
        num_rallies = np.sum(block_vals > 0)
        num_drops = np.sum(block_vals < 0)

        # drop blocks that don't have enough structure
        if num_rallies < min_rallies or num_drops < min_drops:
            continue

        # inside this good block: keep ONLY the first occurrence of each label
        seen = set()
        for j in range(block_start, block_end + 1):
            val = labels[j]
            if val in seen:
                # already saw this label value in this block → skip
                continue
            seen.add(val)
            kept_indices.append(j)

    if not kept_indices:
        print("[Golden] No golden data found (all blocks dropped).")
        return label_df.iloc[0:0]

    kept_indices = np.array(kept_indices, dtype=int)
    kept_indices.sort()   # preserve chronological order across blocks

    final_df = label_df.iloc[kept_indices]

    print(f"[Golden] Final golden rows: {len(final_df)}")
    print("Label distribution:", final_df[label_col].value_counts().to_dict())

    return final_df




# -------------------------------------------------------------
#  MAIN EXECUTION
# -------------------------------------------------------------
def generate_classification_sequences_main():

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

    # Extract main folder path
    main_folder = os.path.dirname(label_path)

    # ---------------- LOAD DATA ----------------
    labels_df_raw = pd.read_csv(label_path, parse_dates=["DateTime"])
    actual_label_col = [c for c in labels_df_raw.columns
                        if "ZigZagNumericLabel" in c][0]

    labels_df = labels_df_raw.set_index("DateTime")[[actual_label_col]]
    core1_df  = pd.read_csv(core1_path, parse_dates=["DateTime"], index_col="DateTime")

    # ---------------- GOLDEN POINT DETECTION ----------------
    golden_times = Detect_Golden_Representative_Data(labels_df)

    # Normalize golden index (works if function returns Series, DataFrame, or Index)
    if isinstance(golden_times, pd.DataFrame) or isinstance(golden_times, pd.Series):
        golden_index = golden_times.index
    else:
        # e.g. list/array of datetimes
        golden_index = pd.Index(golden_times)

    golden_df = labels_df.loc[golden_index]

    print("\nDetected Golden Points:")
    print(golden_df.head())

    # ---------------- SAVE GOLDEN TO EXCEL ----------------
    golden_excel_path = os.path.join(main_folder, "GoldenPoints.xlsx")
    golden_df.to_excel(golden_excel_path)
    print(f"\nGolden points saved → {golden_excel_path}\n")






