import pandas as pd

label_map = {
    # Rally side (up)
    "Rally1": 1,
    "Centroid_L_H": 2,
    "Rally2": 3,
    "ZZH3": 4,
    "ZZH2": 5,
    "ZZH1": 6,
    "ZZH0": 7,
    "ZZH-1": 8,
    "ZZH-2": 9,
    "ZZH-3": 10,

    # Drop side (down)
    "Drop1": -1,
    "Centroid_H_L": -2,
    "Drop2": -3,
    "ZZL3": -4,
    "ZZL2": -5,
    "ZZL1": -6,
    "ZZL0": -7,
    "ZZL-1": -8,
    "ZZL-2": -9,
    "ZZL-3": -10,
}




def generate_ZZ_labels(df, Symbol, TimeFrame, ZigZag_value, deviation, backstep):
    print("inside generate_ZZ_labels")
    df = df.copy()

    zz_high_col   = f"{Symbol}_ZigZag_High_{TimeFrame}_{ZigZag_value}"
    zz_low_col    = f"{Symbol}_ZigZag_Low_{TimeFrame}_{ZigZag_value}"
    label_col     = f"{Symbol}@{TimeFrame}↦ZigZagLabel[0]"
    num_label_col = f"{Symbol}@{TimeFrame}↦ZigZagNumericLabel[0]"

    # sanity check
    if zz_high_col not in df or zz_low_col not in df:
        df[label_col] = None
        df[num_label_col] = None
        return df

    df[label_col] = None
    idx = df.index
    n   = len(df)

    # ---------------------------------------------------------
    # PIVOTS
    # ---------------------------------------------------------
    is_low  = df[zz_low_col].notna().to_numpy()
    is_high = df[zz_high_col].notna().to_numpy()

    low_pos  = is_low.nonzero()[0]
    high_pos = is_high.nonzero()[0]

    windows = []

    # ---------------------------------------------------------
    # SPREAD WINDOW LABELS
    # ---------------------------------------------------------
    def spread(positions, base):
        for pos in positions:
            start = max(0, pos - 3)
            end   = min(n-1, pos + 3)
            windows.append({"start": start, "end": end, "center": pos, "base": base})

            for off in range(-3, 4):
                p = pos + off
                if p < 0 or p >= n:
                    continue

                row = idx[p]

                if off == 0:
                    suffix = "0"
                elif off < 0:
                    suffix = str(-off)
                else:
                    suffix = f"-{off}"

                token = f"{base}{suffix}"

                if off == 0:
                    df.at[row, label_col] = token
                else:
                    # only overwrite if empty
                    if df.at[row, label_col] is None or pd.isna(df.at[row, label_col]):
                        df.at[row, label_col] = token

    spread(low_pos,  "ZZL")
    spread(high_pos, "ZZH")

    # ---------------------------------------------------------
    # PLACE DIRECTIONAL CENTROIDS
    # ---------------------------------------------------------
    windows.sort(key=lambda w: w["center"])

    for k in range(len(windows)-1):
        w1 = windows[k]
        w2 = windows[k+1]

        gap_start = w1["end"] + 1
        gap_end   = w2["start"] - 1
        if gap_start > gap_end:
            continue

        mid = (gap_start + gap_end) // 2
        row = idx[mid]

        if   w1["base"] == "ZZL" and w2["base"] == "ZZH":
            df.at[row, label_col] = "Centroid_L_H"
        elif w1["base"] == "ZZH" and w2["base"] == "ZZL":
            df.at[row, label_col] = "Centroid_H_L"
        else:
            df.at[row, label_col] = "Centroid"

    # ---------------------------------------------------------
    # NEW MOVEMENT CLASSES (Rally/Drop)
    # ---------------------------------------------------------
    for i in range(1, n):
        prev_label = df.at[idx[i - 1], label_col]
        curr_label = df.at[idx[i], label_col]

        # normalize None/NaN
        if pd.isna(prev_label):
            prev_label = None
        if pd.isna(curr_label):
            curr_label = None

        # Only act if they are strings
        if isinstance(prev_label, str) and isinstance(curr_label, str):

            # Rally1: ZZL → Centroid_L_H
            if prev_label.startswith("ZZL") and curr_label == "Centroid_L_H":
                df.at[idx[i], label_col] = "Rally1"

            # Rally2: Centroid_L_H → ZZH
            if prev_label == "Centroid_L_H" and curr_label.startswith("ZZH"):
                df.at[idx[i], label_col] = "Rally2"

            # Drop1: ZZH → Centroid_H_L
            if prev_label.startswith("ZZH") and curr_label == "Centroid_H_L":
                df.at[idx[i], label_col] = "Drop1"

            # Drop2: Centroid_H_L → ZZL
            if prev_label == "Centroid_H_L" and curr_label.startswith("ZZL"):
                df.at[idx[i], label_col] = "Drop2"

    # ---------------------------------------------------------
    # FILL GAPS WITH YOUR 4 REGION LABELS
    # ---------------------------------------------------------
    labels = df[label_col].to_numpy()

    # nearest previous "significant" label
    prev_lab = [None] * n
    last = None
    for i in range(n):
        lab = labels[i]
        if isinstance(lab, str) and lab not in ("", "0"):
            last = lab
        prev_lab[i] = last

    # nearest next "significant" label
    next_lab = [None] * n
    nxt = None
    for i in range(n - 1, -1, -1):
        lab = labels[i]
        if isinstance(lab, str) and lab not in ("", "0"):
            nxt = lab
        next_lab[i] = nxt

    # assign Rally1/Rally2/Drop1/Drop2 to the in-between zones
    for i in range(n):
        # only fill where there is currently no label (or "0")
        if labels[i] is None or (isinstance(labels[i], str) and labels[i] in ("", "0")):
            pl = prev_lab[i]
            nl = next_lab[i]

            if pl == "ZZH-3"        and nl == "Centroid_H_L":
                labels[i] = "Drop1"    # numeric -1
            elif pl == "Centroid_H_L" and nl == "ZZL3":
                labels[i] = "Drop2"    # numeric -3
            elif pl == "ZZL-3"        and nl == "Centroid_L_H":
                labels[i] = "Rally1"   # numeric  1
            elif pl == "Centroid_L_H" and nl == "ZZH3":
                labels[i] = "Rally2"   # numeric  3

    df[label_col] = labels

    # ---------------------------------------------------------
    # NUMERIC MAPPING  (UNCHANGED)
    # ---------------------------------------------------------


    df[num_label_col] = df[label_col].map(label_map)

    print("✔ generate_ZZ_labels complete with rally/drop classes + filled regions")
    return df
