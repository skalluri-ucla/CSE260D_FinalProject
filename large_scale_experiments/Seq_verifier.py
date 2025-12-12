import numpy as np
import pandas as pd

def inspect_npz_contents(npz_path, max_features=20, max_times=10):
    print("\n=== Inspecting NPZ: ", npz_path)
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print("❌ ERROR loading file:", e)
        return

    # ------------------------------
    # PRINT KEYS
    # ------------------------------
    print("NPZ keys:", data.files)

    # ------------------------------
    # FEATURES
    # ------------------------------
    if "X_low" in data.files:
        X = data["X_low"]

        # X shape: (N, T, 1, F, 1)
        n_features = X.shape[3]
        print(f"\nX_low shape: {X.shape}")
        print(f"Total features: {n_features}")

        if "feature_names" in data.files:
            feature_names = data["feature_names"]
            print("Feature names (first few):")
            for i, name in enumerate(feature_names[:max_features]):
                print(f"{i}: {name}")
        else:
            print("No feature_names found. Showing placeholder names:")
            for i in range(min(n_features, max_features)):
                print(f"{i}: Feature_{i}")

    else:
        print("⚠️ X_low not found in this NPZ.")

    # ------------------------------
    # LABELS (y)
    # ------------------------------
    if "y" in data.files:
        y = data["y"]
        print(f"\nTotal labels: {len(y)}")

        unique_labels, counts = np.unique(y, return_counts=True)

        print("Label distribution:")
        for u, c in zip(unique_labels, counts):
            print(f"  {u}: {c}")

    else:
        print("⚠️ y not found in this NPZ.")

    # ------------------------------
    # TIME SEQUENCE (t_seq)
    # ------------------------------
    if "t_seq" in data.files:
        t_seq = data["t_seq"]
        print(f"\nTime sequence length: {len(t_seq)}")

        print("First timestamps:")
        for i in range(min(max_times, len(t_seq))):
            print(f"{i}: {pd.to_datetime(t_seq[i])}")
    else:
        print("⚠️ t_seq not found in this NPZ.")





def verify_sample(sample_idx, path_golden, path_all, show_rows=10):
    """
    Compare one window between GOLDEN and ALL datasets.
    If t_seq exists in both files, also match by datetime.
    """

    data_g = np.load(path_golden, allow_pickle=True)
    data_a = np.load(path_all,    allow_pickle=True)

    print("GOLDEN keys:", data_g.files)
    print("ALL    keys:", data_a.files)

    Xg = data_g["X_low"]
    yg = data_g["y"]

    Xa = data_a["X_low"]
    ya = data_a["y"]

    if sample_idx >= Xg.shape[0]:
        print(f"sample_idx {sample_idx} out of range for GOLDEN (N={Xg.shape[0]})")
        return

    # ---- if t_seq is missing, just show GOLDEN and stop ----
    if "t_seq" not in data_g or "t_seq" not in data_a:
        print("\n[t_seq] is not present in one or both npz files.")
        print("You need to regenerate the npz with t_seq saving enabled "
              "to compare by DateTime.\n")

        xg_2d = Xg[sample_idx][:, 0, :, 0]
        print(f"GOLDEN sample {sample_idx} label =", yg[sample_idx])
        print("First rows of GOLDEN window:")
        print(pd.DataFrame(xg_2d).head(show_rows))
        return

    # ---- full comparison (with datetime) ----
    tg = data_g["t_seq"]
    ta = data_a["t_seq"]

    t_g = pd.to_datetime(tg[sample_idx])

    # locate same datetime in ALL
    idx_all = np.where(ta == tg[sample_idx])[0]
    if len(idx_all) == 0:
        print(f"\nNo matching t_seq for GOLDEN[{sample_idx}] ({t_g}) in ALL.")
        return

    idx_all = idx_all[0]

    xg_2d = Xg[sample_idx][:, 0, :, 0]
    xa_2d = Xa[idx_all][:, 0, :, 0]

    print(f"\nMatched GOLDEN[{sample_idx}] and ALL[{idx_all}] at time {t_g}")
    print("GOLDEN label:", yg[sample_idx])
    print("ALL    label:", ya[idx_all])

    print("\nGOLDEN window (head):")
    print(pd.DataFrame(xg_2d).head(show_rows))

    print("\nALL window (head):")
    print(pd.DataFrame(xa_2d).head(show_rows))







def print_feature_names(npz_path):
    """
    Prints feature names if stored inside the .npz file.
    Otherwise prints inferred placeholder names.
    """
    print(f"\n=== Checking feature names in: {npz_path} ===")
    data = np.load(npz_path, allow_pickle=True)

    print("NPZ keys:", data.files)

    # If feature names exist (only if you saved them earlier)
    if "feature_names" in data.files:
        feat = data["feature_names"]
        print("Feature names FOUND:")
        for i, name in enumerate(feat):
            print(f"{i}: {name}")
        return

    # Otherwise infer names based on X_low shape
    if "X_low" not in data.files:
        print("No X_low found. Cannot infer feature names.")
        return

    X = data["X_low"]
    # X shape: (N, time_steps=200, 1, F, 1)
    n_features = X.shape[3]

    print("No feature_names key found.")
    print(f"Inferring placeholder names for {n_features} features:")
    for i in range(n_features):
        print(f"{i}: Feature_{i}")




def inspect_npz_contents(npz_path, max_features=20, max_times=10):
    print("\n=== Inspecting NPZ: ", npz_path)
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print("❌ ERROR loading file:", e)
        return

    print("NPZ keys:", data.files)

    # ------------------------------
    # 1) FEATURE NAMES OR PLACEHOLDERS
    # ------------------------------
    if "X_low" in data.files:
        X = data["X_low"]
        n_features = X.shape[3]

        print(f"X_low shape: {X.shape}")
        print(f"Total features: {n_features}")

        if "feature_names" in data.files:
            print("Feature names (real):")
            feat_names = data["feature_names"]
            for i, n in enumerate(feat_names[:max_features]):
                print(f"{i}: {n}")
        else:
            print("No feature_names saved. Printing placeholder names:")
            for i in range(min(n_features, max_features)):
                print(f"{i}: Feature_{i}")

    else:
        print("⚠️ No X_low key in this NPZ.")

    # ------------------------------
    # 2) LABELS
    # ------------------------------
    if "y" in data.files:
        y = data["y"]
        print(f"\nLabels: {len(y)} samples")

        # show distribution
        import numpy as np
        unique, counts = np.unique(y, return_counts=True)
        print("Label distribution:")
        for u, c in zip(unique, counts):
            print(f"  {u}: {c}")
    else:
        print("⚠️ No y key in this NPZ.")

    # ------------------------------
    # 3) TIME SEQUENCES
    # ------------------------------
    if "t_seq" in data.files:
        t = data["t_seq"]
        print(f"\nTime sequence length: {len(t)}")

        print("First few time stamps:")
        import pandas as pd
        for i in range(min(max_times, len(t))):
            print(f"{i}: {pd.to_datetime(t[i])}")
    else:
        print("⚠️ No t_seq key in this NPZ.")




import numpy as np
import pandas as pd


def inspect_sequence(npz_path, sample_idx=100):
    # ---------------- LOAD NPZ ----------------
    data = np.load(npz_path, allow_pickle=True)
    print("NPZ keys:", data.files)

    # ---- Figure out X array name ----
    if "X_low" in data.files:
        X = data["X_low"]
    elif "X_low_Golden" in data.files:
        X = data["X_low_Golden"]
    else:
        raise KeyError("Could not find X array (X_low / X_low_Golden) in npz.")

    # ---- Figure out y array name ----
    if "y" in data.files:
        y = data["y"]
    elif "y_Golden" in data.files:
        y = data["y_Golden"]
    else:
        raise KeyError("Could not find y array (y / y_Golden) in npz.")

    # ---- Optional t_seq ----
    t_seq = None
    if "t_seq" in data.files:
        t_seq = data["t_seq"]

    print("\nShapes:")
    print("  X.shape =", X.shape)
    print("  y.shape =", y.shape)
    if t_seq is not None:
        print("  t_seq.shape =", t_seq.shape)
    else:
        print("  t_seq: <NOT PRESENT IN THIS FILE>")

    # ---------------- CHECK INDEX ----------------
    if sample_idx >= len(X):
        raise ValueError(f"sample_idx {sample_idx} out of range (N={len(X)})")

    # ---------------- EXTRACT SAMPLE ----------------
    window = X[sample_idx]  # expected (T, 1, F, 1)
    label = y[sample_idx]
    ts = t_seq[sample_idx] if t_seq is not None else None

    print("\n=== SAMPLE {} ===".format(sample_idx))
    if ts is not None:
        print("  Time index =", ts)
    else:
        print("  Time index = <t_seq not saved in this npz>")
    print("  Label      =", label)
    print("  Window raw shape:", window.shape)

    # Squeeze to (T, F) for readability
    window_2d = window.squeeze()   # (T, F)
    print("  Window 2D shape:", window_2d.shape)

    df_window = pd.DataFrame(
        window_2d,
        columns=[f"F{i}" for i in range(window_2d.shape[1])]
    )
    print("\nFirst 20 rows of this window:")
    print(df_window.head(20))






if __name__ == "__main__":
    path_golden = r"D:\ReViAI\Trained models\MWPS\epochs 500\EURUSD\M15\MWPS EURUSD M15 e.500 V.2025-07-03-04-25\classification_sequences_GOLDEN.npz"
    path_all    = r"D:\ReViAI\Trained models\MWPS\epochs 500\EURUSD\M15\MWPS EURUSD M15 e.500 V.2025-07-03-04-25\MWPSDATA_EURUSD_M15_O21uv1_classification_sequences.npz"



    inspect_npz_contents(path_golden)
    inspect_npz_contents(path_all)



    print_feature_names(path_golden)
    print_feature_names(path_all)


    # choose which sample in the GOLDEN dataset you want to inspect
    sample_idx = 100

    # 1) Inspect the window and label for this sample in the GOLDEN file
    inspect_sequence(path_golden, sample_idx)
    inspect_sequence(path_all, sample_idx)

    # 2) Compare the same t_seq between GOLDEN and ALL
    verify_sample(sample_idx, path_golden, path_all, show_rows=5)



