"""
MWPS Forgettability Analysis

Reads the per-epoch correctness Excel produced by GoldenForgettabilityCallback
and computes per-example difficulty / forgettability scores, then plots them.

Input Excel (example):
  classification_sequences_GOLDEN_forgettability.xlsx

Index:
  - DateTimeIndex (t_seq) from GOLDEN NPZ

Columns:
  - epoch_001, epoch_002, ..., epoch_N  (0 / 1 correctness flags)

Outputs:
  - New Excel with extra columns:
      num_epochs
      sum_correct
      mean_correct
      num_wrong
      forgettability_score         (0 = always correct, 1 = always wrong)
      num_forgetting_events        (# of 1→0 transitions)
      stability_score              (1 - normalized forgetting events)
      learnability_score           (= mean_correct)
      difficulty_score             (combined)

  - Scatter plot:
      x-axis: learnability_score
      y-axis: stability_score
      Easy & stable examples → top-right
      Hard & unstable examples → bottom-left
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_forgettability_excel(xlsx_path: str) -> pd.DataFrame:
    """
    Load the forgettability Excel.

    Assumes:
      - Index: DateTimeIndex or any unique index
      - Columns: epoch_001, epoch_002, ..., epoch_N with 0/1 values
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Excel not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path, index_col=0)
    # Keep only epoch_* columns (just in case there are others)
    epoch_cols = [c for c in df.columns if c.startswith("epoch_")]
    if not epoch_cols:
        raise ValueError("No columns starting with 'epoch_' found in Excel.")

    df = df[epoch_cols].copy()
    return df


def compute_forgettability_metrics(df_epochs: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns epoch_XXX (0/1),
    compute per-example metrics:

      - num_epochs
      - sum_correct
      - mean_correct
      - num_wrong
      - forgettability_score  (fraction of epochs where example was wrong)
      - num_forgetting_events (# times it went 1 → 0 across epochs)
      - stability_score       (1 - normalized forgetting events)
      - learnability_score    (= mean_correct)
      - difficulty_score      (combined: higher = more difficult)
    """
    epoch_cols = [c for c in df_epochs.columns if c.startswith("epoch_")]
    data = df_epochs[epoch_cols].values.astype(np.float32)  # shape: (N, E)

    N, E = data.shape
    num_epochs = E

    # Basic correctness stats
    sum_correct = data.sum(axis=1)              # how many epochs correct
    mean_correct = sum_correct / num_epochs     # in [0, 1]
    num_wrong = num_epochs - sum_correct

    # Forgettability: fraction of epochs wrong
    #  0.0 → always correct (easiest)
    #  1.0 → always wrong (hardest)
    forgettability_score = num_wrong / num_epochs

    # Forgetting events: count 1→0 transitions across epochs
    # For each row, look at differences between consecutive epochs.
    if num_epochs > 1:
        diffs = data[:, 1:] - data[:, :-1]   # shape: (N, E-1)
        # A forgetting event is 1 → 0 → diff = -1
        num_forgetting_events = (diffs == -1.0).sum(axis=1)
    else:
        num_forgetting_events = np.zeros(N, dtype=np.int32)

    max_forgetting = max(num_forgetting_events.max(), 1)
    # Stability: 1 - normalized forgetting events
    #  1.0 → no forgetting
    #  0.0 → as many forgetting events as the max example
    stability_score = 1.0 - (num_forgetting_events / max_forgetting)

    # Learnability: simply how often it was correct
    learnability_score = mean_correct

    # Combined difficulty (you can tweak this formula):
    #   higher difficulty_score → more difficult example
    difficulty_score = 1.0 - 0.5 * (learnability_score + stability_score)

    # Build output DataFrame with all metrics
    df_metrics = df_epochs.copy()
    df_metrics["num_epochs"] = num_epochs
    df_metrics["sum_correct"] = sum_correct
    df_metrics["mean_correct"] = mean_correct
    df_metrics["num_wrong"] = num_wrong
    df_metrics["forgettability_score"] = forgettability_score
    df_metrics["num_forgetting_events"] = num_forgetting_events
    df_metrics["stability_score"] = stability_score
    df_metrics["learnability_score"] = learnability_score
    df_metrics["difficulty_score"] = difficulty_score

    return df_metrics


def plot_learnability_vs_stability(df_metrics: pd.DataFrame,
                                   title: str = "Example Learnability vs Stability",
                                   sample_frac: float = 1.0,
                                   random_state: int = 42):
    """
    Scatter plot:

      x-axis: learnability_score (0..1)
      y-axis: stability_score (0..1)

    Easy, well-learned examples → top-right
    Hard, unstable examples     → bottom-left
    """
    if "learnability_score" not in df_metrics.columns or "stability_score" not in df_metrics.columns:
        raise ValueError("df_metrics must contain 'learnability_score' and 'stability_score' columns.")

    df_plot = df_metrics[["learnability_score", "stability_score"]].copy()

    # Optional subsampling for speed / readability on big datasets
    if 0.0 < sample_frac < 1.0 and len(df_plot) > 0:
        df_plot = df_plot.sample(frac=sample_frac, random_state=random_state)

    x = df_plot["learnability_score"].values
    y = df_plot["stability_score"].values

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=5, alpha=0.35)
    plt.xlabel("Learnability score (mean correctness over epochs)")
    plt.ylabel("Stability score (1 - normalized forgetting events)")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    # Annotate quadrants (just to visually interpret)
    # Top-right: easy, well-remembered
    # Bottom-left: hard, unstable
    plt.tight_layout()
    plt.show()


def analyze_forgettability(
    xlsx_path: str,
    out_excel_path: str | None = None,
    make_plot: bool = True,
    sample_frac_for_plot: float = 1.0,
):
    """
    High-level driver:

      1) Load forgettability Excel
      2) Compute per-example metrics
      3) Save extended Excel (if out_excel_path given)
      4) Plot learnability vs stability

    Parameters
    ----------
    xlsx_path : str
        Path to classification_sequences_GOLDEN_forgettability.xlsx
    out_excel_path : str or None
        Where to save extended metrics Excel. If None, auto-name next to input.
    make_plot : bool
        Whether to show scatter plot.
    sample_frac_for_plot : float
        Fraction of points to plot (1.0 → all).
    """
    print(f"[analyze_forgettability] Loading: {xlsx_path}")
    df_epochs = load_forgettability_excel(xlsx_path)

    print("[analyze_forgettability] Computing metrics...")
    df_metrics = compute_forgettability_metrics(df_epochs)

    # Save extended Excel
    if out_excel_path is None:
        base, ext = os.path.splitext(xlsx_path)
        out_excel_path = base + "_metrics.xlsx"

    df_metrics.to_excel(out_excel_path)
    print(f"[analyze_forgettability] Saved metrics → {out_excel_path}")

    # Plot
    if make_plot:
        title = os.path.basename(xlsx_path).replace("_forgettability", "")
        plot_learnability_vs_stability(
            df_metrics,
            title=f"Learnability vs Stability: {title}",
            sample_frac=sample_frac_for_plot,
        )

    return df_metrics


# =============================================================================
# MAIN (example usage)
# =============================================================================

if __name__ == "__main__":
    # 🔧 Adjust this BASE_DIR and file name to your run
    BASE_DIR = "/mnt/3090D/ReViAI/Trained models/MWPS/epochs 500/EURUSD/M15/MWPS EURUSD M15 e.500 V.2025-07-03-04-25"

    xlsx_path = os.path.join(
        BASE_DIR,
        "classification_sequences_GOLDEN_forgettability.xlsx"
    )

    analyze_forgettability(
        xlsx_path=xlsx_path,
        out_excel_path=None,      # auto name: *_metrics.xlsx
        make_plot=True,
        sample_frac_for_plot=1.0  # or e.g. 0.3 if too many points
    )
