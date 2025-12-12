"""
README: Classifier_Evaluations.py
Author: Madison Killen
Version: 2 (December 2025)

This script loads our 2,000 point golden dataset, trains a classifier,
reports Accuracy and F1, and then runs an ablation study by varying
the amount of training data. It also generates plots to show how
performance changes with dataset size.

Updates:
- Added TARGET_MODE to control how labels are defined:
- full = original 20-class NumericLabel (-10..-1, 1..10)
- sign = 2-class direction-only (-1 for negative, 1 for positive)
- 3class = 3-class: big down (<= -3), near flat (-2..2), big up (>= 3)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

#Adding stronger nonlinear RandomForestClassifier.
from sklearn.ensemble import RandomForestClassifier
#Configuring label types with the options: full, sign, 3class
TARGET_MODE = "sign"  # <-- change this if you want a different label scheme

#Load golden dataset
root = tk.Tk()
root.withdraw()
print("Please select the Golden Dataset as a CSV")
file_path = filedialog.askopenfilename(
    title="Select Golden Dataset CSV",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
)
if not file_path:
    raise ValueError("No file selected.")
print(f"\nLoaded file: {file_path}")
golden = pd.read_csv(file_path)

#Drop any junk index columns
junk_cols = [col for col in golden.columns if col.startswith("Unnamed")]
golden = golden.drop(columns=junk_cols, errors="ignore")
print("Dataset loaded successfully!")
print("Shape:", golden.shape)

#Use all Quant columns as numeric features
feature_cols = [c for c in golden.columns if c.startswith("Quant")]
X = golden[feature_cols]  # Feature matrix

#Raw 20 class labels from dataset
y_raw = golden["NumericLabel"].copy()

#Collapse labels according to TARGET_MODE
if TARGET_MODE == "full":
    y = y_raw
    print("\nUsing FULL 20-class NumericLabel.")
elif TARGET_MODE == "sign":
    # 2 class negative vs positive
    y = y_raw.apply(lambda v: -1 if v < 0 else 1)
    print("\nUsing SIGN labels: -1 (negative) vs 1 (positive).")
elif TARGET_MODE == "3class":
    def collapse_3(v: int) -> int:
        if v <= -3:
            return -1 #big down
        elif v >= 3:
            return 1 #big up
        else:
            return 0 #near flat/mild
    y = y_raw.apply(collapse_3)
    print("\nUsing 3-CLASS labels: -1 (big down), 0 (near flat), 1 (big up).")
else:
    raise ValueError(f"Unknown TARGET_MODE: {TARGET_MODE}")
print("\nNumber of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("\nLabel distribution (after TARGET_MODE collapse):")
print(y.value_counts().sort_index())

#Baseline classifier on full data - Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% test and 80% train
    random_state=42, #for reproducibility
    stratify=y 
)
print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

#Define the classifier pipeline:
#Imputer: handles missing values
#Scaler: standardizes features
clf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )),
])
clf.fit(X_train, y_train)

#Predict on test set
y_pred = clf.predict(X_test)
#Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
print(f"\n=== BASELINE RESULTS ON FULL DATASET (TARGET_MODE='{TARGET_MODE}') ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Macro F1:  {f1:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


#Ablation study
fractions = [0.1, 0.2, 0.5, 1.0]
ablation_results = []
for frac in fractions:
    print(f"\n=== Ablation: Training with {int(frac * 100)}% of data ===")

    n = int(len(X) * frac)
    X_sub = X.sample(n=n, random_state=42)
    y_sub = y.loc[X_sub.index]

    #Train andtest split on the subsample data
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_sub,
        y_sub,
        test_size=0.2,
        random_state=42,
        stratify=y_sub,
    )

    clf_sub = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    clf_sub.fit(X_train_sub, y_train_sub)
    y_pred_sub = clf_sub.predict(X_test_sub)
    acc_sub = accuracy_score(y_test_sub, y_pred_sub)
    f1_sub = f1_score(y_test_sub, y_pred_sub, average="macro")
    ablation_results.append({
        "fraction": frac,
        "accuracy": acc_sub,
        "f1_macro": f1_sub,
    })
    print(f"Accuracy:  {acc_sub:.4f}")
    print(f"Macro F1: {f1_sub:.4f}")

#Convert results to DF for plotting
ablation_df = pd.DataFrame(ablation_results)
print("\n=== Ablation Results Summary (TARGET_MODE='{TARGET_MODE}') ===")
print(ablation_df)
#Plot Accuracy vs dataset size
plt.figure()
plt.plot(ablation_df["fraction"] * 100, ablation_df["accuracy"], marker="o")
plt.title(f"Ablation Study: Accuracy vs Training Set Size (%)\nTARGET_MODE='{TARGET_MODE}'")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

#Plot Macro F1 vs dataset size
plt.figure()
plt.plot(ablation_df["fraction"] * 100, ablation_df["f1_macro"], marker="o")
plt.title(f"Ablation Study: Macro F1 vs Training Set Size (%)\nTARGET_MODE='{TARGET_MODE}'")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Macro F1 Score")
plt.grid(True)
plt.tight_layout()
plt.show()
