"""
README: Classifier_Evaluations.py
Author: Madison Killen
Version: 1 (November 2025)

This script loads our 2,000 point golden dataset, trains a baseline classifier,
reports Accuracy and F1, and then runs an ablation study by varying the amount
of training data. It also generates plots to show how performance changes with dataset size.
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

#STEP 1: Load golden dataset
root = tk.Tk()
root.withdraw()
print("Please select the Golden Dataset as a CSV")
file_path = filedialog.askopenfilename(
    title="Select Golden Dataset CSV",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
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


#STEP 2: Define features X and labels y
#Use all Quant columns as numeric features
feature_cols = [c for c in golden.columns if c.startswith("Quant")]

X = golden[feature_cols] #Feature matrix
y = golden["NumericLabel"] #Multi class target labels

print("\nNumber of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("\nLabel distribution (NumericLabel):")
print(y.value_counts().sort_index())


#STEP 3: Baseline classifier on full golden data

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% test and 80% train
    random_state=42, #for reproducibility
    stratify=y #Keep class proportions similar in train and test
)
print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

#Define the classifier pipeline
#Imputer: handles missing values
#Scaler: standardizes features
#LogisticRegression: simple baseline classifier
clf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

#Fit the classifier
clf.fit(X_train, y_train)

#Predict on test set
y_pred = clf.predict(X_test)

#Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)
print(f"\n=== BASELINE RESULTS ON FULL GOLDEN DATASET ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Macro F1:  {f1:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(cm)


#STEP 4: Ablation study
fractions = [0.1, 0.2, 0.5, 1.0]
ablation_results = []
for frac in fractions:
    print(f"\n=== Ablation: Training with {int(frac * 100)}% of data ===")
    
    #Subsample a fraction of the data
    n = int(len(X) * frac)
    X_sub = X.sample(n=n, random_state=42)
    y_sub = y.loc[X_sub.index]
    
    #Train/test split on the subsampled data
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_sub,
        y_sub,
        test_size=0.2,
        random_state=42,
        stratify=y_sub
    )
    clf_sub = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
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

#Plot
ablation_df = pd.DataFrame(ablation_results)
print("\n=== Ablation Results Summary ===")
print(ablation_df)
plt.figure()
plt.plot(ablation_df["fraction"] * 100, ablation_df["accuracy"], marker="o")
plt.title("Ablation Study: Accuracy vs Training Set Size (%)")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

#F1 vs dataset size
plt.figure()
plt.plot(ablation_df["fraction"] * 100, ablation_df["f1_macro"], marker="o")
plt.title("Ablation Study: Macro F1 vs Training Set Size (%)")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Macro F1 Score")
plt.grid(True)
plt.tight_layout()
plt.show()
