"""
Data Preprocessing Module
=========================
Handles data loading, exploration, cleaning, normalization,
class balancing (SMOTE), and train/test splitting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(filepath: str) -> pd.DataFrame:
    """Load the credit card fraud dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Place 'creditcard.csv' inside the 'data/' folder."
        )
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df: pd.DataFrame, save_dir: str = "results") -> None:
    """Perform exploratory data analysis and save visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    print("\n===== Dataset Info =====")
    print(df.info())
    print("\n===== Statistical Summary =====")
    print(df.describe())
    print("\n===== Class Distribution =====")
    print(df["Class"].value_counts())
    print(f"Fraud ratio: {df['Class'].mean() * 100:.4f}%")

    # --- Class distribution plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    class_counts = df["Class"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].bar(["Legitimate", "Fraudulent"], class_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Transaction Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 500, str(v), ha="center", fontweight="bold")

    # Amount distribution
    axes[1].hist(df[df["Class"] == 0]["Amount"], bins=50, alpha=0.7, label="Legitimate", color="#2ecc71")
    axes[1].hist(df[df["Class"] == 1]["Amount"], bins=50, alpha=0.7, label="Fraudulent", color="#e74c3c")
    axes[1].set_title("Transaction Amount Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Amount")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].set_xlim(0, 2500)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_class_distribution.png"), dpi=150)
    plt.close()

    # --- Correlation heatmap (top features) ---
    fig, ax = plt.subplots(figsize=(16, 12))
    corr = df.corr()
    # Show only features with notable correlation to Class
    top_features = corr["Class"].abs().sort_values(ascending=False).head(15).index
    sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap (Top Features vs Class)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_correlation_heatmap.png"), dpi=150)
    plt.close()

    print(f"\n[INFO] EDA plots saved to '{save_dir}/'")


def preprocess_data(df: pd.DataFrame, apply_smote: bool = True, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocess the dataset:
    1. Normalize 'Time' and 'Amount' features
    2. Split into train/test (stratified)
    3. Optionally apply SMOTE to training data

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    df = df.copy()

    # --- Normalize Time and Amount ---
    scaler = StandardScaler()
    df["Time"] = scaler.fit_transform(df[["Time"]])
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # --- Separate features and labels ---
    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    # --- Stratified train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n[INFO] Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    print(f"[INFO] Train fraud ratio: {y_train.mean() * 100:.4f}%")

    # --- Apply SMOTE to balance training data ---
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"[INFO] After SMOTE — Train set: {X_train.shape[0]} samples")
        print(f"[INFO] After SMOTE — Fraud ratio: {y_train.mean() * 100:.2f}%")

    # --- Normalize all features (fit on train, transform both) ---
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_scaler


def prepare_sequential_data(X_train, X_test):
    """
    Reshape data for CNN and LSTM models.
    CNN/LSTM expect 3D input: (samples, timesteps, features)
    We treat each feature as a single timestep.
    """
    X_train_seq = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_seq = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    print(f"[INFO] Sequential data shape: {X_train_seq.shape}")
    return X_train_seq, X_test_seq
