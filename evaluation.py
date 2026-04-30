"""
Evaluation Module
==================
Functions for evaluating and comparing all models.
Generates confusion matrices, ROC curves, and training history plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, roc_auc_score, f1_score,
                             precision_score, recall_score, accuracy_score)


def evaluate_model(model, X_test, y_test, model_name="Model", save_dir="results"):
    """Evaluate a supervised model and save plots."""
    os.makedirs(save_dir, exist_ok=True)

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print(f"\n===== {model_name} Evaluation =====")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    metrics = {"accuracy": acc, "precision": prec, "recall": rec,
               "f1_score": f1, "auc_roc": auc, "y_pred": y_pred, "y_pred_prob": y_pred_prob}

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax, cbar=True)
    ax.set_title(f"{model_name} - Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name.lower()}_confusion_matrix.png"), dpi=150)
    plt.close()

    return metrics


def plot_training_history(history, model_name="Model", save_dir="results"):
    """Plot training and validation loss/accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title(f"{model_name} - Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if "accuracy" in history.history:
        axes[1].plot(history.history["accuracy"], label="Train Acc", linewidth=2)
        axes[1].plot(history.history["val_accuracy"], label="Val Acc", linewidth=2)
        axes[1].set_title(f"{model_name} - Accuracy", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No accuracy metric", ha="center", va="center", fontsize=14)
        axes[1].set_title(f"{model_name} - Accuracy (N/A)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name.lower()}_training_history.png"), dpi=150)
    plt.close()


def plot_roc_comparison(y_test, all_metrics, save_dir="results"):
    """Plot ROC curves for all models on a single chart."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    for i, (name, metrics) in enumerate(all_metrics.items()):
        if "y_pred_prob" in metrics:
            fpr, tpr, _ = roc_curve(y_test, metrics["y_pred_prob"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={metrics['auc_roc']:.4f})",
                    linewidth=2, color=colors[i % len(colors)])

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_comparison.png"), dpi=150)
    plt.close()


def print_comparison_table(all_metrics):
    """Print a formatted comparison table of all models."""
    print("\n" + "=" * 70)
    print(f"{'MODEL COMPARISON':^70}")
    print("=" * 70)
    print(f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC-ROC':>10}")
    print("-" * 70)
    for name, m in all_metrics.items():
        print(f"{name:<15} {m.get('accuracy', 0):>10.4f} {m.get('precision', 0):>10.4f} "
              f"{m.get('recall', 0):>10.4f} {m.get('f1_score', 0):>10.4f} {m.get('auc_roc', 0):>10.4f}")
    print("=" * 70)
