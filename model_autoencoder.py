"""
Autoencoder Model Module (Anomaly Detection)
=============================================
Trains on legitimate transactions only. High reconstruction error = fraud.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve)


def build_autoencoder(input_dim, encoding_dim=14):
    inp = Input(shape=(input_dim,), name="input")
    x = Dense(64, activation="relu", name="enc1")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu", name="enc2")(x)
    x = BatchNormalization()(x)
    bottleneck = Dense(encoding_dim, activation="relu", name="bottleneck")(x)
    x = Dense(32, activation="relu", name="dec1")(bottleneck)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu", name="dec2")(x)
    x = BatchNormalization()(x)
    out = Dense(input_dim, activation="linear", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="Autoencoder")
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.summary()
    return model


def train_autoencoder(model, X_train_normal, X_val_normal, epochs=100, batch_size=256):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    ]
    history = model.fit(
        X_train_normal, X_train_normal,
        validation_data=(X_val_normal, X_val_normal),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1,
    )
    return history


def compute_reconstruction_error(model, X):
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.power(X - X_pred, 2), axis=1)


def find_optimal_threshold(model, X_val, y_val):
    errors = compute_reconstruction_error(model, X_val)
    precisions, recalls, thresholds = precision_recall_curve(y_val, errors)
    f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    idx = np.argmax(f1)
    threshold = thresholds[idx]
    print(f"\n[Autoencoder] Optimal threshold: {threshold:.6f}, F1: {f1[idx]:.4f}")
    return threshold


def evaluate_autoencoder(model, X_test, y_test, threshold, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    errors = compute_reconstruction_error(model, X_test)
    y_pred = (errors > threshold).astype(int)

    print("\n===== Autoencoder Anomaly Detection Results =====")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    auc = roc_auc_score(y_test, errors)
    print(f"AUC-ROC Score: {auc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(errors[y_test == 0], bins=100, alpha=0.7, label="Legit", color="#2ecc71", density=True)
    axes[0].hist(errors[y_test == 1], bins=100, alpha=0.7, label="Fraud", color="#e74c3c", density=True)
    axes[0].axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Thresh={threshold:.4f}")
    axes[0].set_title("Reconstruction Error Distribution", fontweight="bold")
    axes[0].legend()

    cm = confusion_matrix(y_test, y_pred)
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_title("Autoencoder Confusion Matrix", fontweight="bold")
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Legit", "Fraud"]); axes[1].set_yticklabels(["Legit", "Fraud"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold")
    fig.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "autoencoder_anomaly_results.png"), dpi=150)
    plt.close()
    return {"auc_roc": auc, "threshold": threshold, "y_pred": y_pred, "errors": errors}
