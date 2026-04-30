"""
ANN Model Module
=================
Builds and trains an Artificial Neural Network (feedforward)
for binary classification of fraudulent transactions.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def build_ann(input_dim: int) -> Sequential:
    """
    Build an ANN model for fraud detection.

    Architecture:
        Input -> Dense(128) -> BN -> Dropout
              -> Dense(64)  -> BN -> Dropout
              -> Dense(32)  -> BN -> Dropout
              -> Dense(1, sigmoid)
    """
    model = Sequential([
        Dense(128, activation="relu", input_dim=input_dim, name="dense_1"),
        BatchNormalization(name="bn_1"),
        Dropout(0.3, name="dropout_1"),

        Dense(64, activation="relu", name="dense_2"),
        BatchNormalization(name="bn_2"),
        Dropout(0.3, name="dropout_2"),

        Dense(32, activation="relu", name="dense_3"),
        BatchNormalization(name="bn_3"),
        Dropout(0.2, name="dropout_3"),

        Dense(1, activation="sigmoid", name="output"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )

    model.summary()
    return model


def train_ann(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=256):
    """Train the ANN model with early stopping and learning rate reduction."""
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return history
