"""
CNN Model Module
=================
Builds and trains a 1D Convolutional Neural Network
for fraud detection by extracting local feature patterns.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def build_cnn(input_shape: tuple) -> Sequential:
    """
    Build a 1D-CNN model for fraud detection.

    Architecture:
        Conv1D(64) -> BN -> MaxPool
        Conv1D(128) -> BN -> MaxPool
        Flatten -> Dense(64) -> Dropout -> Dense(1, sigmoid)
    """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", padding="same",
               input_shape=input_shape, name="conv1d_1"),
        BatchNormalization(name="bn_1"),
        MaxPooling1D(pool_size=2, name="maxpool_1"),

        Conv1D(128, kernel_size=3, activation="relu", padding="same", name="conv1d_2"),
        BatchNormalization(name="bn_2"),
        MaxPooling1D(pool_size=2, name="maxpool_2"),

        Flatten(name="flatten"),

        Dense(64, activation="relu", name="dense_1"),
        Dropout(0.3, name="dropout_1"),

        Dense(32, activation="relu", name="dense_2"),
        Dropout(0.2, name="dropout_2"),

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


def train_cnn(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=256):
    """Train the CNN model with early stopping and LR reduction."""
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
