"""
LSTM Model Module
==================
Builds and trains an LSTM network to capture
temporal/sequential patterns in transaction data.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def build_lstm(input_shape: tuple) -> Sequential:
    """
    Build an LSTM model for fraud detection.

    Architecture:
        LSTM(64, return_sequences=True) -> Dropout
        LSTM(32) -> Dropout
        Dense(32) -> Dense(1, sigmoid)
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, name="lstm_1"),
        Dropout(0.3, name="dropout_1"),

        LSTM(32, return_sequences=False, name="lstm_2"),
        Dropout(0.3, name="dropout_2"),

        Dense(32, activation="relu", name="dense_1"),
        BatchNormalization(name="bn_1"),
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


def train_lstm(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=256):
    """Train the LSTM model with early stopping and LR reduction."""
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
