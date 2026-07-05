import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


def build_autoencoder(input_dim: int):
    """Build a dense autoencoder."""

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(4, activation="relu"),
        Dense(8, activation="relu"),
        Dense(16, activation="relu"),
        Dense(input_dim, activation="linear"),
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def train_autoencoder(
    model,
    X_normal,
    epochs: int = 100,
    batch_size: int = 32
):
    """Train Autoencoder on normal records."""

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        X_normal,
        X_normal,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=1
    )

    return history


def reconstruction_errors(model, X):
    """Calculate MSE reconstruction error."""

    reconstruction = model.predict(
        X,
        verbose=0
    )

    errors = np.mean(
        np.square(X - reconstruction),
        axis=1
    )

    return errors


def predict_anomalies(errors, threshold):
    """Classify records based on reconstruction error."""

    return (
        errors > threshold
    ).astype(int)