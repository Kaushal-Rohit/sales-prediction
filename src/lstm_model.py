"""
LSTM-based time-series forecasting model.

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network
capable of learning long-range temporal dependencies.  They can model complex
non-linear patterns but are less interpretable than statistical models and
require more data and careful hyper-parameter tuning.
"""

from __future__ import annotations

import numpy as np

# keras / tensorflow – imported lazily to allow unit-testing without a GPU
from tensorflow import keras  # type: ignore[import]
from tensorflow.keras import layers  # type: ignore[import]


def build_lstm_model(
    look_back: int = 30,
    units: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    """Build and compile a stacked LSTM regression model.

    Architecture
    ------------
    Input  → LSTM(units, return_sequences=True) → Dropout
           → LSTM(units // 2)                   → Dropout
           → Dense(1)

    Parameters
    ----------
    look_back:
        Number of past time steps fed as input.
    units:
        Number of units in the first LSTM layer.
    dropout:
        Dropout rate applied after each LSTM layer.

    Returns
    -------
    keras.Model
        Compiled Keras model ready for training.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(look_back, 1)),
            layers.LSTM(units, return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(units // 2),
            layers.Dropout(dropout),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def fit_lstm(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    validation_split: float = 0.1,
    verbose: int = 0,
) -> keras.callbacks.History:
    """Train the LSTM model.

    Parameters
    ----------
    model:
        A compiled Keras model (from :func:`build_lstm_model`).
    X_train:
        Training input of shape ``(samples, look_back, 1)``.
    y_train:
        Training targets of shape ``(samples,)``.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    validation_split:
        Fraction of training data to use for in-training validation.
    verbose:
        Keras verbosity level.

    Returns
    -------
    keras.callbacks.History
        Training history object.
    """
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=verbose,
    )
    return history


def predict_lstm(
    model: keras.Model,
    X: np.ndarray,
    scaler,
) -> np.ndarray:
    """Run inference and inverse-transform predictions to original scale.

    Parameters
    ----------
    model:
        A trained Keras model.
    X:
        Input array of shape ``(samples, look_back, 1)`` in *scaled* space.
    scaler:
        Fitted ``MinMaxScaler`` used during preprocessing.

    Returns
    -------
    np.ndarray
        Predictions in the *original* (unscaled) units.
    """
    scaled_preds = model.predict(X, verbose=0).flatten()
    preds = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
    return preds
