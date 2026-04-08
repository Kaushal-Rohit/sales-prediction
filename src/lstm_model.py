"""
lstm_model.py
-------------
Build, train, and forecast with an LSTM neural network using Keras/TensorFlow.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def build_lstm_model(lookback: int = 12,
                     lstm_units_1: int = 64,
                     lstm_units_2: int = 32,
                     dropout_rate: float = 0.2,
                     learning_rate: float = 0.001) -> Sequential:
    """
    Build a two-layer stacked LSTM model.

    Architecture
    ------------
    Input → LSTM(64, return_sequences=True) → Dropout
          → LSTM(32) → Dropout
          → Dense(16, relu) → Dense(1, linear)

    Parameters
    ----------
    lookback : int
        Number of past time steps used as input (must match sequence creation)
    lstm_units_1 : int
        Neurons in the first LSTM layer
    lstm_units_2 : int
        Neurons in the second LSTM layer
    dropout_rate : float
        Dropout applied after each LSTM layer to prevent overfitting
    learning_rate : float
        Adam optimizer learning rate

    Returns
    -------
    Compiled Keras Sequential model
    """
    model = Sequential([
        LSTM(lstm_units_1, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout_rate),
        LSTM(lstm_units_2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(1)   # Linear output — we're predicting a continuous value
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    print(model.summary())
    return model


def train_lstm(model: Sequential,
               X_train: np.ndarray,
               y_train: np.ndarray,
               epochs: int = 100,
               batch_size: int = 8,
               validation_split: float = 0.1) -> dict:
    """
    Train the LSTM with early stopping and learning rate reduction.

    Returns
    -------
    Keras History object (access via .history['loss'] etc.)
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8,
                          min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

    print(f"[LSTM] Training complete. Best val_loss: {min(history.history['val_loss']):.6f}")
    return history


def predict_lstm(model: Sequential,
                 X_test: np.ndarray,
                 scaler) -> np.ndarray:
    """
    Generate predictions and inverse-scale back to original sales units.

    Parameters
    ----------
    model : trained Keras model
    X_test : np.ndarray, shape (n_samples, lookback, 1)
    scaler : fitted MinMaxScaler from data_preprocessing.scale_series()

    Returns
    -------
    np.ndarray of predicted sales values in original scale
    """
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds


def recursive_forecast(model: Sequential,
                       last_sequence: np.ndarray,
                       n_steps: int,
                       scaler) -> np.ndarray:
    """
    Recursively forecast n_steps into the future using the trained LSTM.
    Each prediction is fed back as input to the next step (autoregressive).

    Parameters
    ----------
    last_sequence : np.ndarray, shape (lookback,) — last known scaled values
    n_steps : int — number of future steps to forecast
    scaler : fitted MinMaxScaler

    Returns
    -------
    np.ndarray of forecasted values in original scale
    """
    sequence = list(last_sequence.copy())
    predictions = []

    for _ in range(n_steps):
        x_input = np.array(sequence[-len(last_sequence):]).reshape(1, -1, 1)
        pred_scaled = model.predict(x_input, verbose=0)[0][0]
        predictions.append(pred_scaled)
        sequence.append(pred_scaled)

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()


def plot_lstm_training(history, save_path: str = None):
    """Plot training & validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'], label='Training Loss', color='#4C72B0', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation Loss', color='#DD8452',
            linewidth=2, linestyle='--')
    ax.set_title("LSTM — Training History", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.legend(fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.show()


def plot_lstm_forecast(dates_train, train_actual,
                       dates_test, test_actual, test_pred,
                       save_path: str = None):
    """Plot LSTM predictions vs actuals on the test set."""
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(dates_train, train_actual, color='gray', label='Training Actuals',
            linewidth=1.5, alpha=0.7)
    ax.plot(dates_test, test_actual, color='black', label='Test Actuals',
            linewidth=2, marker='o', markersize=5)
    ax.plot(dates_test, test_pred, color='#DD8452', label='LSTM Predictions',
            linewidth=2, linestyle='--', marker='s', markersize=4)

    ax.axvline(dates_test.iloc[0], color='red', linestyle=':', linewidth=1.5,
               label='Train/Test Split')

    ax.set_title("LSTM — Sales Forecast vs Actuals", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Monthly Sales ($)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.show()
