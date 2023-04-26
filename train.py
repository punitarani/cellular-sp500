"""Generate the weights for the grid."""

import multiprocessing
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange

from grid import get_neighbors, load_grid
from lstm import LSTMModel, train_lstm_model

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_sequences(data: np.ndarray, seq_length: int) -> np.ndarray:
    """
    Create sequences of length seq_length from the data.

    Args:
        data (np.ndarray): Data to create sequences from.
        seq_length (int): Length of the sequences.

    Returns:
        np.ndarray: Array of sequences of shape (len(data) - seq_length, seq_length).
    """
    X = []
    y = []

    for i in range(len(data) - seq_length):
        X.append(data[i : (i + seq_length)])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


def create_models_for_pair(
    stock_A: pd.Series, stock_B: pd.Series, seq_length: int = 5, epochs: int = 100
) -> tuple[tuple[LSTMModel, MinMaxScaler], tuple[LSTMModel, MinMaxScaler]]:
    """
    Create LSTM models for a pair of stocks.

    Args:
        stock_A (pd.Series): Time series of stock A.
        stock_B (pd.Series): Time series of stock B.
        seq_length (int): Length of the input sequence.
        epochs (int): Number of epochs to train for.

    Returns:
        Tuple: Tuple containing the trained models and the scaler used to scale the data.
    """

    # Prepare the input and target data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(
        np.concatenate([stock_A.values.reshape(-1, 1), stock_B.values.reshape(-1, 1)])
    )
    scaled_stock_A = scaled_data[: len(stock_A)]
    scaled_stock_B = scaled_data[len(stock_A) :]

    X_A, y_A = create_sequences(scaled_stock_A, seq_length)
    X_B, y_B = create_sequences(scaled_stock_B, seq_length)

    # Reshape input data for LSTM (samples, timesteps, features)
    X_A = X_A.reshape(X_A.shape[0], seq_length, 1)
    X_B = X_B.reshape(X_B.shape[0], seq_length, 1)

    # Create LSTM models
    model_A = LSTMModel()
    model_B = LSTMModel()

    # Move models to GPU if available
    model_A = model_A.to(device)
    model_B = model_B.to(device)

    # Train LSTM models
    train_lstm_model(model_A, X_A, y_A, epochs=epochs)
    train_lstm_model(model_B, X_B, y_B, epochs=epochs)

    return (model_A, scaler), (model_B, scaler)


def save_model_and_scaler(
    model: LSTMModel, scaler: MinMaxScaler, filename: str
) -> None:
    """
    Save the model and scaler to disk.

    Args:
        model (LSTMModel): LSTM model to save.
        scaler (MinMaxScaler): Scaler used to scale the data.
        filename (str): Name of the file to save the model to.
    """

    torch.save(model.state_dict(), f"weights/{filename}.pth")
    torch.save(model, f"models/{filename}.pt")
    with open(f"scalers/{filename}.pkl", "wb") as f:
        pickle.dump(scaler, f)


def load_model_and_scaler(filename: str) -> tuple[LSTMModel, MinMaxScaler]:
    """
    Load the model and scaler from disk.

    Args:
        filename (str): Name of the file to load the model from.

    Returns:
        Tuple: Tuple containing the loaded model and scaler.
    """

    model = LSTMModel()
    model.load_state_dict(torch.load(f"weights/{filename}.pth"))
    model.eval()
    with open(f"scalers/{filename}.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def create_and_train_models(stock_A, stock_B, stock_A_symbol, stock_B_symbol):
    """Create and train models for a pair of stocks.

    Args:
        stock_A (pd.Series): Time series of stock A.
        stock_B (pd.Series): Time series of stock B.
        stock_A_symbol (str): Symbol of stock A.
        stock_B_symbol (str): Symbol of stock B.

    Returns:
        Tuple: Tuple containing the key for the pair and the trained models.
    """
    model_key = f"{stock_A_symbol}-{stock_B_symbol}"
    models_A, models_B = create_models_for_pair(stock_A, stock_B)
    save_model_and_scaler(
        models_A[0], models_A[1], f"model_A_{stock_A_symbol}-{stock_B_symbol}"
    )
    save_model_and_scaler(
        models_B[0], models_B[1], f"model_B_{stock_A_symbol}-{stock_B_symbol}"
    )
    return model_key, (models_A, models_B)


if __name__ == "__main__":
    # Load the grid of stocks
    grid = load_grid()
    rows, cols = grid.shape
    print(f"Loaded grid of shape {grid.shape}")

    # 'df' is the DataFrame with daily change percentages for 500 stocks
    df = pd.read_csv("sp500_daily_change.csv")
    print(f"Loaded DataFrame of shape {df.shape}")

    models_dict = {}

    stock_pairs = []

    for row in range(rows):
        for col in range(cols):
            stock_symbol = grid.iat[row, col]
            stock = df[stock_symbol]
            neighbors = get_neighbors(grid, row, col)

            for neighbor_row, neighbor_col in neighbors:
                neighbor_stock_symbol = grid.iat[neighbor_row, neighbor_col]
                neighbor_stock = df[neighbor_stock_symbol]
                stock_pairs.append(
                    (stock, neighbor_stock, stock_symbol, neighbor_stock_symbol)
                )

    # Use multiprocessing to create and train models
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        trained_models = pool.starmap(
            create_and_train_models,
            tqdm(stock_pairs, desc="Training Models", ncols=100),
            chunksize=1,
        )

    # Update models_dict with trained models
    for model_key, models in trained_models:
        models_dict[model_key] = models

    print("Done training models")
