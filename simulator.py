"""Cellular Automata Simulator"""

import numpy as np
import pandas as pd

from grid import load_grid
from train import load_model_and_scaler

grid = load_grid()


def find_stock_position(grid: pd.DataFrame, stock_symbol: str) -> tuple[int, int]:
    position = np.argwhere(grid == stock_symbol)
    if len(position) == 0:
        raise ValueError(f"Stock symbol {stock_symbol} not found in grid")
    return position[0][0], position[0][1]


def get_plus_neighbors(grid: pd.DataFrame, row: int, col: int) -> list[tuple[int, int]]:
    neighbors = []
    if row > 0:
        neighbors.append((row - 1, col))
    if row < grid.shape[0] - 1:
        neighbors.append((row + 1, col))
    if col > 0:
        neighbors.append((row, col - 1))
    if col < grid.shape[1] - 1:
        neighbors.append((row, col + 1))
    return neighbors


def predict_neighbors(grid: pd.DataFrame, stock_symbol: str, input_seq: np.ndarray) -> dict[str, float]:
    row, col = find_stock_position(grid, stock_symbol)
    neighbors = get_plus_neighbors(grid, row, col)
    predictions = {}

    for neighbor_row, neighbor_col in neighbors:
        neighbor_stock_symbol = grid.iat[neighbor_row, neighbor_col]
        model, scaler = load_model_and_scaler(f"model_A_{stock_symbol}-{neighbor_stock_symbol}")
        prediction = model.predict(scaler, input_seq)
        predictions[neighbor_stock_symbol] = prediction

    return predictions


if __name__ == "__main__":
    # Example input sequence for a stock (5 days)
    input_seq = np.array([[0.0204], [-0.0102], [0.0306], [0.0102], [-0.0204]])

    # Predict the neighbors of "AAPL"
    predictions = predict_neighbors(grid, "AAPL", input_seq)

    print(predictions)
