"""Cellular Automata Simulator"""

import numpy as np
import pandas as pd

from data import load_daily_change_df
from grid import load_grid
from train import load_model_and_scaler

grid = load_grid()
dcdf = load_daily_change_df()


def get_trailing_stock_data(stock: str, data: float, n: int = 4) -> np.ndarray:
    """Prefix the data with the last n days of daily change data."""
    result = np.concatenate((dcdf[stock].iloc[-n:].values, [data]))
    return result.reshape(-1, 1)


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


def simulate(grid: pd.DataFrame, stock_symbol: str, input_seq: np.ndarray) -> dict[str, float]:
    # Initialize predictions dictionary
    predictions = {}
    # Initialize queue for BFS
    queue = [(stock_symbol, input_seq, None)]
    visited = set()

    while queue:
        current_symbol, current_input_seq, prev_neighbor_symbol = queue.pop(0)
        if current_symbol not in visited:
            visited.add(current_symbol)
            row, col = find_stock_position(grid, current_symbol)
            neighbors = get_plus_neighbors(grid, row, col)

            for neighbor_row, neighbor_col in neighbors:
                neighbor_stock_symbol = grid.iat[neighbor_row, neighbor_col]

                # Load model and scaler for the current symbol and neighbor
                model, scaler = load_model_and_scaler(f"model_A_{current_symbol}-{neighbor_stock_symbol}")

                # Predict the value for the current neighbor
                prediction = model.predict(scaler, current_input_seq)

                # If this neighbor has already been predicted by another neighbor, calculate the average
                if neighbor_stock_symbol in predictions and prev_neighbor_symbol != neighbor_stock_symbol:
                    predictions[neighbor_stock_symbol] = (predictions[neighbor_stock_symbol] + prediction) / 2
                else:
                    predictions[neighbor_stock_symbol] = prediction

                # Prepare input sequence for the next neighbor
                next_input_seq = get_trailing_stock_data(neighbor_stock_symbol, prediction)

                # Add the neighbor to the queue to process its neighbors
                queue.append((neighbor_stock_symbol, next_input_seq, current_symbol))

    return predictions


if __name__ == "__main__":
    # Input data
    ticker = "AAPL"
    change = 0.0102

    # Get the input sequence
    input_seq = get_trailing_stock_data(ticker, change)

    # Simulate all the cells in the grid
    predictions = simulate(grid, ticker, input_seq)

    print(predictions)
