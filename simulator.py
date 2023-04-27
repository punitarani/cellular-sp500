"""Cellular Automata Simulator"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

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


def simulate(grid: pd.DataFrame, stock_symbol: str, input_seq: np.ndarray) -> list[dict[str, float]]:
    # Initialize the output list
    simulation_output = []

    # Initialize the predictions dictionary with the input stock_symbol
    predictions = {stock_symbol: input_seq[-1][0]}

    # Initialize queue for BFS
    queue = [(stock_symbol, input_seq, None)]

    # Store the visited cells and the cells that have been updated in the current iteration
    visited = set()
    updated = set()

    # Wrap the while loop with tqdm to show progress
    with tqdm(total=grid.size, desc="Simulating") as pbar:
        while queue:
            current_symbol, current_input_seq, prev_neighbor_symbol = queue.pop(0)
            if current_symbol not in visited:
                visited.add(current_symbol)
                row, col = find_stock_position(grid, current_symbol)
                neighbors = get_plus_neighbors(grid, row, col)

                # Iterate through the neighbors
                for neighbor_row, neighbor_col in neighbors:
                    neighbor_stock_symbol = grid.iat[neighbor_row, neighbor_col]

                    # Skip the input stock_symbol when processing neighbors
                    if neighbor_stock_symbol == stock_symbol:
                        continue

                    # Load model and scaler for the current symbol and neighbor
                    model, scaler = load_model_and_scaler(f"model_A_{current_symbol}-{neighbor_stock_symbol}")

                    # Predict the value for the current neighbor
                    prediction = model.predict(scaler, current_input_seq)

                    # If this neighbor has already been predicted by another neighbor in the same iteration,
                    # calculate the average
                    if neighbor_stock_symbol in updated:
                        predictions[neighbor_stock_symbol] = (predictions[neighbor_stock_symbol] + prediction) / 2
                    else:
                        predictions[neighbor_stock_symbol] = prediction
                        updated.add(neighbor_stock_symbol)

                    # Add the predictions to the simulation_output
                    simulation_output.append(predictions.copy())

                    # If the neighbor has not been visited yet, add it to the queue
                    if neighbor_stock_symbol not in visited:
                        next_input_seq = get_trailing_stock_data(neighbor_stock_symbol, prediction)
                        queue.append((neighbor_stock_symbol, next_input_seq, current_symbol))

                # Update the progress bar
                pbar.update(1)

    return simulation_output


def save_animation_with_progress_bar(animation, filename, fps, writer, total_frames):
    with tqdm(total=total_frames, desc=f"Saving animation {filename}") as pbar:
        def progress_callback(current_frame, total_frames):
            pbar.update(1)

        animation.save(filename, fps=fps, writer=writer, progress_callback=progress_callback)


def plot_simulation(grid: pd.DataFrame, simulations: list[dict[str, float]]):
    fig, ax = plt.subplots()

    # Create a colormap with a variant of green for positive values and a variant of red for negative values
    cmap = colors.LinearSegmentedColormap.from_list("stock_changes", ["red", "white", "green"])

    # Prepare a grid to store the changed percentage values
    percentage_grid = np.zeros_like(grid, dtype=float)

    def update(frame):
        ax.clear()
        simulation = simulations[frame]

        # Fill the grid with the corresponding percentage changes
        for symbol, change in simulation.items():
            row, col = find_stock_position(grid, symbol)
            percentage_grid[row, col] = change

        # Plot the grid with the percentage changes
        ax.imshow(percentage_grid, cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(f"Iteration {frame + 1}")

    # Calculate the interval and fps so animation is ~10s long
    num_frames = len(simulations)
    interval = 10000 // num_frames
    fps = (1000 // interval) * 10

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, repeat=False)

    # Save the animation as a video file
    save_animation_with_progress_bar(ani, 'simulation.gif', fps=fps, writer='pillow', total_frames=num_frames)
    print("Animation saved to simulation.gif")

    plt.show()


if __name__ == "__main__":
    # Input data
    ticker = "AAPL"
    change = 0.5825

    # Get the input sequence
    input_seq = get_trailing_stock_data(ticker, change)

    # Simulate all the cells in the grid
    simulations = simulate(grid, ticker, input_seq)
    # Save the simulations to a file
    with open("simulations.json", "w") as f:
        json.dump(simulations, f)

    # Plot the animation of the simulations
    plot_simulation(grid, simulations)
