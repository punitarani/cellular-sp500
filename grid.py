"""Generate the grid"""

import json

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def load_grid():
    return pd.read_csv("sp500_grid.csv")


def get_neighbors(grid, row, col):
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            new_row, new_col = row + i, col + j
            if 0 <= new_row < grid.shape[0] and 0 <= new_col < grid.shape[1]:
                neighbors.append((new_row, new_col))
    return neighbors


def find_nearest_empty_cell(x, y, grid):
    """
    Find the nearest empty cell in a clockwise spiral pattern starting from (x, y).
    """
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction_index = 0
    steps_same_direction = 1
    steps_left_same_direction = 1
    grid_size_x, grid_size_y = len(grid), len(grid[0])

    while True:
        if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
            if grid[x][y] is None:
                return x, y

        x, y = x + directions[direction_index][0], y + directions[direction_index][1]
        steps_left_same_direction -= 1

        if steps_left_same_direction == 0:
            direction_index = (direction_index + 1) % 4
            if direction_index % 2 == 0:
                steps_same_direction += 1
            steps_left_same_direction = steps_same_direction


if __name__ == "__main__":
    # Load the data
    daily_change_df = pd.read_csv("sp500_daily_change.csv", index_col=0)
    correlation_matrix = pd.read_csv("sp500_correlation_matrix.csv", index_col=0)
    with open("sp500_market_caps.json", "r") as f:
        market_caps = json.load(f)
    print(f"Loaded data of shape {daily_change_df.shape}")

    # Remove 3 of the smallest stocks to make the grid a nice round number (500 = 20 * 25)
    smallest_stocks = sorted(market_caps, key=market_caps.get)[:3]
    daily_change_df.drop(columns=smallest_stocks, inplace=True)
    correlation_matrix.drop(
        index=smallest_stocks, columns=smallest_stocks, inplace=True
    )
    for stock in smallest_stocks:
        del market_caps[stock]

    # Heirarchical clustering
    distances = squareform(1 - correlation_matrix.abs())
    linkage_matrix = linkage(distances, method="ward")
    clusters = fcluster(linkage_matrix, t=16, criterion="maxclust")
    print("Performed heirarchical clustering")

    # Generate the weights
    market_caps_series = pd.Series(market_caps)
    weights = market_caps_series / market_caps_series.sum()

    # Force-Directed Graph
    G = nx.Graph()

    for i, stock in enumerate(correlation_matrix.columns):
        G.add_node(stock, cluster=clusters[i], weight=weights[i])

    for i, stock1 in enumerate(correlation_matrix.columns):
        for j, stock2 in enumerate(correlation_matrix.columns):
            if i < j:
                G.add_edge(
                    stock1, stock2, weight=1 - correlation_matrix.loc[stock1, stock2]
                )

    pos = nx.spring_layout(G, seed=42, weight="weight", k=0.5, iterations=100)
    print("Generated force-directed graph")

    # Place the stocks in a grid
    grid_positions = np.array([(i // 25, i % 25) for i in range(500)])
    pos_df = pd.DataFrame(pos).T
    pos_df["grid_x"] = pd.cut(
        pos_df[0], bins=np.linspace(0, 1, 21), labels=False, include_lowest=True
    )
    pos_df["grid_y"] = pd.cut(
        pos_df[1], bins=np.linspace(0, 1, 26), labels=False, include_lowest=True
    )

    # Check for NaN values and replace them with appropriate indices
    pos_df.loc[pos_df["grid_x"].isna(), "grid_x"] = 0
    pos_df.loc[pos_df["grid_y"].isna(), "grid_y"] = 0

    # Convert 'grid_x' and 'grid_y' columns to integers using applymap()
    pos_df = pos_df.applymap(lambda x: int(x))

    # Place stocks in the 20x25 grid
    grid = [[None for _ in range(25)] for _ in range(20)]

    for stock, row in pos_df.iterrows():
        x, y = row["grid_x"], row["grid_y"]
        if grid[x][y] is None:
            grid[x][y] = stock
        else:
            x, y = find_nearest_empty_cell(x, y, grid)
            grid[x][y] = stock
    print("Placed stocks in a grid")


    # Convert the grid to a DataFrame and save it to a CSV file
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv("sp500_grid.csv", index=False)
    print("Saved grid to sp500_grid.csv")
