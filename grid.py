"""Generate the grid"""

import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def load_grid():
    """Load the grid from the CSV file."""
    return pd.read_csv("sp500_grid.csv")


def get_neighbors(grid, row, col):
    """Get the neighbors of a cell in the grid."""
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            new_row, new_col = row + i, col + j
            if 0 <= new_row < grid.shape[0] and 0 <= new_col < grid.shape[1]:
                neighbors.append((new_row, new_col))
    return neighbors


def plot_grid(grid, sorted_cluster_df):
    """Plot the grid with the stocks colored by cluster."""
    fig, ax = plt.subplots(figsize=(15, 12))
    colors = plt.cm.get_cmap("tab20", len(sorted_cluster_df["cluster"].unique()))

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] is not None:
                stock, cluster = grid[i][j]
                color = colors(cluster - 1)
                rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                ax.annotate(stock, (j + 0.5, i + 0.5), textcoords="offset points", xytext=(0, 0), ha='center',
                            va='center', fontsize=8)

    ax.set_xticks(range(len(grid[0])), minor=False)
    ax.set_yticks(range(len(grid)), minor=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='major', color='black', linewidth=1)

    plt.show()


def find_center(grid):
    """Find suitable center for a cluster"""

    grid_size = (len(grid), len(grid[0]))
    grid_center = (grid_size[0] // 2, grid_size[1] // 2)

    # Check if grid center is empty
    if grid[grid_center[0]][grid_center[1]] is None:
        return grid_center

    # Check if mid point between grid center and corners is empty
    diag1 = (grid_center[0] // 2, grid_center[1] // 2)
    if grid[diag1[0]][diag1[1]] is None:
        return diag1
    diag2 = (grid_center[0] // 2, grid_center[1] + grid_center[1] // 2)
    if grid[diag2[0]][diag2[1]] is None:
        return diag2
    diag3 = (grid_center[0] + grid_center[0] // 2, grid_center[1] // 2)
    if grid[diag3[0]][diag3[1]] is None:
        return diag3
    diag4 = (grid_center[0] + grid_center[0] // 2, grid_center[1] + grid_center[1] // 2)
    if grid[diag4[0]][diag4[1]] is None:
        return diag4

    # Check if the mid point between grid center and the middle of the sides is empty
    side1 = (grid_center[0] // 2, grid_center[1])
    if grid[side1[0]][side1[1]] is None:
        return side1
    side2 = (grid_center[0], grid_center[1] // 2)
    if grid[side2[0]][side2[1]] is None:
        return side2
    side3 = (grid_center[0], grid_center[1] + grid_center[1] // 2)
    if grid[side3[0]][side3[1]] is None:
        return side3
    side4 = (grid_center[0] + grid_center[0] // 2, grid_center[1])
    if grid[side4[0]][side4[1]] is None:
        return side4

    # Expand outward in a spiral pattern from the grid center
    radius = 1
    while True:
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i == -radius or i == radius or j == -radius or j == radius:
                    new_row, new_col = grid_center[0] + i, grid_center[1] + j
                    if 0 <= new_row < grid_size[0] and 0 <= new_col < grid_size[1]:
                        if grid[new_row][new_col] is None:
                            return new_row, new_col
        radius += 1


def get_neighbors_with_radius(center, radius, grid_size):
    """Get the neighbors of a cell at a given radius."""
    neighbors = []

    # Traverse top and bottom edges
    for col_offset in range(-radius, radius + 1):
        top_neighbor = (center[0] - radius, center[1] + col_offset)
        bottom_neighbor = (center[0] + radius, center[1] + col_offset)

        if 0 <= top_neighbor[0] < grid_size[0] and 0 <= top_neighbor[1] < grid_size[1]:
            neighbors.append(top_neighbor)
        if 0 <= bottom_neighbor[0] < grid_size[0] and 0 <= bottom_neighbor[1] < grid_size[1]:
            neighbors.append(bottom_neighbor)

    # Traverse left and right edges (excluding corners)
    for row_offset in range(-radius + 1, radius):
        left_neighbor = (center[0] + row_offset, center[1] - radius)
        right_neighbor = (center[0] + row_offset, center[1] + radius)

        if 0 <= left_neighbor[0] < grid_size[0] and 0 <= left_neighbor[1] < grid_size[1]:
            neighbors.append(left_neighbor)
        if 0 <= right_neighbor[0] < grid_size[0] and 0 <= right_neighbor[1] < grid_size[1]:
            neighbors.append(right_neighbor)

    return neighbors


def place_cluster(grid, sorted_cluster_df, cluster):
    """Place stocks in the 20x25 grid around the first stock of each cluster."""
    cluster_stocks = sorted_cluster_df[sorted_cluster_df['cluster'] == cluster]

    # Find a suitable center for the cluster
    center = find_center(grid)

    # Place the first stock in the cluster at the center
    grid[center[0]][center[1]] = (cluster_stocks.iloc[0]['symbol'], cluster)

    # Place the remaining stocks in the cluster
    for i in range(1, len(cluster_stocks)):
        stock, new_pos = cluster_stocks.iloc[i]['symbol'], None
        radius = 1

        while new_pos is None:
            for neighbor in get_neighbors_with_radius(center, radius, (len(grid), len(grid[0]))):
                new_row, new_col = neighbor
                if grid[new_row][new_col] is None:
                    new_pos = new_row, new_col
                    break
            radius += 1

        grid[new_pos[0]][new_pos[1]] = (stock, cluster)


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

    # Hierarchical clustering
    distances = squareform(1 - correlation_matrix.abs())
    linkage_matrix = linkage(distances, method="ward")
    clusters = fcluster(linkage_matrix, t=16, criterion="maxclust")
    print("Performed hierarchical clustering")

    # Create a DataFrame containing stock symbol, cluster number, and market cap
    cluster_df = pd.DataFrame(
        {"symbol": list(market_caps.keys()), "cluster": clusters, "market_cap": list(market_caps.values())})

    # Calculate the mean of market caps within each cluster and sort the clusters based on the mean
    cluster_mean_df = cluster_df.groupby('cluster').agg({"market_cap": "mean"}).sort_values(by='market_cap',
                                                                                            ascending=False).reset_index()

    # Assign new cluster numbers based on the sorted order
    cluster_mean_df['new_cluster'] = range(1, len(cluster_mean_df) + 1)

    # Ensure both 'cluster' columns are of the same data type
    cluster_df['cluster'] = cluster_df['cluster'].astype(int)
    cluster_mean_df['cluster'] = cluster_mean_df['cluster'].astype(int)

    # Create a dictionary with the old and new cluster numbers
    cluster_map = dict(zip(cluster_mean_df['cluster'], cluster_mean_df['new_cluster']))

    # Update the cluster numbers in the cluster_df dataframe
    cluster_df['cluster'] = cluster_df['cluster'].map(cluster_map)

    # Sort the DataFrame based on cluster number and market cap in descending order
    sorted_cluster_df = cluster_df.sort_values(by=["cluster", "market_cap"], ascending=[True, False])

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
    symbol_to_cluster = dict(zip(sorted_cluster_df["symbol"], sorted_cluster_df["cluster"]))

    # Cluster around the center
    for cluster in sorted_cluster_df['cluster'].unique():
        place_cluster(grid, sorted_cluster_df, cluster)
    print("Placed stocks in a grid")

    # Convert the grid to a DataFrame and save it to a CSV file
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv("sp500_grid.csv", index=False)
    print("Saved grid to sp500_grid.csv")

    plot_grid(grid, sorted_cluster_df)
