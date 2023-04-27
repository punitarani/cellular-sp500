"""Generate the grid"""

import itertools
import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def load_grid():
    return pd.read_csv("sp500_grid.csv")


def get_neighbors(grid, row, col):
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    grid_shape = grid.shape
    return [(row + i, col + j) for i, j in neighbor_offsets if
            0 <= row + i < grid_shape[0] and 0 <= col + j < grid_shape[1]]


def plot_grid(grid, sorted_cluster_df):
    fig, ax = plt.subplots(figsize=(15, 12))
    colors = plt.cm.get_cmap("tab20", len(sorted_cluster_df["cluster"].unique()))

    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell is not None:
                stock, cluster = cell
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
    grid_size = (len(grid), len(grid[0]))
    grid_center = (grid_size[0] // 2, grid_size[1] // 2)

    points_to_check = [(grid_center[0] + x * grid_center[0] // 2, grid_center[1] + y * grid_center[1] // 2) for x, y in
                       [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, 0)]]

    for point in points_to_check:
        if grid[point[0]][point[1]] is None:
            return point

    radius = 1
    while True:
        for neighbor in get_neighbors_with_radius(grid_center, radius, grid_size):
            if grid[neighbor[0]][neighbor[1]] is None:
                return neighbor
        radius += 1


def get_neighbors_with_radius(center, radius, grid_size):
    offsets = [(i, j) for i in range(-radius, radius + 1) for j in range(-radius, radius + 1) if
               abs(i) == radius or abs(j) == radius]
    return [(center[0] + i, center[1] + j) for i, j in offsets if
            0 <= center[0] + i < grid_size[0] and 0 <= center[1] + j < grid_size[1]]


def place_cluster(grid, sorted_cluster_df, cluster):
    """Place stocks in the 20x25 grid around the first stock of each cluster."""
    cluster_stocks = sorted_cluster_df[sorted_cluster_df['cluster'] == cluster]

    # Find a suitable center for the cluster
    center = find_center(grid)

    # Place the first stock in the cluster at the center
    grid[center[0]][center[1]] = (cluster_stocks.iloc[0]['symbol'], cluster)

    # Function to generate positions in a spiral pattern
    def spiral_positions(center):
        for radius in itertools.count(start=1):
            for neighbor in get_neighbors_with_radius(center, radius, (len(grid), len(grid[0]))):
                yield neighbor

    # Place the remaining stocks in the cluster
    spiral_gen = spiral_positions(center)
    for i in range(1, len(cluster_stocks)):
        stock = cluster_stocks.iloc[i]['symbol']

        # Find the next empty position in the spiral pattern
        new_pos = next(pos for pos in spiral_gen if grid[pos[0]][pos[1]] is None)

        grid[new_pos[0]][new_pos[1]] = (stock, cluster)


def evaluate_placement(grid, sorted_cluster_df):
    """Evaluate the placement of clusters by calculating the average distance between the stocks in the same cluster."""

    total_distance = 0
    total_stocks = 0

    for cluster in sorted_cluster_df['cluster'].unique():
        cluster_stocks = sorted_cluster_df[sorted_cluster_df['cluster'] == cluster]
        cluster_positions = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if
                             grid[i][j] is not None and grid[i][j][1] == cluster]

        distance_sum = 0
        stocks_count = len(cluster_positions)

        for stock1_pos, stock2_pos in itertools.combinations(cluster_positions, 2):
            distance_sum += abs(stock1_pos[0] - stock2_pos[0]) + abs(stock1_pos[1] - stock2_pos[1])

        average_distance = distance_sum / (stocks_count * (stocks_count - 1) // 2)
        total_distance += distance_sum
        total_stocks += stocks_count

    return total_distance / (total_stocks * (total_stocks - 1) // 2)


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

    # Cluster around the center
    for cluster in sorted_cluster_df['cluster'].unique():
        place_cluster(grid, sorted_cluster_df, cluster)
    print("Placed stocks in a grid")

    # Evaluate the placement
    score = evaluate_placement(grid, sorted_cluster_df)
    print(f"Placement evaluation score: {score}")

    # Convert the grid to a DataFrame and save it to a CSV file
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv("sp500_grid.csv", index=False)
    print("Saved grid to sp500_grid.csv")

    plot_grid(grid, sorted_cluster_df)
