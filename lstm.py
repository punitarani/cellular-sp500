"""LSTM Model for Grid Weights"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class LSTMModel(nn.Module):
    """LSTM model for grid weights."""

    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        """
        Constructor for LSTMModel.

        Args:
            input_size (int): Number of features in the input.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            output_size (int): Number of features in the output.
        """

        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer used to process the input
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer used to map the LSTM output to the output_size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_lstm_model(
    model: LSTMModel,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.01,
) -> None:
    """
    Train the LSTM model.

    Args:
        model (LSTMModel): LSTM model to train.
        X (np.ndarray): Input data of shape (batch_size, seq_len, input_size).
        y (np.ndarray): Target data of shape (batch_size, output_size).
        epochs (int): Number of epochs to train for.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        None
    """

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs), desc="Training", ncols=100, leave=True):
        inputs = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
