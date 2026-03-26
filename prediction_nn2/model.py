"""Define a small MLP model used by the qmodel training loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MlpConfig:
    """Hold MLP hyperparameters for the prediction model."""

    in_dim: int
    hidden_dims: list[int]
    dropout: float
    dtype: torch.dtype


class MlpRegressor(nn.Module):
    """Map feature vectors into a single-step regression target."""

    def __init__(self, config: MlpConfig) -> None:
        """Build the MLP stack from the provided configuration."""
        super().__init__()

        # Force all hidden layer widths to the upgraded capacity requested by the pipeline milestone.
        hidden_dim = 512

        # Build a list of linear blocks according to hidden_dims.
        layers: list[nn.Module] = []
        prev = int(config.in_dim)
        for _h in list(config.hidden_dims):
            # Add linear + activation + dropout as one semantic block.
            layers.append(nn.Linear(prev, int(hidden_dim)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(float(config.dropout)))
            prev = int(hidden_dim)

        # Add the final projection into a single regression output.
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP forward pass and return a (N, 1) prediction."""
        # Flatten sequence inputs into one feature vector per sample.
        if x.dim() > 2:
            x = x.reshape(int(x.shape[0]), -1)

        # Apply the sequential network on the input batch.
        out = self.net(x)
        return out
