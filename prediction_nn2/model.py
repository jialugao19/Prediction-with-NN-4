"""Define the sequence model used by the qmodel training loop."""

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


@dataclass(frozen=True)
class GruMlpConfig:
    """Hold GRU+MLP hyperparameters for the prediction model."""

    input_size: int
    hidden_size: int
    num_layers: int
    bidirectional: bool
    rnn_dropout: float
    mlp_hidden_dims: list[int]
    mlp_dropout: float
    dtype: torch.dtype


class GruMlpRegressor(nn.Module):
    """Encode a (T,F) sequence with GRU and regress with a small MLP."""

    def __init__(self, config: GruMlpConfig) -> None:
        """Build a GRU encoder plus an MLP head from the provided configuration."""
        super().__init__()

        # Build the GRU encoder that processes (B,T,F) inputs.
        self.rnn = nn.GRU(
            input_size=int(config.input_size),
            hidden_size=int(config.hidden_size),
            num_layers=int(config.num_layers),
            bidirectional=bool(config.bidirectional),
            dropout=float(config.rnn_dropout),
            batch_first=True,
        )

        # Build the MLP head that maps last_hidden into a scalar prediction.
        mlp_width = 512
        layers: list[nn.Module] = []
        prev = int(config.hidden_size) * (2 if bool(config.bidirectional) else 1)
        for _h in list(config.mlp_hidden_dims):
            # Add linear + activation + dropout as one semantic block.
            layers.append(nn.Linear(int(prev), int(mlp_width)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(float(config.mlp_dropout)))
            prev = int(mlp_width)
        layers.append(nn.Linear(int(prev), 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run GRU then MLP and return a (B, 1) prediction."""
        # Encode the sequence with GRU and take last-layer last_hidden.
        _out, h_n = self.rnn(x)
        last_hidden = h_n[-1]

        # Apply the MLP head on the encoded representation.
        pred = self.mlp(last_hidden)
        return pred
