"""Define the dataset protocol used by qmodel data loaders."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class QDataset(Protocol):
    """Describe the dataset methods required by qmodel trainers."""

    def __init__(self, group: str, dtype: torch.dtype, *args, **kwargs):
        """Initialize a dataset for one split and dtype."""
        # Protocol method only declares the constructor contract.
        ...

    def __len__(self) -> int:
        """Return the number of rows in this dataset."""
        # Protocol method only declares the length contract.
        ...

    def __getitem__(self, index) -> Any:
        """Return one sample by index."""
        # Protocol method only declares the scalar access contract.
        ...

    def __getitems__(self, indices) -> Any:
        """Return one batched sample by indices."""
        # Protocol method only declares the batched access contract.
        ...

