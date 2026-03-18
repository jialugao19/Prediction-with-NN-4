"""Provide a map-style dataset that returns prepacked CPU tensors for qmodel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class NpzDatasetSpec:
    """Describe where to load group arrays and how to materialize tensors."""

    data_dir: Path
    pin_memory: bool


class Stock1mNpzDataset:
    """Load precomputed feature/label/meta arrays from NPZ for qmodel training and eval."""

    def __init__(self, group: str, dtype: torch.dtype, spec: NpzDatasetSpec) -> None:
        """Load group arrays from disk and keep them in CPU memory."""
        # Resolve file path for the requested split.
        group = str(group)
        path = Path(spec.data_dir) / f"{group}.npz"
        if not path.exists():
            raise FileNotFoundError(path.as_posix())

        # Load arrays and cast them into stable dtypes.
        with np.load(path, allow_pickle=False) as z:
            x = z["x"].astype(np.float32, copy=False)
            y = z["y"].astype(np.float32, copy=False)
            meta = z["meta"].astype(np.int64, copy=False)

        # Materialize tensors for fast __getitems__ batching.
        self._x = torch.from_numpy(x).to(dtype=dtype)
        self._y = torch.from_numpy(y).to(dtype=dtype)
        self._meta = torch.from_numpy(meta)
        self._pin_memory = bool(spec.pin_memory)

        # Validate expected shapes to match qmodel evaluator conventions.
        if self._x.dim() != 2:
            raise RuntimeError(f"x must be 2D, got: {tuple(self._x.shape)}")
        if self._y.dim() != 2 or self._y.shape[1] != 1:
            raise RuntimeError(f"y must be (N,1), got: {tuple(self._y.shape)}")
        if self._meta.dim() != 2 or self._meta.shape[1] != 3:
            raise RuntimeError(f"meta must be (N,3) as [code,date,time], got: {tuple(self._meta.shape)}")

    def __len__(self) -> int:
        """Return the number of samples in the split."""
        # Return row count from the feature tensor.
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one sample; DataLoader should prefer __getitems__ for batching."""
        # Slice the tensors for the given index and materialize row tensors.
        x = self._x[int(index)]
        y = self._y[int(index)]
        meta = self._meta[int(index)]

        # Optionally pin memory to support CUDA async transfer in GPU mode.
        if self._pin_memory:
            x = x.pin_memory()
            y = y.pin_memory()
            meta = meta.pin_memory()
        return x, y, meta

    def __getitems__(self, indices) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a pre-batched tuple (x, y, meta) for DataLoader fast-path."""
        # Convert indices into a torch tensor for advanced indexing.
        idx = torch.as_tensor(indices, dtype=torch.int64)

        # Gather rows into contiguous tensors for the batch.
        x = self._x.index_select(0, idx)
        y = self._y.index_select(0, idx)
        meta = self._meta.index_select(0, idx)

        # Optionally pin memory to match qmodel async CUDA trainer assumptions.
        if self._pin_memory:
            x = x.pin_memory()
            y = y.pin_memory()
            meta = meta.pin_memory()
        return x, y, meta

