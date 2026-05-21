"""Provide a resume-aware sampler without binary extension dependencies."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Sampler


class CustomSampler(Sampler[int]):
    """Yield deterministic shuffled or sequential indices for train and eval."""

    def __init__(self, dataset, seed: int, shuffle: bool = True, infinite: bool = False):
        """Store sampler settings and resume state."""
        # Record immutable sampling inputs.
        self.dataset = dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.infinite = bool(infinite)

        # Track resume offset for the next iterator construction.
        self.not_yet_started = True
        self.start = 0

    def set_start_iter(self, iteration: int) -> None:
        """Record the absolute sample offset used for resume."""
        # Store the global sample offset consumed before the next yielded sample.
        self.start = int(iteration)

    def __iter__(self):
        """Yield indices for one or more epochs."""
        # Initialize epoch and DDP position state.
        assert self.not_yet_started
        self.not_yet_started = False
        n_rows = int(len(self.dataset))
        chunk_size = 100_000
        start = int(self.start % n_rows)
        epoch = int(self.start // n_rows)
        rank, world_size = _distributed_rank_state()

        # Produce epochs until the caller stops or finite eval completes.
        while True:
            if self.shuffle:
                yield from self._iter_shuffled_epoch(n_rows, chunk_size, start, epoch, rank, world_size)
            else:
                yield from self._iter_sequential_epoch(n_rows, start, rank, world_size)

            # Stop after one pass for finite evaluation.
            if not self.infinite:
                break
            epoch += 1
            start = 0

    def __len__(self) -> int:
        """Return the local sampler length for one epoch."""
        # Keep the historical DataLoader length contract.
        return int(len(self.dataset))

    def _iter_shuffled_epoch(
        self,
        n_rows: int,
        chunk_size: int,
        start: int,
        epoch: int,
        rank: int,
        world_size: int,
    ):
        """Yield a deterministic no-replacement affine permutation for one epoch."""
        # Build an epoch-specific bijection over dataset positions.
        stride = _coprime_stride(n_rows, self.seed + epoch * 1009)
        offset = int((self.seed * 6364136223846793005 + epoch * 1442695040888963407) % n_rows)

        # Vectorize permutation in chunks to keep Python overhead and memory bounded.
        for chunk_start in range(int(start), int(n_rows), int(chunk_size)):
            chunk_end = min(chunk_start + int(chunk_size), int(n_rows))
            positions = np.arange(chunk_start, chunk_end, dtype=np.int64)
            positions = positions[(positions % int(world_size)) == int(rank)]
            if positions.size == 0:
                continue
            indices = (positions * int(stride) + int(offset)) % int(n_rows)
            yield from indices.tolist()

    def _iter_sequential_epoch(self, n_rows: int, start: int, rank: int, world_size: int):
        """Yield sequential indices for one epoch with DDP striding."""
        # Align the first position for this rank after the resume offset.
        first = int(start) + ((int(rank) - int(start)) % int(world_size))
        yield from range(first, int(n_rows), int(world_size))


def _distributed_rank_state() -> tuple[int, int]:
    """Return current DDP rank and world size."""
    # Query torch.distributed only when the default process group exists.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    return 0, 1


def _coprime_stride(n_rows: int, seed: int) -> int:
    """Choose a deterministic stride that is coprime to the dataset length."""
    # Start from an odd seed-derived candidate so even dataset sizes are not immediately invalid.
    stride = int(seed % max(int(n_rows), 1))
    stride = max(1, stride)
    if stride % 2 == 0:
        stride += 1

    # Advance deterministically until the affine map is a bijection.
    while math.gcd(int(stride), int(n_rows)) != 1:
        stride += 2
        if stride >= int(n_rows):
            stride = 1
    return int(stride)

