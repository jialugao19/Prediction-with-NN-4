"""Provide a map-style dataset that returns prepacked CPU tensors for qmodel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class NpzDatasetSpec:
    """Describe where to load group arrays and how to materialize tensors."""

    data_dir: Path
    pin_memory: bool
    window_size: int


class Stock1mNpzDataset:
    """Load precomputed feature/label/meta arrays from raw binaries for qmodel training and eval."""

    def __init__(self, group: str, dtype: torch.dtype, spec: NpzDatasetSpec) -> None:
        """Load group arrays from disk via memmap to avoid multi-year RAM blowups."""
        # Resolve group name and load the shared storage metadata.
        group = str(group)
        meta_path = Path(spec.data_dir) / "meta.yaml"
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        storage = dict(meta["storage"])
        groups = dict(storage["groups"])
        g = dict(groups[group])

        # Memory-map x/y/meta with shapes recorded in meta.yaml.
        x_path = Path(spec.data_dir) / str(g["x"])
        y_path = Path(spec.data_dir) / str(g["y"])
        meta_bin_path = Path(spec.data_dir) / str(g["meta"])
        rows = int(g["rows"])
        feature_dim = int(g["feature_dim"])
        window_size = int(spec.window_size)
        x = np.memmap(x_path, mode="r", dtype=np.float32, shape=(int(rows), int(feature_dim)))
        y = np.memmap(y_path, mode="r", dtype=np.float32, shape=(int(rows), 1))
        meta_arr = np.memmap(meta_bin_path, mode="r", dtype=np.int64, shape=(int(rows), 3))

        # Wrap the memmaps with tensors so DataLoader can batch without reading everything upfront.
        self._x = torch.from_numpy(x).to(dtype=dtype)
        self._y = torch.from_numpy(y).to(dtype=dtype)
        self._meta = torch.from_numpy(meta_arr)

        # Validate expected shapes to match qmodel evaluator conventions.
        if self._x.dim() != 2:
            raise RuntimeError(f"x must be 2D, got: {tuple(self._x.shape)}")
        if self._y.dim() != 2 or self._y.shape[1] != 1:
            raise RuntimeError(f"y must be (N,1), got: {tuple(self._y.shape)}")
        if self._meta.dim() != 2 or self._meta.shape[1] != 3:
            raise RuntimeError(f"meta must be (N,3) as [code,date,time], got: {tuple(self._meta.shape)}")

        # Persist sequence metadata so batching can materialize trailing windows on demand.
        self._feature_dim = int(feature_dim)
        self._window_size = int(window_size)
        if int(window_size) > 1:
            valid_end = _load_or_build_valid_end_cache(
                data_dir=Path(spec.data_dir),
                group=str(group),
                window_size=int(window_size),
                rows=int(rows),
                meta_bin_path=meta_bin_path,
                meta_arr=meta_arr,
            )
            self._valid_end = valid_end
            self._window_offsets = torch.arange(-(int(window_size) - 1), 1, dtype=torch.int64)
        else:
            self._valid_end = None
            self._window_offsets = None

    def __len__(self) -> int:
        """Return the number of samples in the split."""
        # Return either raw rows or valid window endpoints depending on mode.
        if self._valid_end is None:
            return int(self._x.shape[0])
        return int(self._valid_end.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one sample; DataLoader should prefer __getitems__ for batching."""
        # Resolve the backing row index for this logical sample.
        row = self._resolve_row_index(int(index))

        # Slice the tensors for the given index and materialize row tensors.
        if int(self._window_size) > 1:
            st = int(row - int(self._window_size) + 1)
            x = self._x[int(st) : int(row) + 1].reshape(int(self._window_size), int(self._feature_dim))
        else:
            x = self._x[int(row)]
        y = self._y[int(row)]
        meta = self._meta[int(row)]
        return x, y, meta

    def __getitems__(self, indices) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a pre-batched tuple (x, y, meta) for DataLoader fast-path."""
        # Convert indices into a torch tensor for advanced indexing.
        idx = torch.as_tensor(indices, dtype=torch.int64)

        # Resolve logical sample indices into raw row endpoints.
        row_idx = self._resolve_row_indices(idx)

        # Gather rows into contiguous tensors for the batch.
        if int(self._window_size) > 1:
            assert self._window_offsets is not None
            gather_idx = row_idx.unsqueeze(1) + self._window_offsets.unsqueeze(0)
            x = self._x.index_select(0, gather_idx.reshape(-1)).reshape(int(idx.shape[0]), int(self._window_size), int(self._feature_dim))
        else:
            x = self._x.index_select(0, row_idx)
        y = self._y.index_select(0, row_idx)
        meta = self._meta.index_select(0, row_idx)
        return x, y, meta

    def _resolve_row_index(self, index: int) -> int:
        """Map one logical sample index into the backing raw row index."""
        # Return the original row directly when sequence mode is disabled.
        if self._valid_end is None:
            return int(index)

        # Use the cached valid end-row array when sequence mode is enabled.
        return int(self._valid_end[int(index)])

    def _resolve_row_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Map batched logical sample indices into backing raw row indices."""
        # Return the original rows directly when sequence mode is disabled.
        if self._valid_end is None:
            return indices

        # Materialize the selected valid end rows as torch int64 indices.
        raw = self._valid_end[indices.cpu().numpy()]
        return torch.as_tensor(raw, dtype=torch.int64)


def _valid_end_cache_paths(data_dir: Path, group: str, window_size: int) -> tuple[Path, Path]:
    """Resolve the data and metadata paths for one valid-end cache."""
    # Keep both cache files adjacent so invalidation can be checked cheaply.
    stem = f"{str(group)}_window{int(window_size)}_valid_end"
    return Path(data_dir) / f"{stem}.i32", Path(data_dir) / f"{stem}.yaml"


def _valid_end_cache_contract(group: str, window_size: int, rows: int, meta_bin_path: Path) -> dict[str, object]:
    """Build the cache contract used to validate one valid-end file."""
    # Record the source meta file identity so stale caches can be detected.
    stat = Path(meta_bin_path).stat()
    return {
        "group": str(group),
        "window_size": int(window_size),
        "rows": int(rows),
        "meta_size_bytes": int(stat.st_size),
        "meta_mtime_ns": int(stat.st_mtime_ns),
    }


def _load_or_build_valid_end_cache(
    *,
    data_dir: Path,
    group: str,
    window_size: int,
    rows: int,
    meta_bin_path: Path,
    meta_arr: np.ndarray,
) -> np.ndarray:
    """Load one valid-end cache when it matches the current meta file, else rebuild it."""
    # Resolve cache paths and the expected cache contract before touching disk.
    valid_path, meta_path = _valid_end_cache_paths(Path(data_dir), str(group), int(window_size))
    expected_contract = _valid_end_cache_contract(str(group), int(window_size), int(rows), Path(meta_bin_path))

    # Reuse the cache only when both files exist and the metadata contract matches exactly.
    if valid_path.exists() and meta_path.exists():
        cache_contract = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        if dict(cache_contract) == dict(expected_contract):
            return np.fromfile(valid_path, dtype=np.int32)

    # Rebuild the cache from meta_arr and overwrite both cache files atomically enough for local use.
    valid_end = _build_valid_window_end_indices(meta_arr, int(window_size))
    valid_end.astype(np.int32, copy=False).tofile(valid_path)
    meta_path.write_text(yaml.safe_dump(expected_contract, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return valid_end


def _build_valid_window_end_indices(meta_arr: np.ndarray, window_size: int) -> np.ndarray:
    """Compute row indices whose trailing run forms one contiguous stock-day window."""
    # Short-circuit empty and too-short arrays.
    rows = int(meta_arr.shape[0])
    if int(rows) < int(window_size):
        return np.empty((0,), dtype=np.int32)

    # Decode stock/date/time metadata into one continuous intraday minute axis.
    codes = meta_arr[:, 0].astype(np.int64, copy=False)
    dates = meta_arr[:, 1].astype(np.int64, copy=False)
    minutes = _time_int_to_session_minute(meta_arr[:, 2].astype(np.int64, copy=False))

    # Mark rows that continue the same stock/date minute-by-minute run.
    contiguous = np.zeros((int(rows),), dtype=bool)
    contiguous[1:] = (codes[1:] == codes[:-1]) & (dates[1:] == dates[:-1]) & (minutes[1:] == minutes[:-1] + 1)

    # Convert breakpoints into run starts and trailing run lengths.
    starts = np.zeros((int(rows),), dtype=np.int64)
    starts[1:] = np.where(contiguous[1:], -1, np.arange(1, int(rows), dtype=np.int64))
    starts = np.maximum.accumulate(starts)
    run_lengths = np.arange(int(rows), dtype=np.int64) - starts + 1

    # Keep row endpoints whose contiguous run is long enough for one window.
    valid = np.where(run_lengths >= int(window_size))[0].astype(np.int32, copy=False)
    return valid


def _time_int_to_session_minute(time_int: np.ndarray) -> np.ndarray:
    """Convert hhmmss integers into a continuous minute index across both sessions."""
    # Split hhmmss into hour/minute components.
    hour = time_int // 10000
    minute = (time_int // 100) % 100
    minute_of_day = hour * 60 + minute

    # Map morning [09:30,11:29] and afternoon [13:00,14:59] into [0,239].
    out = np.empty_like(minute_of_day, dtype=np.int64)
    morning = minute_of_day < (12 * 60)
    out[morning] = minute_of_day[morning] - (9 * 60 + 30)
    out[~morning] = 120 + minute_of_day[~morning] - (13 * 60)
    return out
