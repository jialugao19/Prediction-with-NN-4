"""Write and read streaming predict outputs as parquet chunks with a manifest."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class PredictChunksManifest:
    """Describe a parquet-chunk prediction dataset written by the evaluator."""

    format: str
    columns: list[str]
    row_count: int
    chunk_row_count: int
    chunk_count: int
    chunk_files: list[str]
    stream_write_seconds: float
    iter: int
    group: str
    sort_key: list[str]
    date_min: int
    date_max: int

    def to_dict(self) -> dict[str, object]:
        """Convert the manifest into a stable YAML-serializable dict."""
        # Serialize with plain Python scalars for yaml.safe_dump.
        return {
            "format": str(self.format),
            "columns": list(self.columns),
            "row_count": int(self.row_count),
            "chunk_row_count": int(self.chunk_row_count),
            "chunk_count": int(self.chunk_count),
            "chunk_files": list(self.chunk_files),
            "stream_write_seconds": float(self.stream_write_seconds),
            "iter": int(self.iter),
            "group": str(self.group),
            "sort_key": list(self.sort_key),
            "date_min": int(self.date_min),
            "date_max": int(self.date_max),
        }


@dataclass(frozen=True)
class LivePredictChunksManifest:
    """Describe one legacy live-style parquet-chunk prediction dataset."""

    format: str
    columns: list[str]
    row_count: int
    chunk_row_count: int
    chunk_count: int
    chunk_files: list[str]
    stream_write_seconds: float
    iter: int
    group: str
    sort_key: list[str]
    date_min: int
    date_max: int

    def to_dict(self) -> dict[str, object]:
        """Convert the manifest into a stable YAML-serializable dict."""
        # Serialize with plain Python scalars for yaml.safe_dump.
        return {
            "format": str(self.format),
            "columns": list(self.columns),
            "row_count": int(self.row_count),
            "chunk_row_count": int(self.chunk_row_count),
            "chunk_count": int(self.chunk_count),
            "chunk_files": list(self.chunk_files),
            "stream_write_seconds": float(self.stream_write_seconds),
            "iter": int(self.iter),
            "group": str(self.group),
            "sort_key": list(self.sort_key),
            "date_min": int(self.date_min),
            "date_max": int(self.date_max),
        }


InferenceChunksManifest = LivePredictChunksManifest


class PredictChunkWriter:
    """Stream prediction rows to disk as parquet chunks and write a manifest at the end."""

    def __init__(self, *, iter_dir: Path, it: int, chunk_row_count: int, group: str) -> None:
        """Initialize a chunk writer rooted at an evaluator iter directory."""
        # Create output directories and initialize counters.
        self._iter_dir = Path(iter_dir)
        self._it = int(it)
        self._chunk_row_count = int(chunk_row_count)
        self._group = str(group)
        self._chunk_dir = Path(self._iter_dir) / "predict_chunks"
        self._chunk_dir.mkdir(parents=True, exist_ok=True)

        # Track buffered columns so we can flush in large contiguous writes.
        self._buf_row_id: list[np.ndarray] = []
        self._buf_pred: list[np.ndarray] = []
        self._buf_tgt: list[np.ndarray] = []
        self._buf_code: list[np.ndarray] = []
        self._buf_date: list[np.ndarray] = []
        self._buf_time: list[np.ndarray] = []
        self._buf_rows = 0

        # Track manifest metadata as we stream.
        self._next_row_id = 0
        self._part = 0
        self._row_count = 0
        self._chunk_files: list[str] = []
        self._date_min: int | None = None
        self._date_max: int | None = None
        self._write_seconds = 0.0

    @property
    def write_seconds(self) -> float:
        """Return total wall time spent writing parquet chunks."""
        # Expose accumulated IO time for perf_audit.yaml.
        return float(self._write_seconds)

    @property
    def row_count(self) -> int:
        """Return total streamed row count."""
        # Expose a scalar so callers can record it in perf audits.
        return int(self._row_count)

    @property
    def chunk_files(self) -> list[str]:
        """Return the list of chunk files written so far (relative paths)."""
        # Return a shallow copy to prevent accidental mutation.
        return list(self._chunk_files)

    def append(self, pred, target, meta) -> None:
        """Append one batch of (prediction, target, meta) rows into the chunk buffer."""
        # Convert tensors into numpy arrays with stable shapes and dtypes.
        pred_np = pred.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()
        meta_np = meta.detach().cpu().numpy()

        # Flatten prediction/target to 1D arrays and require meta schema [code,date,time].
        pred_1d = pred_np.reshape(-1).astype(np.float32, copy=False)
        tgt_1d = tgt_np.reshape(-1).astype(np.float32, copy=False)
        meta_2d = meta_np.reshape(-1, 3).astype(np.int64, copy=False)

        # Build a monotonic row_id block and update date range stats.
        n = int(pred_1d.shape[0])
        row_id = np.arange(int(self._next_row_id), int(self._next_row_id) + int(n), dtype=np.int64)
        self._next_row_id += int(n)
        self._row_count += int(n)
        self._buf_rows += int(n)

        # Track min/max dates for manifest summary.
        dates = meta_2d[:, 1]
        dmin = int(dates.min()) if int(dates.shape[0]) > 0 else None
        dmax = int(dates.max()) if int(dates.shape[0]) > 0 else None
        if dmin is not None:
            self._date_min = int(dmin) if self._date_min is None else int(min(int(self._date_min), int(dmin)))
            self._date_max = int(dmax) if self._date_max is None else int(max(int(self._date_max), int(dmax)))

        # Append arrays into per-column buffers for the next flush.
        self._buf_row_id.append(row_id)
        self._buf_pred.append(pred_1d)
        self._buf_tgt.append(tgt_1d)
        self._buf_code.append(meta_2d[:, 0])
        self._buf_date.append(meta_2d[:, 1])
        self._buf_time.append(meta_2d[:, 2])

        # Flush when the buffer reaches the configured chunk size.
        if int(self._buf_rows) >= int(self._chunk_row_count):
            self._flush_one_part()

    def close(self) -> Path:
        """Flush remaining buffered rows and write the final manifest YAML."""
        # Flush any remaining rows as the last part.
        if int(self._buf_rows) > 0:
            self._flush_one_part()

        # Write a manifest that points to the relative chunk files.
        manifest = PredictChunksManifest(
            format="parquet_chunks",
            columns=["row_id", "prediction", "target", "code", "date", "time"],
            row_count=int(self._row_count),
            chunk_row_count=int(self._chunk_row_count),
            chunk_count=int(len(self._chunk_files)),
            chunk_files=list(self._chunk_files),
            stream_write_seconds=float(self._write_seconds),
            iter=int(self._it),
            group=str(self._group),
            sort_key=["row_id"],
            date_min=int(self._date_min) if self._date_min is not None else 0,
            date_max=int(self._date_max) if self._date_max is not None else 0,
        )
        manifest_path = Path(self._iter_dir) / "predict_manifest.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
        return manifest_path

    def _flush_one_part(self) -> None:
        """Flush current column buffers into one parquet part file."""
        # Concatenate buffered arrays into one contiguous dataframe.
        t0 = time.perf_counter()
        df = pd.DataFrame(
            {
                "row_id": np.concatenate(self._buf_row_id, axis=0),
                "prediction": np.concatenate(self._buf_pred, axis=0),
                "target": np.concatenate(self._buf_tgt, axis=0),
                "code": np.concatenate(self._buf_code, axis=0),
                "date": np.concatenate(self._buf_date, axis=0),
                "time": np.concatenate(self._buf_time, axis=0),
            }
        )

        # Write the parquet part to the chunk directory with a stable naming scheme.
        filename = f"part_{int(self._part):06d}.parquet"
        out_path = Path(self._chunk_dir) / filename
        df.to_parquet(out_path.as_posix(), index=False)
        self._chunk_files.append(str(Path("predict_chunks") / filename))
        self._part += 1

        # Reset buffers for the next part and accumulate IO wall time.
        self._buf_row_id.clear()
        self._buf_pred.clear()
        self._buf_tgt.clear()
        self._buf_code.clear()
        self._buf_date.clear()
        self._buf_time.clear()
        self._buf_rows = 0
        self._write_seconds += float(time.perf_counter() - t0)


class LivePredictChunkWriter:
    """Stream legacy live-style prediction rows to disk as parquet chunks."""

    def __init__(self, *, iter_dir: Path, it: int, chunk_row_count: int, group: str) -> None:
        """Initialize one legacy live chunk writer rooted at an evaluator iter directory."""
        # Create output directories and initialize counters.
        self._iter_dir = Path(iter_dir)
        self._it = int(it)
        self._chunk_row_count = int(chunk_row_count)
        self._group = str(group)
        self._chunk_dir = Path(self._iter_dir) / "predict_chunks"
        self._chunk_dir.mkdir(parents=True, exist_ok=True)

        # Track buffered columns so we can flush in large contiguous writes.
        self._buf_pred: list[np.ndarray] = []
        self._buf_code: list[np.ndarray] = []
        self._buf_date: list[np.ndarray] = []
        self._buf_time: list[np.ndarray] = []
        self._buf_rows = 0

        # Track manifest metadata as we stream.
        self._part = 0
        self._row_count = 0
        self._chunk_files: list[str] = []
        self._date_min: int | None = None
        self._date_max: int | None = None
        self._write_seconds = 0.0

    @property
    def write_seconds(self) -> float:
        """Return total wall time spent writing parquet chunks."""
        # Expose accumulated IO time for perf_audit.yaml.
        return float(self._write_seconds)

    @property
    def row_count(self) -> int:
        """Return total streamed row count."""
        # Expose a scalar so callers can record it in perf audits.
        return int(self._row_count)

    @property
    def chunk_files(self) -> list[str]:
        """Return the list of chunk files written so far (relative paths)."""
        # Return a shallow copy to prevent accidental mutation.
        return list(self._chunk_files)

    def append(self, pred, meta) -> None:
        """Append one batch of (prediction, meta) rows into the chunk buffer."""
        # Convert tensors into numpy arrays with stable shapes and dtypes.
        pred_np = pred.detach().cpu().numpy()
        meta_np = meta.detach().cpu().numpy()

        # Flatten prediction to 1D arrays and require meta schema [code,date,time].
        pred_1d = pred_np.reshape(-1).astype(np.float32, copy=False)
        meta_2d = meta_np.reshape(-1, 3).astype(np.int64, copy=False)

        # Update row counters and date range stats.
        n = int(pred_1d.shape[0])
        self._row_count += int(n)
        self._buf_rows += int(n)

        # Track min/max dates for manifest summary.
        dates = meta_2d[:, 1]
        dmin = int(dates.min()) if int(dates.shape[0]) > 0 else None
        dmax = int(dates.max()) if int(dates.shape[0]) > 0 else None
        if dmin is not None:
            self._date_min = int(dmin) if self._date_min is None else int(min(int(self._date_min), int(dmin)))
            self._date_max = int(dmax) if self._date_max is None else int(max(int(self._date_max), int(dmax)))

        # Append arrays into per-column buffers for the next flush.
        self._buf_pred.append(pred_1d)
        self._buf_code.append(meta_2d[:, 0])
        self._buf_date.append(meta_2d[:, 1])
        self._buf_time.append(meta_2d[:, 2])

        # Flush when the buffer reaches the configured chunk size.
        if int(self._buf_rows) >= int(self._chunk_row_count):
            self._flush_one_part()

    def close(self) -> Path:
        """Flush remaining buffered rows and write the final legacy live manifest YAML."""
        # Flush any remaining rows as the last part.
        if int(self._buf_rows) > 0:
            self._flush_one_part()

        # Write a manifest that points to the relative chunk files.
        manifest = LivePredictChunksManifest(
            format="parquet_chunks",
            columns=["prediction", "code", "date", "time"],
            row_count=int(self._row_count),
            chunk_row_count=int(self._chunk_row_count),
            chunk_count=int(len(self._chunk_files)),
            chunk_files=list(self._chunk_files),
            stream_write_seconds=float(self._write_seconds),
            iter=int(self._it),
            group=str(self._group),
            sort_key=["date", "time", "code"],
            date_min=int(self._date_min) if self._date_min is not None else 0,
            date_max=int(self._date_max) if self._date_max is not None else 0,
        )
        manifest_path = Path(self._iter_dir) / "predict_live_manifest.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
        return manifest_path

    def _flush_one_part(self) -> None:
        """Flush current column buffers into one parquet part file."""
        # Concatenate buffered arrays into one contiguous dataframe.
        t0 = time.perf_counter()
        df = pd.DataFrame(
            {
                "prediction": np.concatenate(self._buf_pred, axis=0),
                "code": np.concatenate(self._buf_code, axis=0),
                "date": np.concatenate(self._buf_date, axis=0),
                "time": np.concatenate(self._buf_time, axis=0),
            }
        )

        # Write the parquet part to the chunk directory with a stable naming scheme.
        filename = f"part_{int(self._part):06d}.parquet"
        out_path = Path(self._chunk_dir) / filename
        df.to_parquet(out_path.as_posix(), index=False)
        self._chunk_files.append(str(Path("predict_chunks") / filename))
        self._part += 1

        # Reset buffers for the next part and accumulate IO wall time.
        self._buf_pred.clear()
        self._buf_code.clear()
        self._buf_date.clear()
        self._buf_time.clear()
        self._buf_rows = 0
        self._write_seconds += float(time.perf_counter() - t0)


class InferenceChunkWriter(LivePredictChunkWriter):
    """Stream inference rows to disk as parquet chunks and write an inference manifest."""

    def close(self) -> Path:
        """Flush remaining buffered rows and write the final inference manifest YAML."""
        # Flush any remaining rows as the last part.
        if int(self._buf_rows) > 0:
            self._flush_one_part()

        # Write a manifest that points to the relative chunk files.
        manifest = InferenceChunksManifest(
            format="parquet_chunks",
            columns=["prediction", "code", "date", "time"],
            row_count=int(self._row_count),
            chunk_row_count=int(self._chunk_row_count),
            chunk_count=int(len(self._chunk_files)),
            chunk_files=list(self._chunk_files),
            stream_write_seconds=float(self._write_seconds),
            iter=int(self._it),
            group=str(self._group),
            sort_key=["date", "time", "code"],
            date_min=int(self._date_min) if self._date_min is not None else 0,
            date_max=int(self._date_max) if self._date_max is not None else 0,
        )
        manifest_path = Path(self._iter_dir) / "inference_manifest.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
        return manifest_path
