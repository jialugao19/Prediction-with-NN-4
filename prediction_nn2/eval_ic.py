"""Compute multi-dimensional IC diagnostics from qmodel evaluator feather outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qmodel.util import merge_date_time_dataframe


@dataclass(frozen=True)
class EvalConfig:
    """Define evaluation IO and rolling-group knobs."""

    stock1m_dir: Path
    window_size: int
    step_size: int
    horizon_minutes: int


_ROLLING_GLOBAL_PRED_DF: pd.DataFrame | None = None
_ROLLING_GLOBAL_IDX_BY_DATE: dict[int, np.ndarray] | None = None
_ROLLING_GLOBAL_EVAL_CFG: EvalConfig | None = None


def _rolling_bins_merge(dst: dict[float, dict[str, float]], src: dict[float, dict[str, float]]) -> None:
    """Merge one rolling-bin accumulator into another in-place."""
    # Sum per-bin moments so multi-process partials can be aggregated without keeping raw windows.
    for k, v in src.items():
        if k not in dst:
            dst[k] = dict(v)
            continue
        acc = dst[k]
        for name in ["sum_center_rank", "sum_ic", "sum_ic2", "n_ic", "sum_rank_ic", "sum_rank_ic2", "n_rank_ic", "count"]:
            acc[name] = float(acc[name]) + float(v[name])


def _rolling_bins_to_curve(bins: dict[float, dict[str, float]]) -> pd.DataFrame:
    """Convert rolling-bin accumulators into the stable curve dataframe schema."""
    # Convert aggregated moments into the curve table expected by plotting/reporting.
    if len(bins) == 0:
        return _empty_group_schema()

    rows: list[dict[str, object]] = []
    for rank_bin in sorted(bins.keys()):
        # Convert sums into mean/std while guarding against negative variance from float drift.
        acc = bins[rank_bin]
        c = float(acc["count"])
        mean_center = float(acc["sum_center_rank"] / c)

        # Compute mean/std for IC metrics using finite-only counts to match pandas semantics.
        n_ic = float(acc["n_ic"])
        n_rank_ic = float(acc["n_rank_ic"])
        mean_ic = float(acc["sum_ic"] / n_ic) if n_ic > 0.0 else float("nan")
        var_ic = float(acc["sum_ic2"] / n_ic - mean_ic * mean_ic) if n_ic > 1.0 else float("nan")
        mean_rank_ic = float(acc["sum_rank_ic"] / n_rank_ic) if n_rank_ic > 0.0 else float("nan")
        var_rank_ic = float(acc["sum_rank_ic2"] / n_rank_ic - mean_rank_ic * mean_rank_ic) if n_rank_ic > 1.0 else float("nan")
        rows.append(
            {
                "group_center_rank": float(mean_center),
                "mean_ic": float(mean_ic),
                "std_ic": float(np.sqrt(max(var_ic, 0.0))) if np.isfinite(var_ic) else float("nan"),
                "mean_rank_ic": float(mean_rank_ic),
                "std_rank_ic": float(np.sqrt(max(var_rank_ic, 0.0))) if np.isfinite(var_rank_ic) else float("nan"),
                "count": int(c),
            }
        )
    out = pd.DataFrame(rows).sort_values("group_center_rank", kind="stable").reset_index(drop=True)
    return out


def _write_group_curve_summary(df: pd.DataFrame, out_yaml: Path, curve_name: str) -> None:
    """Write a compact YAML summary for one rolling IC curve."""
    # Build a small scalar summary so downstream review does not depend on giant raw tables.
    summary: dict[str, object] = {"curve": str(curve_name), "rows": int(df.shape[0])}

    # Extract scalar extrema and rank coverage when the curve is non-empty.
    if int(df.shape[0]) > 0:
        ranks = df["group_center_rank"].to_numpy(dtype=float)
        mean_ic = df["mean_ic"].to_numpy(dtype=float)
        mean_rank_ic = df["mean_rank_ic"].to_numpy(dtype=float)
        summary["center_rank_min"] = float(np.nanmin(ranks))
        summary["center_rank_max"] = float(np.nanmax(ranks))
        summary["count_sum"] = int(df["count"].to_numpy(dtype=int).sum())
        if bool(np.isfinite(mean_ic).any()):
            i_max = int(np.nanargmax(mean_ic))
            i_min = int(np.nanargmin(mean_ic))
            summary["pearson_ic_max"] = {"value": float(mean_ic[i_max]), "group_center_rank": float(ranks[i_max])}
            summary["pearson_ic_min"] = {"value": float(mean_ic[i_min]), "group_center_rank": float(ranks[i_min])}
        if bool(np.isfinite(mean_rank_ic).any()):
            i_max_rank = int(np.nanargmax(mean_rank_ic))
            i_min_rank = int(np.nanargmin(mean_rank_ic))
            summary["rank_ic_max"] = {"value": float(mean_rank_ic[i_max_rank]), "group_center_rank": float(ranks[i_max_rank])}
            summary["rank_ic_min"] = {"value": float(mean_rank_ic[i_min_rank]), "group_center_rank": float(ranks[i_min_rank])}

    # Persist the compact summary as YAML for local inspection and report linkage.
    import yaml

    out_yaml.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation for two 1D arrays."""
    # Keep only finite rows and require at least two samples.
    m = np.isfinite(x) & np.isfinite(y)
    x2 = x[m].astype(float, copy=False)
    y2 = y[m].astype(float, copy=False)
    if int(x2.shape[0]) < 2:
        return float("nan")
    return float(np.corrcoef(x2, y2)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation via rank Pearson."""
    # Keep only finite rows and require at least two samples.
    m = np.isfinite(x) & np.isfinite(y)
    x2 = x[m].astype(float, copy=False)
    y2 = y[m].astype(float, copy=False)
    if int(x2.shape[0]) < 2:
        return float("nan")
    xr = pd.Series(x2).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y2).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(xr, yr)[0, 1])


def load_eval_predictions(shard_path: Path) -> pd.DataFrame:
    """Load a rank0 feather shard written by qmodel evaluator."""
    # Read the feather file and validate required columns.
    df = pd.read_feather(Path(shard_path))
    need = {"prediction", "target", "StockCode", "DateTime", "date", "time"}
    if not need.issubset(set(df.columns)):
        raise RuntimeError(f"Missing columns in shard: {sorted(need - set(df.columns))}")
    return df


class PredictionChunkReader:
    """Iterate prediction parquet chunks written by the streaming predict evaluator."""

    def __init__(self, manifest_path: Path) -> None:
        """Load and validate a predict manifest YAML for chunk iteration."""
        # Parse YAML and keep a normalized manifest dict for downstream readers.
        import yaml

        self._manifest_path = Path(manifest_path)
        self._base_dir = Path(self._manifest_path).parent
        manifest = yaml.safe_load(self._manifest_path.read_text(encoding="utf-8"))
        self._manifest = dict(manifest)

    @property
    def manifest(self) -> dict[str, object]:
        """Return the raw manifest mapping."""
        # Return a shallow copy so callers do not mutate internal state.
        return dict(self._manifest)

    def iter_chunks(self, columns: list[str]) -> list[pd.DataFrame]:
        """Load every parquet chunk as a dataframe list with a fixed column projection."""
        # Resolve chunk paths relative to the manifest directory.
        parts: list[pd.DataFrame] = []
        for rel in list(self._manifest["chunk_files"]):
            # Read one chunk with column pruning to reduce IO.
            path = Path(self._base_dir) / str(rel)
            parts.append(pd.read_parquet(path, columns=list(columns)))
        return parts

    def iter_prediction_chunks(self, columns: list[str]):
        """Yield parquet chunks one-by-one as projected dataframes."""
        # Stream chunks in the on-disk order recorded by the manifest.
        for rel in list(self._manifest["chunk_files"]):
            path = Path(self._base_dir) / str(rel)
            yield pd.read_parquet(path, columns=list(columns))

    def iter_timestamp_groups(self, columns: list[str]):
        """Yield full (date,time) groups, handling chunk boundary carry-over."""
        # Keep a carry dataframe for the last (date,time) in the previous chunk.
        carry: pd.DataFrame | None = None
        for chunk in self.iter_prediction_chunks(list(columns)):
            # Prepend the carry group to the next chunk so it becomes whole.
            df = chunk if carry is None else pd.concat([carry, chunk], axis=0, ignore_index=True)

            # Save the last group as the new carry to handle chunk boundaries.
            last_date = int(df.iloc[-1]["date"])
            last_time = int(df.iloc[-1]["time"])
            last_mask = (df["date"].astype(int) == int(last_date)) & (df["time"].astype(int) == int(last_time))
            carry = df.loc[last_mask].copy()
            ready = df.loc[~last_mask]

            # Yield all full groups in stable appearance order.
            for (d, t), g in ready.groupby(["date", "time"], sort=False):
                yield int(d), int(t), g

        # Yield the final carry group after all chunks are consumed.
        if carry is not None and int(carry.shape[0]) > 0:
            d = int(carry.iloc[0]["date"])
            t = int(carry.iloc[0]["time"])
            yield int(d), int(t), carry

    def iter_date_groups(self, columns: list[str]):
        """Yield full date groups, handling chunk boundary carry-over."""
        # Keep a carry dataframe for the last date in the previous chunk.
        carry: pd.DataFrame | None = None
        for chunk in self.iter_prediction_chunks(list(columns)):
            # Prepend the carry day to the next chunk so it becomes whole.
            df = chunk if carry is None else pd.concat([carry, chunk], axis=0, ignore_index=True)

            # Save the last date as the new carry to handle chunk boundaries.
            last_date = int(df.iloc[-1]["date"])
            last_mask = df["date"].astype(int) == int(last_date)
            carry = df.loc[last_mask].copy()
            ready = df.loc[~last_mask]

            # Yield all full date groups in stable appearance order.
            for d, g in ready.groupby("date", sort=False):
                yield int(d), g

        # Yield the final carry group after all chunks are consumed.
        if carry is not None and int(carry.shape[0]) > 0:
            d = int(carry.iloc[0]["date"])
            yield int(d), carry


class OnlinePearsonAccumulator:
    """Accumulate Pearson correlation moments in one pass."""

    def __init__(self) -> None:
        """Initialize empty raw-moment sums."""
        # Track sums for exact Pearson computation over finite pairs.
        self._n = 0
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_x2 = 0.0
        self._sum_y2 = 0.0
        self._sum_xy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Update the accumulator with a vector chunk of samples."""
        # Filter finite pairs and accumulate float64 sums.
        m = np.isfinite(x) & np.isfinite(y)
        x2 = x[m].astype(np.float64, copy=False)
        y2 = y[m].astype(np.float64, copy=False)
        n = int(x2.shape[0])
        if int(n) == 0:
            return
        self._n += int(n)
        self._sum_x += float(x2.sum(dtype=np.float64))
        self._sum_y += float(y2.sum(dtype=np.float64))
        self._sum_x2 += float((x2 * x2).sum(dtype=np.float64))
        self._sum_y2 += float((y2 * y2).sum(dtype=np.float64))
        self._sum_xy += float((x2 * y2).sum(dtype=np.float64))

    def finalize(self) -> float:
        """Return the accumulated Pearson correlation as a scalar."""
        # Compute covariance and variances from raw sums.
        n = int(self._n)
        if int(n) < 2:
            return float("nan")
        nf = float(n)
        cov = float(self._sum_xy - (self._sum_x * self._sum_y) / nf)
        var_x = float(self._sum_x2 - (self._sum_x * self._sum_x) / nf)
        var_y = float(self._sum_y2 - (self._sum_y * self._sum_y) / nf)
        if not (np.isfinite(cov) and np.isfinite(var_x) and np.isfinite(var_y)):
            return float("nan")
        if float(var_x) <= 0.0 or float(var_y) <= 0.0:
            return float("nan")
        return float(cov / float(np.sqrt(var_x * var_y)))

    def count(self) -> int:
        """Return the number of finite pairs accumulated."""
        # Expose count to match existing pooled_ic output schema.
        return int(self._n)


class OnlineMoments:
    """Accumulate mean/std and sign ratio for a scalar stream."""

    def __init__(self) -> None:
        """Initialize Welford running-moment state."""
        # Store count/mean/M2 and sign count for stable streaming stats.
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._pos = 0

    def update(self, x: float) -> None:
        """Update the running moments with one scalar observation."""
        # Skip non-finite values to match pandas dropna behavior.
        if not np.isfinite(float(x)):
            return
        if float(x) > 0.0:
            self._pos += 1
        self._n += 1
        delta = float(x) - float(self._mean)
        self._mean += delta / float(self._n)
        delta2 = float(x) - float(self._mean)
        self._m2 += delta * delta2

    def finalize(self) -> dict[str, float]:
        """Return a summary dict with count/mean/std/t_stat/positive_ratio."""
        # Convert Welford state into the same schema as ic_time_series_summary.
        n = int(self._n)
        mean = float(self._mean) if int(n) > 0 else float("nan")
        std = float(np.sqrt(self._m2 / float(n - 1))) if int(n) > 1 else float("nan")
        t_stat = float(mean / (std / np.sqrt(float(n)))) if int(n) > 1 and float(std) > 0.0 else float("nan")
        pos_ratio = float(self._pos) / float(n) if int(n) > 0 else float("nan")
        return {"count": int(n), "mean": float(mean), "std": float(std), "t_stat": float(t_stat), "positive_ratio": float(pos_ratio)}


class InMemoryPooledICAccumulator:
    """Accumulate pooled IC inputs and finalize Pearson/Rank IC."""

    def __init__(self) -> None:
        """Initialize Pearson moments and exact-rank buffers."""
        # Keep Pearson online while retaining finite pairs for exact pooled Spearman.
        self._pearson = OnlinePearsonAccumulator()
        self._pred_parts: list[np.ndarray] = []
        self._target_parts: list[np.ndarray] = []

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Add one vector chunk to the pooled IC accumulator."""
        # Filter finite pairs once so Pearson and rank IC use the same sample.
        m = np.isfinite(x) & np.isfinite(y)
        x2 = x[m].astype(np.float64, copy=False)
        y2 = y[m].astype(np.float64, copy=False)
        if int(x2.shape[0]) == 0:
            return

        # Update streaming Pearson moments and store compact arrays for final rank IC.
        self._pearson.update(x2, y2)
        self._pred_parts.append(x2.astype(np.float32, copy=False))
        self._target_parts.append(y2.astype(np.float32, copy=False))

    def finalize(self) -> dict[str, float]:
        """Return Pearson/Rank IC and finite sample count."""
        # Concatenate retained finite pairs before exact pooled ranking.
        count = int(self._pearson.count())
        if int(count) < 2:
            return {"pearson_ic": float("nan"), "rank_ic": float("nan"), "count": int(count)}

        # Compute pooled rank IC with scipy ranks to avoid pandas object overhead.
        pred = np.concatenate(self._pred_parts).astype(np.float64, copy=False)
        target = np.concatenate(self._target_parts).astype(np.float64, copy=False)
        pred_rank = stats.rankdata(pred, method="average").astype(np.float64, copy=False)
        target_rank = stats.rankdata(target, method="average").astype(np.float64, copy=False)
        rank_ic = _pearson(pred_rank, target_rank)
        return {"pearson_ic": float(self._pearson.finalize()), "rank_ic": float(rank_ic), "count": int(count)}


def _daily_pearson_ic_for_one_date(day_df: pd.DataFrame) -> float:
    """Compute one day's daily IC as the mean of intraday cross-sectional ICs."""
    # Compute one cross-sectional Pearson IC for each timestamp in the day.
    day_ics: list[float] = []
    for (_date, _time), group in day_df.groupby(["date", "time"], sort=False):
        pred = group["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = group["target"].to_numpy(dtype=np.float64, copy=False)
        day_ics.append(float(_pearson(pred, tgt)))

    # Average finite intraday ICs to get one daily IC observation.
    vals = np.asarray(day_ics, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if int(vals.shape[0]) == 0:
        return float("nan")
    return float(vals.mean(dtype=np.float64))


def _daily_rank_ic_for_one_date(day_df: pd.DataFrame) -> float:
    """Compute one day's daily Rank IC as the mean of intraday cross-sectional Rank ICs."""
    # Compute one cross-sectional Rank IC for each timestamp in the day.
    day_ics: list[float] = []
    for (_date, _time), group in day_df.groupby(["date", "time"], sort=False):
        pred = group["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = group["target"].to_numpy(dtype=np.float64, copy=False)
        day_ics.append(float(_spearman(pred, tgt)))

    # Average finite intraday Rank ICs to get one daily observation.
    vals = np.asarray(day_ics, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if int(vals.shape[0]) == 0:
        return float("nan")
    return float(vals.mean(dtype=np.float64))


def _rolling_rank_ic_summary(daily: pd.DataFrame, window_days: int) -> dict[str, object]:
    """Summarize rolling-window daily Rank IC mean and ICIR."""
    # Sort daily observations by date before assigning fixed rolling windows.
    daily2 = daily.dropna(subset=["rank_ic"]).sort_values("date", kind="stable").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for start in range(0, int(daily2.shape[0]), int(window_days)):
        # Slice one non-overlapping window and compute mean/std/ICIR.
        window = daily2.iloc[start : start + int(window_days)]
        vals = window["rank_ic"].to_numpy(dtype=np.float64, copy=False)
        vals = vals[np.isfinite(vals)]
        mean = float(vals.mean(dtype=np.float64)) if int(vals.shape[0]) > 0 else float("nan")
        std = float(vals.std(ddof=1)) if int(vals.shape[0]) > 1 else float("nan")
        icir = float(mean / std) if np.isfinite(mean) and np.isfinite(std) and float(std) > 0.0 else float("nan")
        if int(window.shape[0]) == 0:
            continue
        rows.append(
            {
                "window_id": int(len(rows)),
                "start_date": int(window.iloc[0]["date"]),
                "end_date": int(window.iloc[-1]["date"]),
                "day_count": int(vals.shape[0]),
                "rank_ic_mean": float(mean),
                "rank_ic_std": float(std),
                "rank_icir": float(icir),
            }
        )

    # Aggregate window-level statistics for compact report display.
    out = pd.DataFrame(rows)
    mean_vals = out["rank_ic_mean"].to_numpy(dtype=np.float64, copy=False) if int(out.shape[0]) > 0 else np.asarray([], dtype=np.float64)
    icir_vals = out["rank_icir"].to_numpy(dtype=np.float64, copy=False) if int(out.shape[0]) > 0 else np.asarray([], dtype=np.float64)
    return {
        "window_days": int(window_days),
        "window_count": int(out.shape[0]),
        "rank_ic_mean": float(np.nanmean(mean_vals)) if bool(np.isfinite(mean_vals).any()) else float("nan"),
        "rank_icir_mean": float(np.nanmean(icir_vals)) if bool(np.isfinite(icir_vals).any()) else float("nan"),
        "windows": rows,
    }


def _daily_rank_ic_from_manifest_duckdb(manifest_path: Path) -> pd.DataFrame:
    """Compute daily Rank IC observations from a manifest with DuckDB."""
    # Rank prediction/target inside each timestamp cross-section and average timestamp ICs by date.
    import duckdb

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=16")
    scan = _duckdb_manifest_scan(Path(manifest_path))
    daily = con.execute(
        f"""
        WITH base AS (
            SELECT
                date::INTEGER AS date,
                time::INTEGER AS time,
                prediction::DOUBLE AS prediction,
                target::DOUBLE AS target
            FROM {scan}
            WHERE isfinite(prediction) AND isfinite(target)
        ),
        ranked AS (
            SELECT
                date,
                time,
                rank() OVER (PARTITION BY date, time ORDER BY prediction)
                    + (count(*) OVER (PARTITION BY date, time, prediction) - 1) / 2.0 AS prediction_rank,
                rank() OVER (PARTITION BY date, time ORDER BY target)
                    + (count(*) OVER (PARTITION BY date, time, target) - 1) / 2.0 AS target_rank
            FROM base
        ),
        timestamp_ic AS (
            SELECT date, time, corr(prediction_rank, target_rank) AS rank_ic
            FROM ranked
            GROUP BY date, time
        )
        SELECT date, avg(rank_ic) AS rank_ic
        FROM timestamp_ic
        GROUP BY date
        ORDER BY date
        """
    ).fetchdf()
    con.close()
    return daily


def core_ic_summary_from_manifest(manifest_path: Path, out_yaml: Path, rolling_window_days: int) -> dict[str, object]:
    """Write core pooled IC and rolling daily Rank IC metrics from a manifest."""
    # Compute exact pooled Pearson/Rank IC directly from parquet chunks.
    import yaml

    pooled = pooled_ic_from_manifest(Path(manifest_path))

    # Compute daily Rank IC observations with DuckDB so rolling summaries do not stream in Python.
    daily = _daily_rank_ic_from_manifest_duckdb(Path(manifest_path))
    rolling = _rolling_rank_ic_summary(daily, int(rolling_window_days))
    summary = {"pooled": dict(pooled), "rolling_rank_ic": dict(rolling), "day_count": int(daily.shape[0])}
    Path(out_yaml).write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return summary


def _manifest_parquet_paths(manifest_path: Path) -> list[Path]:
    """Resolve every parquet chunk path recorded in a predict manifest."""
    # Load the manifest and expand relative chunk paths against the manifest directory.
    import yaml

    manifest_path = Path(manifest_path)
    manifest = dict(yaml.safe_load(manifest_path.read_text(encoding="utf-8")))
    base_dir = Path(manifest_path).parent
    return [base_dir / str(rel) for rel in list(manifest["chunk_files"])]


def _duckdb_path_list(paths: list[Path]) -> str:
    """Render a DuckDB SQL list literal for parquet file paths."""
    # Quote paths explicitly so DuckDB scans only the manifest's recorded chunks.
    quoted = []
    for path in list(paths):
        s = Path(path).as_posix().replace("'", "''")
        quoted.append(f"'{s}'")
    return "[" + ", ".join(quoted) + "]"


def _duckdb_manifest_scan(manifest_path: Path) -> str:
    """Build a DuckDB read_parquet expression for one predict manifest."""
    # Keep this as a small SQL fragment shared by pooled and grouped IC queries.
    return f"read_parquet({_duckdb_path_list(_manifest_parquet_paths(Path(manifest_path)))})"


def pooled_ic_from_manifest(manifest_path: Path) -> dict[str, float]:
    """Compute pooled Pearson/Rank IC from a predict manifest with DuckDB parquet scans."""
    # Use DuckDB window ranks to avoid multi-GB TSV materialization and repeated external sorts.
    import duckdb

    scan = _duckdb_manifest_scan(Path(manifest_path))
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=16")
    row = con.execute(
        f"""
        WITH base AS (
            SELECT prediction::DOUBLE AS prediction, target::DOUBLE AS target
            FROM {scan}
            WHERE isfinite(prediction) AND isfinite(target)
        ),
        ranked AS (
            SELECT
                prediction,
                target,
                rank() OVER (ORDER BY prediction) + (count(*) OVER (PARTITION BY prediction) - 1) / 2.0 AS prediction_rank,
                rank() OVER (ORDER BY target) + (count(*) OVER (PARTITION BY target) - 1) / 2.0 AS target_rank
            FROM base
        )
        SELECT
            corr(prediction, target) AS pearson_ic,
            corr(prediction_rank, target_rank) AS rank_ic,
            count(*) AS count
        FROM ranked
        """
    ).fetchone()
    con.close()
    return {"pearson_ic": float(row[0]), "rank_ic": float(row[1]), "count": int(row[2])}


def pooled_nonzero_prediction_ic_from_manifest(manifest_path: Path) -> dict[str, float]:
    """Compute pooled Pearson/Rank IC on rows where prediction is nonzero."""
    # Use DuckDB window ranks over the filtered finite nonzero prediction sample.
    import duckdb

    scan = _duckdb_manifest_scan(Path(manifest_path))
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=16")
    row = con.execute(
        f"""
        WITH base AS (
            SELECT prediction::DOUBLE AS prediction, target::DOUBLE AS target
            FROM {scan}
            WHERE isfinite(prediction) AND isfinite(target) AND prediction != 0.0
        ),
        ranked AS (
            SELECT
                prediction,
                target,
                rank() OVER (ORDER BY prediction) + (count(*) OVER (PARTITION BY prediction) - 1) / 2.0 AS prediction_rank,
                rank() OVER (ORDER BY target) + (count(*) OVER (PARTITION BY target) - 1) / 2.0 AS target_rank
            FROM base
        )
        SELECT
            corr(prediction, target) AS pearson_ic,
            corr(prediction_rank, target_rank) AS rank_ic,
            count(*) AS count
        FROM ranked
        """
    ).fetchone()
    con.close()
    return {
        "pearson_ic": float(row[0]),
        "rank_ic": float(row[1]),
        "count": int(row[2]),
    }


def pooled_top_decile_return_from_manifest(manifest_path: Path) -> dict[str, float]:
    """Compute pooled top prediction decile return and related target diagnostics."""
    # Rank globally by prediction, then summarize the pooled top and bottom deciles.
    import duckdb

    scan = _duckdb_manifest_scan(Path(manifest_path))
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=16")
    row = con.execute(
        f"""
        WITH base AS (
            SELECT prediction::DOUBLE AS prediction, target::DOUBLE AS target
            FROM {scan}
            WHERE isfinite(prediction) AND isfinite(target)
        ),
        bucketed AS (
            SELECT
                prediction,
                target,
                ntile(10) OVER (ORDER BY prediction) AS prediction_decile
            FROM base
        )
        SELECT
            avg(CASE WHEN prediction_decile = 10 THEN target ELSE NULL END) * 1e4 AS top_decile_return_bps,
            avg(CASE WHEN prediction_decile = 1 THEN target ELSE NULL END) * 1e4 AS bottom10_mean_target_bps,
            avg(CASE WHEN prediction_decile = 10 AND target > 0.0 THEN 1.0 WHEN prediction_decile = 10 THEN 0.0 ELSE NULL END) AS top10_hit_rate,
            avg(CASE WHEN prediction_decile = 10 THEN prediction ELSE NULL END) AS top10_pred_mean,
            stddev_pop(CASE WHEN prediction_decile = 10 THEN prediction ELSE NULL END) AS top10_pred_std,
            count(CASE WHEN prediction_decile = 10 THEN 1 ELSE NULL END) AS top10_count,
            count(*) AS count
        FROM bucketed
        """
    ).fetchone()
    con.close()
    return {
        "top_decile_return_bps": float(row[0]),
        "bottom10_mean_target_bps": float(row[1]),
        "top10_hit_rate": float(row[2]),
        "top10_pred_mean": float(row[3]),
        "top10_pred_std": float(row[4]),
        "top10_count": int(row[5]),
        "count": int(row[6]),
    }


def pooled_pearson_ic_from_manifest(manifest_path: Path) -> dict[str, float]:
    """Compute pooled Pearson IC from a manifest without exact rank IC work."""
    # Stream prediction chunks once and accumulate only Pearson moments.
    reader = PredictionChunkReader(Path(manifest_path))
    acc = OnlinePearsonAccumulator()
    for chunk in reader.iter_prediction_chunks(["prediction", "target"]):
        pred = chunk["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = chunk["target"].to_numpy(dtype=np.float64, copy=False)
        acc.update(pred, tgt)

    # Return the minimal schema needed by the train report.
    return {"pearson_ic": float(acc.finalize()), "count": int(acc.count())}


def _write_average_ranks_from_value_sorted(value_sorted_path: Path, out_rank_path: Path) -> None:
    """Assign pandas-style average ranks to a value-sorted TSV stream and write (row_id,rank)."""
    # Walk the sorted stream, accumulate tie groups, and emit one rank per row_id.
    pos = 1
    tie_value: float | None = None
    tie_row_ids: list[int] = []

    def _flush() -> None:
        # Write the current tie group with an average rank and advance the position cursor.
        nonlocal pos, tie_value, tie_row_ids
        if len(tie_row_ids) == 0:
            return
        start = int(pos)
        end = int(pos) + int(len(tie_row_ids)) - 1
        avg_rank = 0.5 * float(start + end)
        for rid in tie_row_ids:
            out.write(f"{int(rid)}\t{float(avg_rank):.12g}\n")
        pos = int(pos) + int(len(tie_row_ids))
        tie_value = None
        tie_row_ids = []

    with Path(value_sorted_path).open("r", encoding="utf-8") as fp, Path(out_rank_path).open("w", encoding="utf-8") as out:
        for line in fp:
            v_str, rid_str = line.rstrip("\n").split("\t")
            v = float(v_str)
            rid = int(rid_str)
            if tie_value is None:
                tie_value = float(v)
                tie_row_ids.append(int(rid))
                continue
            if float(v) == float(tie_value):
                tie_row_ids.append(int(rid))
                continue
            _flush()
            tie_value = float(v)
            tie_row_ids.append(int(rid))
        _flush()


def _pearson_from_row_id_rank_files(pred_rank_sorted: Path, tgt_rank_sorted: Path) -> float:
    """Compute Pearson correlation from two row_id-sorted rank TSVs."""
    # Stream-merge the two rank files by row_id and update an online Pearson accumulator.
    acc = OnlinePearsonAccumulator()
    buf_pred: list[float] = []
    buf_tgt: list[float] = []
    buf_cap = 200000

    def _flush() -> None:
        # Flush buffered rank pairs into the vectorized accumulator update.
        nonlocal buf_pred, buf_tgt
        if len(buf_pred) == 0:
            return
        x = np.asarray(buf_pred, dtype=np.float64)
        y = np.asarray(buf_tgt, dtype=np.float64)
        acc.update(x, y)
        buf_pred = []
        buf_tgt = []

    with Path(pred_rank_sorted).open("r", encoding="utf-8") as fp_pred, Path(tgt_rank_sorted).open("r", encoding="utf-8") as fp_tgt:
        while True:
            lp = fp_pred.readline()
            lt = fp_tgt.readline()
            if lp == "" and lt == "":
                break
            rid_p_str, r_p_str = lp.rstrip("\n").split("\t")
            rid_t_str, r_t_str = lt.rstrip("\n").split("\t")
            if int(rid_p_str) != int(rid_t_str):
                raise RuntimeError(f"row_id mismatch in rank merge: pred={rid_p_str}, tgt={rid_t_str}")
            buf_pred.append(float(r_p_str))
            buf_tgt.append(float(r_t_str))
            if int(len(buf_pred)) >= int(buf_cap):
                _flush()
        _flush()
    return float(acc.finalize())


def annual_pooled_ic_from_manifest(manifest_path: Path, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute annual pooled Pearson IC from a predict manifest and persist CSV/plot."""
    # Stream once to accumulate per-year Pearson and spill rows for exact per-year Spearman via external sort.
    import subprocess
    import tempfile

    reader = PredictionChunkReader(Path(manifest_path))
    by_year: dict[int, OnlinePearsonAccumulator] = {}
    with tempfile.TemporaryDirectory(prefix="annual_rank_ic_") as td:
        tmp_dir = Path(td)
        pred_value = Path(tmp_dir) / "pred_value.tsv"
        tgt_value = Path(tmp_dir) / "tgt_value.tsv"
        pred_sorted = Path(tmp_dir) / "pred_value_sorted.tsv"
        tgt_sorted = Path(tmp_dir) / "tgt_value_sorted.tsv"
        pred_rank = Path(tmp_dir) / "pred_rank.tsv"
        tgt_rank = Path(tmp_dir) / "tgt_rank.tsv"
        pred_rank_sorted = Path(tmp_dir) / "pred_rank_sorted.tsv"
        tgt_rank_sorted = Path(tmp_dir) / "tgt_rank_sorted.tsv"

        # Write raw (year,value,row_id) TSVs for prediction and target while updating per-year Pearson moments.
        with pred_value.open("w", encoding="utf-8") as fp_pred, tgt_value.open("w", encoding="utf-8") as fp_tgt:
            for chunk in reader.iter_prediction_chunks(["row_id", "date", "prediction", "target"]):
                row_id = chunk["row_id"].to_numpy(dtype=np.int64, copy=False)
                dates = chunk["date"].to_numpy(dtype=np.int64, copy=False)
                years = (2000 + (dates // 10000)).astype(np.int64, copy=False)
                pred = chunk["prediction"].to_numpy(dtype=np.float32, copy=False)
                tgt = chunk["target"].to_numpy(dtype=np.float32, copy=False)
                for y in np.unique(years):
                    if int(y) not in by_year:
                        by_year[int(y)] = OnlinePearsonAccumulator()
                    m_year = years == int(y)
                    by_year[int(y)].update(pred[m_year].astype(np.float64, copy=False), tgt[m_year].astype(np.float64, copy=False))
                m = np.isfinite(pred) & np.isfinite(tgt)
                if int(m.sum()) == 0:
                    continue
                pred_tbl = pd.DataFrame({"year": years[m], "value": pred[m], "row_id": row_id[m]})
                tgt_tbl = pd.DataFrame({"year": years[m], "value": tgt[m], "row_id": row_id[m]})
                pred_tbl.to_csv(fp_pred, sep="\t", header=False, index=False, float_format="%.9g")
                tgt_tbl.to_csv(fp_tgt, sep="\t", header=False, index=False, float_format="%.9g")

        # External-sort by (year,value,row_id) so ranks reset per year and ties follow pandas ordering.
        subprocess.run(["sort", "-t", "\t", "-k1,1n", "-k2,2g", "-k3,3n", pred_value.as_posix(), "-o", pred_sorted.as_posix()], check=True)
        subprocess.run(["sort", "-t", "\t", "-k1,1n", "-k2,2g", "-k3,3n", tgt_value.as_posix(), "-o", tgt_sorted.as_posix()], check=True)

        # Convert value-sorted tables into (row_id,year,rank) with per-year average-tie ranks.
        _write_average_ranks_from_year_value_sorted(Path(pred_sorted), Path(pred_rank))
        _write_average_ranks_from_year_value_sorted(Path(tgt_sorted), Path(tgt_rank))

        # External-sort the rank tables by row_id for a streaming merge join.
        subprocess.run(["sort", "-t", "\t", "-k1,1n", pred_rank.as_posix(), "-o", pred_rank_sorted.as_posix()], check=True)
        subprocess.run(["sort", "-t", "\t", "-k1,1n", tgt_rank.as_posix(), "-o", tgt_rank_sorted.as_posix()], check=True)

        # Merge pred/tgt ranks by row_id and compute per-year Pearson on ranks in one pass.
        year_rank_ic = _annual_pearson_from_row_id_year_rank_files(Path(pred_rank_sorted), Path(tgt_rank_sorted))

    # Convert per-year accumulators into a table consistent with the original schema.
    rows: list[dict[str, object]] = []
    for y in sorted(by_year.keys()):
        acc = by_year[int(y)]
        rows.append({"year": int(y), "pearson_ic": float(acc.finalize()), "rank_ic": float(year_rank_ic.get(int(y), float("nan"))), "count": int(acc.count())})
    out = pd.DataFrame(rows).sort_values("year", kind="stable").reset_index(drop=True)
    out.to_csv(Path(out_csv), index=False)

    # Plot yearly IC bars for Pearson and placeholder Rank IC.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    xs = out["year"].to_numpy(dtype=int)
    ax.bar(xs - 0.15, out["pearson_ic"].to_numpy(dtype=float), width=0.3, label="Pearson IC")
    ax.bar(xs + 0.15, out["rank_ic"].to_numpy(dtype=float), width=0.3, label="Rank IC (Spearman)")
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title("Annual pooled IC (prediction vs target)")
    ax.set_xlabel("year")
    ax.set_ylabel("IC")
    ax.set_xticks(xs, [str(int(x)) for x in xs])
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_png), dpi=160)
    plt.close(fig)
    return out


def annual_pooled_pearson_ic_from_manifest(manifest_path: Path, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute annual pooled Pearson IC from a manifest without exact rank-IC sorting."""
    # Stream once and update one Pearson accumulator per calendar year.
    reader = PredictionChunkReader(Path(manifest_path))
    by_year: dict[int, OnlinePearsonAccumulator] = {}
    for chunk in reader.iter_prediction_chunks(["date", "prediction", "target"]):
        # Update yearly Pearson accumulators for this chunk.
        dates = chunk["date"].to_numpy(dtype=np.int64, copy=False)
        years = dates // 10000
        pred = chunk["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = chunk["target"].to_numpy(dtype=np.float64, copy=False)
        for year in np.unique(years):
            if int(year) not in by_year:
                by_year[int(year)] = OnlinePearsonAccumulator()
            mask = years == int(year)
            by_year[int(year)].update(pred[mask], tgt[mask])

    # Materialize a compact annual Pearson table and persist it as CSV.
    rows: list[dict[str, object]] = []
    for year in sorted(by_year.keys()):
        acc = by_year[int(year)]
        rows.append({"year": int(year), "pearson_ic": float(acc.finalize()), "count": int(acc.count())})
    out = pd.DataFrame(rows).sort_values("year", kind="stable").reset_index(drop=True)
    out.to_csv(Path(out_csv), index=False)

    # Skip a one-point figure because the table is the clearer representation.
    if int(out.shape[0]) <= 1:
        Path(out_png).unlink(missing_ok=True)
        return out

    # Plot the yearly Pearson IC curve with a zero reference line.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(out["year"].to_numpy(dtype=int), out["pearson_ic"].to_numpy(dtype=float), marker="o", linewidth=1.8, label="Pearson IC")
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title("Annual pooled IC")
    ax.set_xlabel("year")
    ax.set_ylabel("pooled IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_png), dpi=160)
    plt.close(fig)
    return out


def _write_average_ranks_from_year_value_sorted(value_sorted_path: Path, out_rank_path: Path) -> None:
    """Assign per-year average ranks to a (year,value,row_id)-sorted TSV and write (row_id,year,rank)."""
    # Walk the sorted stream, reset at year boundaries, accumulate per-year tie groups, and emit ranks.
    year: int | None = None
    pos = 1
    tie_value: float | None = None
    tie_row_ids: list[int] = []

    def _flush() -> None:
        # Write the current per-year tie group with an average rank and advance the position cursor.
        nonlocal pos, tie_value, tie_row_ids
        if len(tie_row_ids) == 0:
            return
        start = int(pos)
        end = int(pos) + int(len(tie_row_ids)) - 1
        avg_rank = 0.5 * float(start + end)
        for rid in tie_row_ids:
            out.write(f"{int(rid)}\t{int(year)}\t{float(avg_rank):.12g}\n")
        pos = int(pos) + int(len(tie_row_ids))
        tie_value = None
        tie_row_ids = []

    with Path(value_sorted_path).open("r", encoding="utf-8") as fp, Path(out_rank_path).open("w", encoding="utf-8") as out:
        for line in fp:
            y_str, v_str, rid_str = line.rstrip("\n").split("\t")
            y = int(y_str)
            v = float(v_str)
            rid = int(rid_str)
            if year is None:
                year = int(y)
            if int(y) != int(year):
                _flush()
                year = int(y)
                pos = 1
            if tie_value is None:
                tie_value = float(v)
                tie_row_ids.append(int(rid))
                continue
            if float(v) == float(tie_value):
                tie_row_ids.append(int(rid))
                continue
            _flush()
            tie_value = float(v)
            tie_row_ids.append(int(rid))
        _flush()


def _annual_pearson_from_row_id_year_rank_files(pred_rank_sorted: Path, tgt_rank_sorted: Path) -> dict[int, float]:
    """Compute per-year Pearson correlation from two row_id-sorted annual rank TSVs."""
    # Stream-merge the two rank files by row_id and update a per-year Pearson accumulator map.
    by_year: dict[int, OnlinePearsonAccumulator] = {}
    buf_pred: dict[int, list[float]] = {}
    buf_tgt: dict[int, list[float]] = {}
    buf_cap = 200000

    def _flush_year(y: int) -> None:
        # Flush one year's buffered rank pairs into the vectorized accumulator update.
        xs = buf_pred.get(int(y), [])
        ys = buf_tgt.get(int(y), [])
        if len(xs) == 0:
            return
        if int(y) not in by_year:
            by_year[int(y)] = OnlinePearsonAccumulator()
        by_year[int(y)].update(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
        buf_pred[int(y)] = []
        buf_tgt[int(y)] = []

    with Path(pred_rank_sorted).open("r", encoding="utf-8") as fp_pred, Path(tgt_rank_sorted).open("r", encoding="utf-8") as fp_tgt:
        while True:
            lp = fp_pred.readline()
            lt = fp_tgt.readline()
            if lp == "" and lt == "":
                break
            rid_p_str, y_p_str, r_p_str = lp.rstrip("\n").split("\t")
            rid_t_str, y_t_str, r_t_str = lt.rstrip("\n").split("\t")
            if int(rid_p_str) != int(rid_t_str):
                raise RuntimeError(f"row_id mismatch in annual rank merge: pred={rid_p_str}, tgt={rid_t_str}")
            if int(y_p_str) != int(y_t_str):
                raise RuntimeError(f"year mismatch in annual rank merge: pred={y_p_str}, tgt={y_t_str}")
            y = int(y_p_str)
            if int(y) not in buf_pred:
                buf_pred[int(y)] = []
                buf_tgt[int(y)] = []
            buf_pred[int(y)].append(float(r_p_str))
            buf_tgt[int(y)].append(float(r_t_str))
            if int(len(buf_pred[int(y)])) >= int(buf_cap):
                _flush_year(int(y))
        for y in list(buf_pred.keys()):
            _flush_year(int(y))

    out: dict[int, float] = {}
    for y in sorted(by_year.keys()):
        out[int(y)] = float(by_year[int(y)].finalize())
    return out


def ic_time_series_summary_from_manifest(manifest_path: Path, out_yaml: Path) -> dict[str, object]:
    """Write timestamp-level IC summary metrics from a predict manifest."""
    # Stream full (date,time) groups and update running summary stats.
    import yaml

    reader = PredictionChunkReader(Path(manifest_path))
    pearson_stats = OnlineMoments()
    rank_stats = OnlineMoments()
    timestamp_count = 0
    for _d, _t, g in reader.iter_timestamp_groups(["date", "time", "prediction", "target"]):
        # Compute cross-sectional ICs for this timestamp and update the summary accumulators.
        pred = g["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = g["target"].to_numpy(dtype=np.float64, copy=False)
        ic = _pearson(pred, tgt)
        ric = _spearman(pred, tgt)
        pearson_stats.update(float(ic))
        rank_stats.update(float(ric))
        timestamp_count += 1

    # Persist YAML summary with the same shape as ic_time_series_summary.
    summary = {"pearson_ic": pearson_stats.finalize(), "rank_ic": rank_stats.finalize(), "timestamp_count": int(timestamp_count)}
    Path(out_yaml).write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return summary


def pearson_ic_time_series_summary_from_manifest(manifest_path: Path, out_yaml: Path) -> dict[str, object]:
    """Write timestamp-level Pearson IC summary metrics from a manifest."""
    # Stream full (date,time) groups and update Pearson-only running summary stats.
    import yaml

    reader = PredictionChunkReader(Path(manifest_path))
    pearson_stats = OnlineMoments()
    timestamp_count = 0
    for _d, _t, g in reader.iter_timestamp_groups(["date", "time", "prediction", "target"]):
        # Compute one timestamp Pearson IC and update the running summary.
        pred = g["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = g["target"].to_numpy(dtype=np.float64, copy=False)
        pearson_stats.update(float(_pearson(pred, tgt)))
        timestamp_count += 1

    # Persist YAML with the subset schema used by the train report.
    summary = {"pearson_ic": pearson_stats.finalize(), "timestamp_count": int(timestamp_count)}
    Path(out_yaml).write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return summary


def daily_pearson_ic_summary_from_manifest(manifest_path: Path, out_yaml: Path) -> dict[str, object]:
    """Write daily-IC summary metrics from a manifest."""
    # Stream full date groups and update one daily IC observation per date.
    import yaml

    reader = PredictionChunkReader(Path(manifest_path))
    daily_stats = OnlineMoments()
    day_count = 0
    for _d, day_df in reader.iter_date_groups(["date", "time", "prediction", "target"]):
        # Compute one daily IC from the day's intraday cross-sectional IC sequence.
        daily_ic = float(_daily_pearson_ic_for_one_date(day_df))
        daily_stats.update(float(daily_ic))
        day_count += 1

    # Convert mean/std into ICIR while keeping the existing summary schema compact.
    pearson_summary = dict(daily_stats.finalize())
    std = float(pearson_summary["std"])
    mean = float(pearson_summary["mean"])
    icir = float(mean / std) if np.isfinite(mean) and np.isfinite(std) and float(std) > 0.0 else float("nan")
    pearson_summary["icir"] = float(icir)

    # Persist YAML with the day-level schema used by downstream reports.
    summary = {"pearson_ic": pearson_summary, "day_count": int(day_count)}
    Path(out_yaml).write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return summary


def intraday_time_series_ic_from_manifest(manifest_path: Path, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute intraday mean/std IC curve from a predict manifest."""
    # Stream per-timestamp IC and aggregate by minute-of-day using online moments.
    reader = PredictionChunkReader(Path(manifest_path))
    pearson_by_time: dict[int, OnlineMoments] = {}
    rank_by_time: dict[int, OnlineMoments] = {}
    for _d, t, g in reader.iter_timestamp_groups(["date", "time", "prediction", "target"]):
        # Compute this timestamp's ICs and update the per-time accumulators.
        pred = g["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = g["target"].to_numpy(dtype=np.float64, copy=False)
        ic = _pearson(pred, tgt)
        ric = _spearman(pred, tgt)
        if int(t) not in pearson_by_time:
            pearson_by_time[int(t)] = OnlineMoments()
            rank_by_time[int(t)] = OnlineMoments()
        pearson_by_time[int(t)].update(float(ic))
        rank_by_time[int(t)].update(float(ric))

    # Materialize the final curve table and persist it as CSV.
    rows: list[dict[str, object]] = []
    for t in sorted(pearson_by_time.keys()):
        # Convert moment states into mean/std for plotting.
        p = pearson_by_time[int(t)]
        r = rank_by_time[int(t)]
        p_sum = p.finalize()
        r_sum = r.finalize()
        rows.append({"time": int(t), "mean_ic": float(p_sum["mean"]), "std_ic": float(p_sum["std"]), "mean_rank_ic": float(r_sum["mean"]), "std_rank_ic": float(r_sum["std"]), "count": int(p_sum["count"])})
    agg = pd.DataFrame(rows).sort_values("time", kind="stable").reset_index(drop=True)
    agg.to_csv(Path(out_csv), index=False)

    # Plot the intraday mean IC curve on an HH:MM axis.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    xs = agg["time"].to_numpy(dtype=int)
    labels = [f"{int(tt)//10000:02d}:{(int(tt)%10000)//100:02d}" for tt in xs]
    ax.plot(np.arange(len(labels)), agg["mean_ic"].to_numpy(dtype=float), label="Pearson IC", linewidth=1.8)
    ax.plot(np.arange(len(labels)), agg["mean_rank_ic"].to_numpy(dtype=float), label="Rank IC (Spearman)", linewidth=1.8)
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title("Intraday IC curve (mean across dates)")
    ax.set_xlabel("time (minute bars; lunch break absent)")
    ax.set_ylabel("mean IC")
    tick_pos = np.linspace(0, max(len(labels) - 1, 1), 10).round().astype(int)
    ax.set_xticks(tick_pos, [labels[i] for i in tick_pos], rotation=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_png), dpi=160)
    plt.close(fig)
    return agg


def _intraday_pearson_curve_from_manifest(manifest_path: Path) -> pd.DataFrame:
    """Compute one intraday Pearson IC curve from a manifest."""
    # Stream timestamp groups and aggregate Pearson IC by minute-of-day.
    reader = PredictionChunkReader(Path(manifest_path))
    pearson_by_time: dict[int, OnlineMoments] = {}
    for _d, t, g in reader.iter_timestamp_groups(["date", "time", "prediction", "target"]):
        # Update the minute-of-day accumulator with this timestamp's Pearson IC.
        pred = g["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = g["target"].to_numpy(dtype=np.float64, copy=False)
        if int(t) not in pearson_by_time:
            pearson_by_time[int(t)] = OnlineMoments()
        pearson_by_time[int(t)].update(float(_pearson(pred, tgt)))

    # Materialize the final curve table with stable column names.
    rows: list[dict[str, object]] = []
    for t in sorted(pearson_by_time.keys()):
        stats = pearson_by_time[int(t)].finalize()
        rows.append({"time": int(t), "mean_ic": float(stats["mean"]), "std_ic": float(stats["std"]), "count": int(stats["count"])})
    return pd.DataFrame(rows).sort_values("time", kind="stable").reset_index(drop=True)


def intraday_time_series_ic_train_test_from_manifest(train_manifest_path: Path, test_manifest_path: Path, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute intraday minute-of-day Pearson IC curves for train and test from manifests."""
    # Build the train intraday Pearson curve.
    train_agg = _intraday_pearson_curve_from_manifest(Path(train_manifest_path)).rename(
        columns={"mean_ic": "mean_ic_train", "std_ic": "std_ic_train", "count": "count_train"}
    )

    # Build the test intraday Pearson curve.
    test_agg = _intraday_pearson_curve_from_manifest(Path(test_manifest_path)).rename(
        columns={"mean_ic": "mean_ic_test", "std_ic": "std_ic_test", "count": "count_test"}
    )

    # Merge the aligned minute-of-day curves so we can plot them together.
    agg = train_agg.merge(test_agg, on="time", how="inner")
    agg.to_csv(Path(out_csv), index=False)

    # Plot the train/test Pearson IC curves on one shared intraday axis.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    xs = agg["time"].to_numpy(dtype=int)
    labels = [f"{int(tt)//10000:02d}:{(int(tt)%10000)//100:02d}" for tt in xs]
    ax.plot(np.arange(len(labels)), agg["mean_ic_train"].to_numpy(dtype=float), label="Train Pearson IC", linewidth=1.8)
    ax.plot(np.arange(len(labels)), agg["mean_ic_test"].to_numpy(dtype=float), label="Test Pearson IC", linewidth=1.8)
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title("Intraday IC curve (Pearson; mean across dates)")
    ax.set_xlabel("time (minute bars; lunch break absent)")
    ax.set_ylabel("mean IC")
    tick_pos = np.linspace(0, max(len(labels) - 1, 1), 10).round().astype(int)
    ax.set_xticks(tick_pos, [labels[i] for i in tick_pos], rotation=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_png), dpi=160)
    plt.close(fig)
    return agg


def prediction_rank_turnover_from_manifest(manifest_path: Path, out_csv: Path, out_png: Path, out_yaml: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute adjacent-timestamp prediction-rank turnover from a predict manifest."""
    # Stream timestamp groups and compute adjacent rank corr/turnover within each day.
    import yaml

    reader = PredictionChunkReader(Path(manifest_path))
    rows: list[dict[str, object]] = []
    prev_day: int | None = None
    prev_time: int | None = None
    prev_tbl: pd.DataFrame | None = None
    for d, t, g in reader.iter_timestamp_groups(["date", "time", "code", "prediction"]):
        # Start a new day by clearing the previous timestamp state.
        if prev_day is None or int(d) != int(prev_day):
            prev_day = int(d)
            prev_time = None
            prev_tbl = None

        # Build the current timestamp prediction table.
        curr = g[["code", "prediction"]].dropna(subset=["prediction"]).copy()
        curr = curr.rename(columns={"code": "StockCode", "prediction": "prediction_curr"})
        curr["StockCode"] = curr["StockCode"].astype(int)

        # Compute adjacent turnover when a previous timestamp exists.
        if prev_tbl is not None and prev_time is not None:
            merged = prev_tbl.merge(curr, on="StockCode", how="inner")
            if int(merged.shape[0]) >= 2:
                rp = stats.rankdata(merged["prediction_prev"].to_numpy(dtype=float), method="average").astype(np.float64, copy=False)
                rc = stats.rankdata(merged["prediction_curr"].to_numpy(dtype=float), method="average").astype(np.float64, copy=False)
                rank_corr = _pearson(rp, rc)
                rank_turnover = float(1.0 - rank_corr) if np.isfinite(rank_corr) else float("nan")
                rows.append({"date": int(d), "prev_time": int(prev_time), "time": int(t), "rank_corr": float(rank_corr), "rank_turnover": float(rank_turnover), "count": int(merged.shape[0])})

        # Store the current timestamp as the new previous state.
        prev_tbl = curr.rename(columns={"prediction_curr": "prediction_prev"})
        prev_time = int(t)

    # Aggregate adjacent-turnover rows by minute-of-day for a stable intraday curve.
    raw = pd.DataFrame(rows).sort_values(["date", "time"], kind="stable").reset_index(drop=True)
    agg = raw.groupby("time", sort=True).agg(
        mean_rank_corr=("rank_corr", "mean"),
        mean_rank_turnover=("rank_turnover", "mean"),
        std_rank_turnover=("rank_turnover", "std"),
        count=("rank_turnover", "count"),
    )
    agg = agg.reset_index().sort_values("time", kind="stable").reset_index(drop=True)
    agg.to_csv(Path(out_csv), index=False)

    # Plot the mean adjacent-turnover curve across the trading day.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    xs = agg["time"].to_numpy(dtype=int)
    labels = [f"{int(tt)//10000:02d}:{(int(tt)%10000)//100:02d}" for tt in xs]
    ax.plot(np.arange(len(labels)), agg["mean_rank_turnover"].to_numpy(dtype=float), label="1 - corr(rank_t, rank_t-1)", linewidth=1.8)
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title("Prediction rank turnover (adjacent timestamps)")
    ax.set_xlabel("time")
    ax.set_ylabel("mean turnover")
    tick_pos = np.linspace(0, max(len(labels) - 1, 1), 10).round().astype(int)
    ax.set_xticks(tick_pos, [labels[i] for i in tick_pos], rotation=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_png), dpi=160)
    plt.close(fig)

    # Persist a compact YAML summary for report consumption.
    if int(agg.shape[0]) > 0:
        best_idx = int(agg["mean_rank_turnover"].idxmin())
        worst_idx = int(agg["mean_rank_turnover"].idxmax())
        summary = {
            "row_count": int(raw.shape[0]),
            "mean_rank_corr": float(raw["rank_corr"].mean()),
            "mean_rank_turnover": float(raw["rank_turnover"].mean()),
            "positive_rank_corr_ratio": float((raw["rank_corr"] > 0.0).mean()),
            "lowest_turnover_time": int(agg.loc[int(best_idx), "time"]),
            "lowest_turnover_value": float(agg.loc[int(best_idx), "mean_rank_turnover"]),
            "highest_turnover_time": int(agg.loc[int(worst_idx), "time"]),
            "highest_turnover_value": float(agg.loc[int(worst_idx), "mean_rank_turnover"]),
        }
    else:
        summary = {"row_count": int(raw.shape[0]), "mean_rank_corr": float("nan"), "mean_rank_turnover": float("nan"), "positive_rank_corr_ratio": float("nan")}
    Path(out_yaml).write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return agg, summary


def residual_diagnostics_from_manifest(manifest_path: Path, out_yaml: Path, out_png: Path) -> dict[str, float]:
    """Compute residual diagnostics from a predict manifest with a bounded reservoir sample."""
    # Stream prediction/target chunks, accumulate moments, and keep a small sample for plotting.
    import yaml

    reader = PredictionChunkReader(Path(manifest_path))
    rng = np.random.default_rng(7)
    sample_cap = 20000
    sample_pred: list[float] = []
    sample_resid: list[float] = []
    sample_seen = 0

    n = 0
    sum_r = 0.0
    sum_r2 = 0.0
    sum_r3 = 0.0
    sum_r4 = 0.0
    sum_p = 0.0
    sum_p2 = 0.0
    sum_pr = 0.0

    for chunk in reader.iter_prediction_chunks(["prediction", "target"]):
        # Build finite residual vectors and update raw power sums.
        pred = chunk["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = chunk["target"].to_numpy(dtype=np.float64, copy=False)
        m = np.isfinite(pred) & np.isfinite(tgt)
        p = pred[m]
        r = (tgt[m] - pred[m]).astype(np.float64, copy=False)
        if int(r.shape[0]) == 0:
            continue

        n_chunk = int(r.shape[0])
        n += int(n_chunk)
        sum_r += float(r.sum(dtype=np.float64))
        sum_r2 += float((r * r).sum(dtype=np.float64))
        sum_r3 += float((r * r * r).sum(dtype=np.float64))
        sum_r4 += float((r * r * r * r).sum(dtype=np.float64))
        sum_p += float(p.sum(dtype=np.float64))
        sum_p2 += float((p * p).sum(dtype=np.float64))
        sum_pr += float((p * r).sum(dtype=np.float64))

        # Update a fixed-size reservoir sample for plot readability.
        for idx in range(n_chunk):
            sample_seen += 1
            if int(len(sample_pred)) < int(sample_cap):
                sample_pred.append(float(p[idx]))
                sample_resid.append(float(r[idx]))
                continue
            j = int(rng.integers(0, int(sample_seen)))
            if int(j) < int(sample_cap):
                sample_pred[int(j)] = float(p[idx])
                sample_resid[int(j)] = float(r[idx])

    # Derive scalar residual statistics from raw moments.
    if int(n) < 2:
        raise RuntimeError("Residual diagnostics require at least 2 finite samples.")
    nf = float(n)
    mean_r = float(sum_r / nf)
    var_r = float(sum_r2 / nf - mean_r * mean_r)
    std_r = float(np.sqrt(max(var_r, 0.0)))
    mse = float(sum_r2 / nf)
    rmse = float(np.sqrt(mse))
    mae = float("nan")

    # Compute MAE in a second pass to avoid storing residuals.
    abs_sum = 0.0
    for chunk in reader.iter_prediction_chunks(["prediction", "target"]):
        # Recompute residual magnitudes for exact MAE.
        pred = chunk["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = chunk["target"].to_numpy(dtype=np.float64, copy=False)
        m = np.isfinite(pred) & np.isfinite(tgt)
        r = (tgt[m] - pred[m]).astype(np.float64, copy=False)
        abs_sum += float(np.abs(r).sum(dtype=np.float64))
    mae = float(abs_sum / nf)

    # Compute skew/kurtosis from central moments.
    mu2 = float(sum_r2 / nf - mean_r * mean_r)
    mu3 = float(sum_r3 / nf - 3.0 * mean_r * (sum_r2 / nf) + 2.0 * mean_r * mean_r * mean_r)
    mu4 = float(sum_r4 / nf - 4.0 * mean_r * (sum_r3 / nf) + 6.0 * mean_r * mean_r * (sum_r2 / nf) - 3.0 * mean_r * mean_r * mean_r * mean_r)
    skew = float(mu3 / (max(mu2, 1e-18) ** 1.5))
    kurt = float(mu4 / (max(mu2, 1e-18) ** 2.0) - 3.0)

    # Compute corr(prediction, residual) from raw sums.
    cov_pr = float(sum_pr / nf - (sum_p / nf) * mean_r)
    var_p = float(sum_p2 / nf - (sum_p / nf) * (sum_p / nf))
    corr_pr = float(cov_pr / float(np.sqrt(max(var_p, 0.0) * max(mu2, 0.0)))) if float(var_p) > 0.0 and float(mu2) > 0.0 else float("nan")

    summary = {
        "count": int(n),
        "residual_mean": float(mean_r),
        "residual_std": float(std_r),
        "residual_skew": float(skew),
        "residual_kurtosis": float(kurt),
        "mae": float(mae),
        "rmse": float(rmse),
        "corr_prediction_residual": float(corr_pr),
    }
    Path(out_yaml).write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Draw a compact histogram plus prediction-vs-residual scatter panel from the sample.
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    resid_sample = np.asarray(sample_resid, dtype=float)
    pred_sample = np.asarray(sample_pred, dtype=float)
    ax1.hist(resid_sample, bins=80, density=True, alpha=0.6, color="#4c72b0")
    grid = np.linspace(float(np.quantile(resid_sample, 0.001)), float(np.quantile(resid_sample, 0.999)), 400)
    ax1.plot(grid, stats.norm.pdf(grid, loc=float(resid_sample.mean()), scale=float(max(resid_sample.std(ddof=0), 1e-12))), color="#dd8452", linewidth=2.0)
    ax1.set_title("Residual distribution (sampled)")
    ax1.set_xlabel("target - prediction")
    ax1.set_ylabel("density")
    ax2.scatter(pred_sample, resid_sample, s=4, alpha=0.15, color="#4c72b0")
    ax2.axhline(0.0, color="#999999", linewidth=1.0)
    ax2.set_title("Residual vs prediction (sampled)")
    ax2.set_xlabel("prediction")
    ax2.set_ylabel("target - prediction")
    fig.tight_layout()
    fig.savefig(Path(out_png), dpi=160)
    plt.close(fig)
    return {str(k): float(v) for k, v in summary.items()}


def score_ret_rank_plot_from_manifest(manifest_path: Path, out_png: Path) -> pd.DataFrame:
    """Plot prediction-rank deciles vs mean target and win-rate from a predict manifest."""
    # Stream timestamps, compute per-timestamp deciles, and accumulate decile-level sums.
    reader = PredictionChunkReader(Path(manifest_path))
    sum_target = np.zeros((10,), dtype=np.float64)
    win_count = np.zeros((10,), dtype=np.int64)
    count = np.zeros((10,), dtype=np.int64)
    for _d, _t, g in reader.iter_timestamp_groups(["date", "time", "prediction", "target"]):
        # Compute decile bins within this timestamp and update global accumulators.
        tmp = g[["prediction", "target"]].dropna(subset=["prediction", "target"]).copy()
        if int(tmp.shape[0]) == 0:
            continue
        pred = tmp["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = tmp["target"].to_numpy(dtype=np.float64, copy=False)
        ranks = stats.rankdata(pred, method="average").astype(np.float64, copy=False)
        pct = ranks / float(pred.shape[0])
        dec = np.minimum((pct * 10.0).astype(np.int64), 9)
        for k in range(10):
            m = dec == int(k)
            if not bool(m.any()):
                continue
            vals = tgt[m]
            sum_target[int(k)] += float(vals.sum(dtype=np.float64))
            win_count[int(k)] += int((vals > 0.0).sum())
            count[int(k)] += int(vals.shape[0])

    # Convert decile accumulators into a small dataframe for plotting.
    rows: list[dict[str, object]] = []
    for k in range(10):
        c = int(count[int(k)])
        mean_t = float(sum_target[int(k)] / float(c)) if int(c) > 0 else float("nan")
        win = float(win_count[int(k)] / float(c)) if int(c) > 0 else float("nan")
        rows.append({"decile": int(k), "mean_target": float(mean_t), "win_rate": float(win), "count": int(c)})
    agg = pd.DataFrame(rows)

    # Plot mean return and win-rate on dual axes.
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    xs = agg["decile"].to_numpy(dtype=int)
    ax1.plot(xs, agg["mean_target"].to_numpy(dtype=float), color="#4c72b0", linewidth=2.0, label="mean target")
    ax2.plot(xs, agg["win_rate"].to_numpy(dtype=float), color="#dd8452", linewidth=2.0, label="win rate")
    ax1.set_xlabel("prediction rank decile (0=low, 9=high)")
    ax1.set_ylabel("mean target")
    ax2.set_ylabel("win rate (target>0)")
    ax1.set_title("Prediction vs target: rank curve")
    fig.tight_layout()
    fig.savefig(Path(out_png), dpi=160)
    plt.close(fig)
    return agg


def rolling_group_ic_from_manifest(manifest_path: Path, config: EvalConfig, label_col: str, out_csv: Path, out_png: Path, out_yaml: Path) -> pd.DataFrame:
    """Compute rolling-group IC curve from a predict manifest via per-date chunk aggregation."""
    # Stream date groups, attach labels per day, and merge rolling-bin accumulators.
    reader = PredictionChunkReader(Path(manifest_path))
    bins: dict[float, dict[str, float]] = {}
    for d, day in reader.iter_date_groups(["date", "time", "code", "prediction", "target"]):
        # Add StockCode/DateTime columns for label merge on this day only.
        day2 = day.copy()
        day2["StockCode"] = day2["code"].astype(int)
        day2["DateTime"] = merge_date_time_dataframe(day2, "date", "time")

        # Attach price and volatility labels from stock1m for this single day.
        labeled = attach_labels(day2, config)

        # Accumulate rolling-window IC bins for this day and merge into the global bins.
        day_bins = _rolling_group_ic_bins(labeled, str(label_col), int(config.window_size), int(config.step_size))
        _rolling_bins_merge(bins, day_bins)

    # Convert merged bins into the final curve table and persist artifacts.
    agg = _rolling_bins_to_curve(bins)
    agg.to_csv(Path(out_csv), index=False)
    if int(agg.shape[0]) == 0:
        warnings.warn("Empty rolling IC curve from manifest.", RuntimeWarning)
        _plot_empty_group_curve(f"{str(label_col)} rolling IC", Path(out_png))
    else:
        _plot_group_curve(agg, f"{str(label_col)} rolling IC", Path(out_png))
    _write_group_curve_summary(agg, Path(out_yaml), f"{str(label_col)}_rolling_ic")
    return agg


def _stock1m_trade_dates(stock1m_dir: Path) -> list[int]:
    """List stock1m trade dates as yyyymmdd integers."""
    # Scan year folders and parse feather filenames into sorted trade dates.
    dates: list[int] = []
    for path in sorted(Path(stock1m_dir).glob("*/*.feather")):
        dates.append(int(Path(path).stem))
    return sorted(dates)


def _market_amount_by_date(config: EvalConfig, dates: list[int]) -> dict[int, float]:
    """Compute full-market traded amount for each requested yyyymmdd date."""
    # Read only Amount so regime construction does not load unnecessary columns.
    out: dict[int, float] = {}
    for d in list(dates):
        year = int(d) // 10000
        path = Path(config.stock1m_dir) / str(year) / f"{int(d)}.feather"
        day = pd.read_feather(path, columns=["Amount"])
        out[int(d)] = float(day["Amount"].to_numpy(dtype=np.float64, copy=False).sum(dtype=np.float64))
    return out


def _prior_5d_market_amount_regime(config: EvalConfig, manifest_dates_yymmdd: list[int]) -> dict[int, int]:
    """Assign each manifest date to a 3-bucket prior-5d market-liquidity regime."""
    # Restrict the stock1m calendar to manifest dates plus the five prior sessions needed for rolling scores.
    manifest_yyyymmdd = sorted([20000000 + int(d) for d in list(manifest_dates_yymmdd)])
    calendar = _stock1m_trade_dates(Path(config.stock1m_dir))
    start_idx = int(calendar.index(int(min(manifest_yyyymmdd))))
    end_idx = int(calendar.index(int(max(manifest_yyyymmdd)))) + 1
    dates = calendar[max(0, int(start_idx) - 5) : int(end_idx)]

    # Build a shifted 5-day rolling market-amount score on the restricted calendar.
    amount_by_date = _market_amount_by_date(config, list(dates))
    cal = pd.DataFrame({"date": list(dates), "amount": [float(amount_by_date[int(d)]) for d in list(dates)]})
    cal["prior_5d_amount"] = cal["amount"].rolling(window=5, min_periods=5).sum().shift(1)

    # Split manifest dates into low/mid/high liquidity regimes by prior-5d market amount.
    keep = cal.loc[cal["date"].isin(list(manifest_yyyymmdd))].dropna(subset=["prior_5d_amount"]).copy()
    keep["regime"] = pd.qcut(keep["prior_5d_amount"], q=3, labels=[0, 1, 2]).astype(int)
    return {int(row["date"]) - 20000000: int(row["regime"]) for row in keep.to_dict(orient="records")}


def _update_prediction_quantile_ic(accs: dict[str, InMemoryPooledICAccumulator], group: pd.DataFrame) -> None:
    """Update prediction top/bottom pooled IC accumulators for one timestamp."""
    # Rank prediction within the timestamp cross-section and select requested quantile regions.
    tmp = group[["prediction", "target"]].dropna(subset=["prediction", "target"]).copy()
    if int(tmp.shape[0]) < 2:
        return
    pct = tmp["prediction"].rank(method="average", pct=True).to_numpy(dtype=np.float64, copy=False)
    pred = tmp["prediction"].to_numpy(dtype=np.float64, copy=False)
    target = tmp["target"].to_numpy(dtype=np.float64, copy=False)

    # Accumulate bottom 90% and top 10% prediction samples into separate pooled ICs.
    accs["prediction_low_p90"].update(pred[pct <= 0.9], target[pct <= 0.9])
    accs["prediction_high_p10"].update(pred[pct > 0.9], target[pct > 0.9])


def _update_label_bucket_ic(accs: dict[str, InMemoryPooledICAccumulator], group: pd.DataFrame, label_col: str, prefix: str) -> None:
    """Update high/low label-bucket pooled IC accumulators for one timestamp."""
    # Rank the requested label within timestamp and split into low/high halves.
    tmp = group[["prediction", "target", label_col]].dropna(subset=["prediction", "target", label_col]).copy()
    if int(tmp.shape[0]) < 2:
        return
    pct = tmp[str(label_col)].rank(method="average", pct=True).to_numpy(dtype=np.float64, copy=False)
    pred = tmp["prediction"].to_numpy(dtype=np.float64, copy=False)
    target = tmp["target"].to_numpy(dtype=np.float64, copy=False)

    # Accumulate low and high bucket samples into independent pooled ICs.
    accs[f"{str(prefix)}_low"].update(pred[pct <= 0.5], target[pct <= 0.5])
    accs[f"{str(prefix)}_high"].update(pred[pct > 0.5], target[pct > 0.5])


def _manifest_dates_from_duckdb(manifest_path: Path) -> list[int]:
    """Read the sorted date list from a predict manifest with DuckDB."""
    # Query distinct dates from parquet metadata/data without materializing prediction rows in Python.
    import duckdb

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=16")
    scan = _duckdb_manifest_scan(Path(manifest_path))
    dates = con.execute(f"SELECT DISTINCT date::INTEGER AS date FROM {scan} ORDER BY date").fetchdf()["date"].astype(int).tolist()
    con.close()
    return [int(d) for d in list(dates)]


def _build_group_ic_label_cache_one_date(task: tuple[int, str, int, str]) -> int:
    """Build one date's grouped-IC label cache parquet."""
    # Decode the multiprocessing task and skip dates that already have a cache file.
    d_yymmdd, stock1m_dir, horizon_minutes, label_dir = task
    out_path = Path(label_dir) / f"{int(d_yymmdd)}.parquet"
    if out_path.exists():
        return int(d_yymmdd)

    # Load same-day stock1m fields and compute labels aligned to qmodel date/time/code keys.
    cfg = EvalConfig(stock1m_dir=Path(stock1m_dir), window_size=1, step_size=1, horizon_minutes=int(horizon_minutes))
    d_yyyymmdd = 20000000 + int(d_yymmdd)
    panel = _load_price_panel_for_dates(cfg, [int(d_yyyymmdd)])
    panel["date"] = int(d_yymmdd)
    panel["time"] = pd.to_datetime(panel["DateTime"]).dt.strftime("%H%M%S").astype(int)
    panel["code"] = panel["StockCode"].astype(np.int64)
    panel["liquidity_label"] = panel["Amount"].astype(np.float64)
    panel["volatility_label"] = _forward_vol_label(panel, int(horizon_minutes)).astype(np.float64)
    panel[["date", "time", "code", "liquidity_label", "volatility_label"]].to_parquet(out_path, index=False)
    return int(d_yymmdd)


def _build_group_ic_label_cache(manifest_path: Path, config: EvalConfig, label_dir: Path) -> list[int]:
    """Build or reuse per-date liquidity/volatility label parquet files."""
    # Resolve manifest dates and create the cache directory used by DuckDB joins.
    dates_yymmdd = _manifest_dates_from_duckdb(Path(manifest_path))
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Materialize missing label files in parallel; existing files are cache hits.
    tasks = [
        (int(d_yymmdd), Path(config.stock1m_dir).as_posix(), int(config.horizon_minutes), Path(label_dir).as_posix())
        for d_yymmdd in list(dates_yymmdd)
        if not (Path(label_dir) / f"{int(d_yymmdd)}.parquet").exists()
    ]
    if len(tasks) > 0:
        workers = min(16, int(len(tasks)))
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=int(workers)) as pool:
            pool.map(_build_group_ic_label_cache_one_date, list(tasks))
    return [int(d) for d in list(dates_yymmdd)]


def _regime_dataframe(config: EvalConfig, manifest_dates_yymmdd: list[int]) -> pd.DataFrame:
    """Build a date-to-regime dataframe from prior-5d full-market traded amount."""
    # Compute prior-5d market amount using the same shifted convention as regime IC.
    manifest_yyyymmdd = sorted([20000000 + int(d) for d in list(manifest_dates_yymmdd)])
    calendar = _stock1m_trade_dates(Path(config.stock1m_dir))
    start_idx = int(calendar.index(int(min(manifest_yyyymmdd))))
    end_idx = int(calendar.index(int(max(manifest_yyyymmdd)))) + 1
    dates = calendar[max(0, int(start_idx) - 5) : int(end_idx)]
    amount_by_date = _market_amount_by_date(config, list(dates))
    cal = pd.DataFrame({"date_yyyymmdd": list(dates), "amount": [float(amount_by_date[int(d)]) for d in list(dates)]})
    cal["prior_5d_amount"] = cal["amount"].rolling(window=5, min_periods=5).sum().shift(1)

    # Convert tertiles into stable group names consumed by the grouped IC query.
    keep = cal.loc[cal["date_yyyymmdd"].isin(list(manifest_yyyymmdd))].dropna(subset=["prior_5d_amount"]).copy()
    keep["date"] = keep["date_yyyymmdd"].astype(int) - 20000000
    keep["regime_id"] = pd.qcut(keep["prior_5d_amount"], q=3, labels=[0, 1, 2]).astype(int)
    names = {0: "regime_low", 1: "regime_mid", 2: "regime_high"}
    keep["metric_group"] = keep["regime_id"].map(dict(names))
    return keep[["date", "metric_group", "prior_5d_amount"]].reset_index(drop=True)


def _duckdb_grouped_ic_dataframe(manifest_path: Path, config: EvalConfig, label_dir: Path, dates_yymmdd: list[int]) -> pd.DataFrame:
    """Compute grouped pooled ICs with DuckDB over prediction and cached label parquet."""
    # Register the small regime dataframe and scan prediction/label parquet directly in DuckDB.
    import duckdb

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=16")
    con.register("regime_df", _regime_dataframe(config, list(dates_yymmdd)))
    pred_scan = _duckdb_manifest_scan(Path(manifest_path))
    label_glob = (Path(label_dir) / "*.parquet").as_posix().replace("'", "''")

    # Compute all group memberships, then rank within each metric_group using average-tie ranks.
    out = con.execute(
        f"""
        WITH pred AS (
            SELECT
                date::INTEGER AS date,
                time::INTEGER AS time,
                code::BIGINT AS code,
                prediction::DOUBLE AS prediction,
                target::DOUBLE AS target
            FROM {pred_scan}
            WHERE isfinite(prediction) AND isfinite(target)
        ),
        prediction_pct AS (
            SELECT *, cume_dist() OVER (PARTITION BY date, time ORDER BY prediction) AS prediction_rank_pct
            FROM pred
        ),
        prediction_groups AS (
            SELECT 'prediction_low_p90' AS metric_group, prediction, target
            FROM prediction_pct
            WHERE prediction_rank_pct <= 0.9
            UNION ALL
            SELECT 'prediction_high_p10' AS metric_group, prediction, target
            FROM prediction_pct
            WHERE prediction_rank_pct > 0.9
        ),
        joined AS (
            SELECT
                pred.date,
                pred.time,
                pred.code,
                pred.prediction,
                pred.target,
                label.liquidity_label::DOUBLE AS liquidity_label,
                label.volatility_label::DOUBLE AS volatility_label
            FROM pred
            JOIN read_parquet('{label_glob}') AS label
            USING (date, time, code)
        ),
        liquidity_pct AS (
            SELECT *, cume_dist() OVER (PARTITION BY date, time ORDER BY liquidity_label) AS liquidity_rank_pct
            FROM joined
            WHERE isfinite(liquidity_label)
        ),
        volatility_pct AS (
            SELECT *, cume_dist() OVER (PARTITION BY date, time ORDER BY volatility_label) AS volatility_rank_pct
            FROM joined
            WHERE isfinite(volatility_label)
        ),
        label_groups AS (
            SELECT 'liquidity_low' AS metric_group, prediction, target
            FROM liquidity_pct
            WHERE liquidity_rank_pct <= 0.5
            UNION ALL
            SELECT 'liquidity_high' AS metric_group, prediction, target
            FROM liquidity_pct
            WHERE liquidity_rank_pct > 0.5
            UNION ALL
            SELECT 'volatility_low' AS metric_group, prediction, target
            FROM volatility_pct
            WHERE volatility_rank_pct <= 0.5
            UNION ALL
            SELECT 'volatility_high' AS metric_group, prediction, target
            FROM volatility_pct
            WHERE volatility_rank_pct > 0.5
        ),
        regime_groups AS (
            SELECT regime_df.metric_group, pred.prediction, pred.target
            FROM pred
            JOIN regime_df USING (date)
        ),
        selected AS (
            SELECT * FROM prediction_groups
            UNION ALL
            SELECT * FROM label_groups
            UNION ALL
            SELECT * FROM regime_groups
        ),
        ranked AS (
            SELECT
                metric_group,
                prediction,
                target,
                rank() OVER (PARTITION BY metric_group ORDER BY prediction)
                    + (count(*) OVER (PARTITION BY metric_group, prediction) - 1) / 2.0 AS prediction_rank,
                rank() OVER (PARTITION BY metric_group ORDER BY target)
                    + (count(*) OVER (PARTITION BY metric_group, target) - 1) / 2.0 AS target_rank
            FROM selected
        )
        SELECT
            metric_group AS "group",
            corr(prediction, target) AS pearson_ic,
            corr(prediction_rank, target_rank) AS rank_ic,
            count(*) AS count
        FROM ranked
        GROUP BY metric_group
        ORDER BY metric_group
        """
    ).fetchdf()
    con.close()
    return out


def grouped_pooled_ic_from_manifest(manifest_path: Path, config: EvalConfig, out_csv: Path, out_yaml: Path) -> dict[str, object]:
    """Compute prediction-quantile, liquidity, volatility, and regime pooled IC metrics."""
    # Build or reuse label cache beside the output CSV and compute grouped ICs in DuckDB.
    import yaml

    label_dir = Path(out_csv).parent / f"group_ic_label_cache_h{int(config.horizon_minutes)}"
    dates_yymmdd = _build_group_ic_label_cache(Path(manifest_path), config, Path(label_dir))
    out = _duckdb_grouped_ic_dataframe(Path(manifest_path), config, Path(label_dir), list(dates_yymmdd))
    out.to_csv(Path(out_csv), index=False)
    summary = {
        "label_cache_dir": Path(label_dir).as_posix(),
        "groups": {str(row["group"]): {k: row[k] for k in ["pearson_ic", "rank_ic", "count"]} for row in out.to_dict(orient="records")},
    }
    Path(out_yaml).write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return summary


@dataclass(frozen=True)
class TestEvaluationReportArtifacts:
    """Bundle the test-evaluation streaming report outputs computed from a manifest."""

    pooled: dict[str, float]
    pooled_nonzero_prediction: dict[str, float]
    top_decile_return: dict[str, float]
    top_decile_return_yaml: Path
    ic_summary: dict[str, object]
    ic_summary_yaml: Path
    core_ic: dict[str, object]
    core_ic_yaml: Path
    grouped_ic: dict[str, object]
    grouped_ic_csv: Path
    grouped_ic_yaml: Path
    annual_tbl: pd.DataFrame
    annual_csv: Path
    annual_png: Path
    intraday_csv: Path
    intraday_png: Path
    rank_png: Path
    turnover_csv: Path
    turnover_png: Path
    turnover_yaml: Path
    turnover_summary: dict[str, object]
    residual_yaml: Path
    residual_png: Path
    residual_summary: dict[str, object]
    vol_curve: pd.DataFrame
    vol_csv: Path
    vol_png: Path
    vol_yaml: Path
    price_curve: pd.DataFrame
    price_csv: Path
    price_png: Path
    price_yaml: Path


def compute_test_evaluation_report_from_manifest(manifest_path: Path, eval_cfg: EvalConfig, out_root: Path) -> TestEvaluationReportArtifacts:
    """Compute the test-evaluation report artifacts by streaming parquet chunks from a manifest."""
    # Resolve all output paths under the report root to keep pipeline wiring minimal.
    import yaml

    out_root = Path(out_root)
    ic_summary_yaml = Path(out_root) / "test_daily_ic_summary.yaml"
    core_ic_yaml = Path(out_root) / "test_core_ic_summary.yaml"
    grouped_ic_csv = Path(out_root) / "test_grouped_pooled_ic.csv"
    grouped_ic_yaml = Path(out_root) / "test_grouped_pooled_ic.yaml"
    annual_csv = Path(out_root) / "annual_ic.csv"
    annual_png = Path(out_root) / "annual_ic.png"
    intraday_csv = Path(out_root) / "test_intraday_ic.csv"
    intraday_png = Path(out_root) / "test_intraday_ic.png"
    rank_png = Path(out_root) / "test_pred_vs_target_rank.png"
    turnover_csv = Path(out_root) / "test_prediction_rank_turnover.csv"
    turnover_png = Path(out_root) / "test_prediction_rank_turnover.png"
    turnover_yaml = Path(out_root) / "test_prediction_rank_turnover.yaml"
    residual_yaml = Path(out_root) / "test_residual_diagnostics.yaml"
    residual_png = Path(out_root) / "test_residual_diagnostics.png"
    vol_csv = Path(out_root) / "test_vol_rolling_ic.csv"
    vol_png = Path(out_root) / "test_vol_rolling_ic.png"
    vol_yaml = Path(out_root) / "test_vol_rolling_ic.yaml"
    price_csv = Path(out_root) / "test_price_rolling_ic.csv"
    price_png = Path(out_root) / "test_price_rolling_ic.png"
    price_yaml = Path(out_root) / "test_price_rolling_ic.yaml"

    # Compute exact pooled Pearson/Rank IC and pooled tail return diagnostics.
    pooled = pooled_ic_from_manifest(Path(manifest_path))
    pooled_nonzero_prediction = pooled_nonzero_prediction_ic_from_manifest(Path(manifest_path))
    top_decile_return = pooled_top_decile_return_from_manifest(Path(manifest_path))
    top_decile_return_yaml = Path(out_root) / "test_top_decile_return.yaml"
    Path(top_decile_return_yaml).write_text(yaml.safe_dump(top_decile_return, sort_keys=False, allow_unicode=True), encoding="utf-8")
    core_ic = {
        "pooled": dict(pooled),
        "pooled_nonzero_prediction": dict(pooled_nonzero_prediction),
        "rolling_rank_ic": {
            "status": "skipped",
            "reason": "timestamp cross-sectional IC is temporarily disabled in report postprocess",
        },
    }
    Path(core_ic_yaml).write_text(yaml.safe_dump(core_ic, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Write a skipped daily-IC artifact so downstream readers can distinguish skipped from missing.
    ic_summary = {
        "split": "test",
        "status": "skipped",
        "reason": "timestamp cross-sectional IC is temporarily disabled in report postprocess",
    }
    Path(ic_summary_yaml).write_text(yaml.safe_dump(ic_summary, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Compute annual pooled Pearson IC by streaming and bucketing rows by year.
    annual_tbl = annual_pooled_pearson_ic_from_manifest(Path(manifest_path), Path(annual_csv), Path(annual_png))

    # Remove stale intraday artifacts because this report no longer computes timestamp IC curves.
    Path(intraday_csv).unlink(missing_ok=True)
    Path(intraday_png).unlink(missing_ok=True)

    # Compute test-side diagnostics that depend on per-timestamp ranks or residuals.
    score_ret_rank_plot_from_manifest(Path(manifest_path), Path(rank_png))
    _turnover_tbl, turnover_summary = prediction_rank_turnover_from_manifest(Path(manifest_path), Path(turnover_csv), Path(turnover_png), Path(turnover_yaml))
    residual_summary = residual_diagnostics_from_manifest(Path(manifest_path), Path(residual_yaml), Path(residual_png))

    # Compute rolling-group IC curves by streaming one date at a time and attaching labels per day.
    vol_curve = rolling_group_ic_from_manifest(Path(manifest_path), eval_cfg, "volatility_label", Path(vol_csv), Path(vol_png), Path(vol_yaml))
    price_curve = rolling_group_ic_from_manifest(Path(manifest_path), eval_cfg, "price_label", Path(price_csv), Path(price_png), Path(price_yaml))
    grouped_ic = grouped_pooled_ic_from_manifest(Path(manifest_path), eval_cfg, Path(grouped_ic_csv), Path(grouped_ic_yaml))

    # Return a compact artifact bundle so the pipeline can render report.html without extra IO.
    return TestEvaluationReportArtifacts(
        pooled=dict(pooled),
        pooled_nonzero_prediction=dict(pooled_nonzero_prediction),
        top_decile_return=dict(top_decile_return),
        top_decile_return_yaml=Path(top_decile_return_yaml),
        ic_summary=dict(ic_summary),
        ic_summary_yaml=Path(ic_summary_yaml),
        core_ic=dict(core_ic),
        core_ic_yaml=Path(core_ic_yaml),
        grouped_ic=dict(grouped_ic),
        grouped_ic_csv=Path(grouped_ic_csv),
        grouped_ic_yaml=Path(grouped_ic_yaml),
        annual_tbl=annual_tbl,
        annual_csv=Path(annual_csv),
        annual_png=Path(annual_png),
        intraday_csv=Path(intraday_csv),
        intraday_png=Path(intraday_png),
        rank_png=Path(rank_png),
        turnover_csv=Path(turnover_csv),
        turnover_png=Path(turnover_png),
        turnover_yaml=Path(turnover_yaml),
        turnover_summary=dict(turnover_summary),
        residual_yaml=Path(residual_yaml),
        residual_png=Path(residual_png),
        residual_summary=dict(residual_summary),
        vol_curve=vol_curve,
        vol_csv=Path(vol_csv),
        vol_png=Path(vol_png),
        vol_yaml=Path(vol_yaml),
        price_curve=price_curve,
        price_csv=Path(price_csv),
        price_png=Path(price_png),
        price_yaml=Path(price_yaml),
    )


def pooled_ic(df: pd.DataFrame) -> dict[str, float]:
    """Compute pooled IC across all (stock,time) samples."""
    # Compute Pearson and Spearman across the full dataframe.
    pred = df["prediction"].to_numpy(dtype=float)
    tgt = df["target"].to_numpy(dtype=float)
    return {"pearson_ic": _pearson(pred, tgt), "rank_ic": _spearman(pred, tgt), "count": int(np.isfinite(pred).sum())}


def cross_sectional_ic_series(df: pd.DataFrame) -> pd.DataFrame:
    """Compute one cross-sectional IC row for each timestamp."""
    # Build one IC observation per (date, time) cross-section.
    tmp = df[["date", "time", "prediction", "target"]].copy()
    rows: list[dict[str, object]] = []
    for (d, t), g in tmp.groupby(["date", "time"], sort=True):
        # Compute Pearson and Rank IC on this timestamp cross-section.
        pred = g["prediction"].to_numpy(dtype=float)
        tgt = g["target"].to_numpy(dtype=float)
        rows.append(
            {
                "date": int(d),
                "time": int(t),
                "ic": _pearson(pred, tgt),
                "rank_ic": _spearman(pred, tgt),
                "count": int(np.isfinite(pred).sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["date", "time"], kind="stable").reset_index(drop=True)
    return out


def ic_time_series_summary(df: pd.DataFrame, out_yaml: Path) -> dict[str, float]:
    """Write timestamp-level IC summary metrics including t-stat and positive ratio."""
    # Build the timestamp IC series that the summary statistics operate on.
    import yaml

    cs = cross_sectional_ic_series(df)
    ic = cs["ic"].to_numpy(dtype=float)
    rank_ic = cs["rank_ic"].to_numpy(dtype=float)

    # Compute t-stat and sign ratio on finite timestamp IC observations.
    def _summary(xs: np.ndarray) -> dict[str, float]:
        """Summarize one IC series with mean, std, t-stat, and sign ratio."""
        # Filter finite values and compute scalar diagnostics.
        vals = xs[np.isfinite(xs)]
        n = int(vals.shape[0])
        mean = float(vals.mean()) if int(n) > 0 else float("nan")
        std = float(vals.std(ddof=1)) if int(n) > 1 else float("nan")
        t_stat = float(mean / (std / np.sqrt(float(n)))) if int(n) > 1 and float(std) > 0.0 else float("nan")
        positive_ratio = float((vals > 0.0).mean()) if int(n) > 0 else float("nan")
        return {"count": int(n), "mean": float(mean), "std": float(std), "t_stat": float(t_stat), "positive_ratio": float(positive_ratio)}

    # Persist a compact YAML summary for report ingestion.
    ic_summary = _summary(ic)
    rank_ic_summary = _summary(rank_ic)
    summary = {
        "pearson_ic": dict(ic_summary),
        "rank_ic": dict(rank_ic_summary),
        "timestamp_count": int(cs.shape[0]),
    }
    out_yaml.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return summary


def intraday_time_series_ic(df: pd.DataFrame, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute intraday minute-of-day aggregated cross-sectional IC curve."""
    # Compute per-minute cross-sectional IC for each day.
    cs = cross_sectional_ic_series(df).rename(columns={"count": "n"})

    # Aggregate across dates by minute-of-day.
    agg = cs.groupby("time", sort=True).agg(
        mean_ic=("ic", "mean"),
        std_ic=("ic", "std"),
        mean_rank_ic=("rank_ic", "mean"),
        std_rank_ic=("rank_ic", "std"),
        count=("ic", "count"),
    )
    agg = agg.reset_index().sort_values("time", kind="stable").reset_index(drop=True)
    agg.to_csv(out_csv, index=False)

    # Plot the intraday mean IC curve on an HH:MM axis.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    xs = agg["time"].to_numpy(dtype=int)
    labels = [f"{int(t)//10000:02d}:{(int(t)%10000)//100:02d}" for t in xs]
    ax.plot(np.arange(len(labels)), agg["mean_ic"].to_numpy(dtype=float), label="Pearson IC", linewidth=1.8)
    ax.plot(np.arange(len(labels)), agg["mean_rank_ic"].to_numpy(dtype=float), label="Rank IC (Spearman)", linewidth=1.8)
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title("Intraday IC curve (mean across dates)")
    ax.set_xlabel("time (minute bars; lunch break absent)")
    ax.set_ylabel("mean IC")
    tick_pos = np.linspace(0, max(len(labels) - 1, 1), 10).round().astype(int)
    ax.set_xticks(tick_pos, [labels[i] for i in tick_pos], rotation=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return agg


def intraday_time_series_ic_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute intraday minute-of-day Pearson IC curves for train and test on one axis."""
    # Compute the per-minute Pearson IC series for train and aggregate by minute-of-day.
    train_cs = cross_sectional_ic_series(train_df).rename(columns={"count": "n"})
    train_agg = train_cs.groupby("time", sort=True).agg(
        mean_ic_train=("ic", "mean"),
        std_ic_train=("ic", "std"),
        count_train=("ic", "count"),
    )
    train_agg = train_agg.reset_index().sort_values("time", kind="stable").reset_index(drop=True)

    # Compute the per-minute Pearson IC series for test and aggregate by minute-of-day.
    test_cs = cross_sectional_ic_series(test_df).rename(columns={"count": "n"})
    test_agg = test_cs.groupby("time", sort=True).agg(
        mean_ic_test=("ic", "mean"),
        std_ic_test=("ic", "std"),
        count_test=("ic", "count"),
    )
    test_agg = test_agg.reset_index().sort_values("time", kind="stable").reset_index(drop=True)

    # Merge the aligned minute-of-day curves so we can plot them together.
    agg = train_agg.merge(test_agg, on="time", how="inner")
    agg.to_csv(out_csv, index=False)

    # Plot the intraday mean Pearson IC curves on an HH:MM axis.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    xs = agg["time"].to_numpy(dtype=int)
    labels = [f"{int(t)//10000:02d}:{(int(t)%10000)//100:02d}" for t in xs]
    ax.plot(np.arange(len(labels)), agg["mean_ic_train"].to_numpy(dtype=float), label="Train Pearson IC", linewidth=1.8)
    ax.plot(np.arange(len(labels)), agg["mean_ic_test"].to_numpy(dtype=float), label="Test Pearson IC", linewidth=1.8)
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title("Intraday IC curve (Pearson; mean across dates)")
    ax.set_xlabel("time (minute bars; lunch break absent)")
    ax.set_ylabel("mean IC")
    tick_pos = np.linspace(0, max(len(labels) - 1, 1), 10).round().astype(int)
    ax.set_xticks(tick_pos, [labels[i] for i in tick_pos], rotation=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return agg


def prediction_rank_turnover(df: pd.DataFrame, out_csv: Path, out_png: Path, out_yaml: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute adjacent-timestamp prediction-rank turnover and persist diagnostics."""
    # Build one prediction table per timestamp so adjacent times can be matched by stock code.
    import yaml

    tmp = df[["date", "time", "StockCode", "prediction"]].dropna(subset=["prediction"]).copy()
    rows: list[dict[str, object]] = []
    for d, day in tmp.groupby("date", sort=True):
        # Sort timestamps inside one day and evaluate adjacent rank correlation.
        day_tables = {int(t): g[["StockCode", "prediction"]].copy() for t, g in day.groupby("time", sort=True)}
        day_times = sorted(day_tables.keys())
        for idx in range(1, len(day_times)):
            # Align consecutive timestamps on the stock intersection.
            prev_time = int(day_times[idx - 1])
            curr_time = int(day_times[idx])
            prev_tbl = day_tables[int(prev_time)].rename(columns={"prediction": "prediction_prev"})
            curr_tbl = day_tables[int(curr_time)].rename(columns={"prediction": "prediction_curr"})
            merged = prev_tbl.merge(curr_tbl, on="StockCode", how="inner")
            if int(merged.shape[0]) < 2:
                continue

            # Rank predictions within each timestamp and compute adjacent correlation.
            merged["rank_prev"] = merged["prediction_prev"].rank(method="average")
            merged["rank_curr"] = merged["prediction_curr"].rank(method="average")
            rank_prev = merged["rank_prev"].to_numpy(dtype=float)
            rank_curr = merged["rank_curr"].to_numpy(dtype=float)
            rank_corr = _pearson(rank_prev, rank_curr)
            rank_turnover = float(1.0 - rank_corr) if np.isfinite(rank_corr) else float("nan")
            rows.append(
                {
                    "date": int(d),
                    "prev_time": int(prev_time),
                    "time": int(curr_time),
                    "rank_corr": float(rank_corr),
                    "rank_turnover": float(rank_turnover),
                    "count": int(merged.shape[0]),
                }
            )

    # Aggregate adjacent-turnover rows by minute-of-day for a stable intraday curve.
    raw = pd.DataFrame(rows).sort_values(["date", "time"], kind="stable").reset_index(drop=True)
    agg = raw.groupby("time", sort=True).agg(
        mean_rank_corr=("rank_corr", "mean"),
        mean_rank_turnover=("rank_turnover", "mean"),
        std_rank_turnover=("rank_turnover", "std"),
        count=("rank_turnover", "count"),
    )
    agg = agg.reset_index().sort_values("time", kind="stable").reset_index(drop=True)
    agg.to_csv(out_csv, index=False)

    # Plot the mean adjacent-turnover curve across the trading day.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    xs = agg["time"].to_numpy(dtype=int)
    labels = [f"{int(t)//10000:02d}:{(int(t)%10000)//100:02d}" for t in xs]
    ax.plot(np.arange(len(labels)), agg["mean_rank_turnover"].to_numpy(dtype=float), label="1 - corr(rank_t, rank_t-1)", linewidth=1.8)
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title("Prediction rank turnover (adjacent timestamps)")
    ax.set_xlabel("time")
    ax.set_ylabel("mean turnover")
    tick_pos = np.linspace(0, max(len(labels) - 1, 1), 10).round().astype(int)
    ax.set_xticks(tick_pos, [labels[i] for i in tick_pos], rotation=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    # Persist a compact YAML summary for report consumption.
    best_idx = int(agg["mean_rank_turnover"].idxmin())
    worst_idx = int(agg["mean_rank_turnover"].idxmax())
    summary = {
        "row_count": int(raw.shape[0]),
        "mean_rank_corr": float(raw["rank_corr"].mean()),
        "mean_rank_turnover": float(raw["rank_turnover"].mean()),
        "positive_rank_corr_ratio": float((raw["rank_corr"] > 0.0).mean()),
        "lowest_turnover_time": int(agg.loc[int(best_idx), "time"]),
        "lowest_turnover_value": float(agg.loc[int(best_idx), "mean_rank_turnover"]),
        "highest_turnover_time": int(agg.loc[int(worst_idx), "time"]),
        "highest_turnover_value": float(agg.loc[int(worst_idx), "mean_rank_turnover"]),
    }
    out_yaml.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return agg, summary


def residual_diagnostics(df: pd.DataFrame, out_yaml: Path, out_png: Path) -> dict[str, float]:
    """Compute residual summary statistics and write a compact residual plot."""
    # Build finite prediction/target vectors and derive residuals.
    import yaml

    tmp = df[["prediction", "target"]].dropna(subset=["prediction", "target"]).copy()
    pred = tmp["prediction"].to_numpy(dtype=float)
    tgt = tmp["target"].to_numpy(dtype=float)
    resid = tgt - pred

    # Compute scalar residual diagnostics for report consumption.
    mse = float(np.mean(resid * resid))
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(mse))
    resid_std = float(resid.std(ddof=1)) if int(resid.shape[0]) > 1 else float("nan")
    summary = {
        "count": int(resid.shape[0]),
        "residual_mean": float(resid.mean()),
        "residual_std": float(resid_std),
        "residual_skew": float(stats.skew(resid, bias=False)),
        "residual_kurtosis": float(stats.kurtosis(resid, fisher=True, bias=False)),
        "mae": float(mae),
        "rmse": float(rmse),
        "corr_prediction_residual": float(_pearson(pred, resid)),
    }
    out_yaml.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Draw a compact histogram plus prediction-vs-residual scatter panel.
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.hist(resid, bins=80, density=True, alpha=0.6, color="#4c72b0")
    grid = np.linspace(float(np.quantile(resid, 0.001)), float(np.quantile(resid, 0.999)), 400)
    ax1.plot(grid, stats.norm.pdf(grid, loc=float(resid.mean()), scale=float(max(resid.std(ddof=0), 1e-12))), color="#dd8452", linewidth=2.0)
    ax1.set_title("Residual distribution")
    ax1.set_xlabel("target - prediction")
    ax1.set_ylabel("density")

    # Downsample deterministically so the scatter remains readable on large test sets.
    sample_n = min(int(pred.shape[0]), 20000)
    sample_idx = np.linspace(0, max(int(pred.shape[0]) - 1, 0), sample_n, dtype=int)
    ax2.scatter(pred[sample_idx], resid[sample_idx], s=4, alpha=0.15, color="#4c72b0")
    ax2.axhline(0.0, color="#999999", linewidth=1.0)
    ax2.set_title("Residual vs prediction")
    ax2.set_xlabel("prediction")
    ax2.set_ylabel("target - prediction")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return summary


def _load_price_panel_for_dates(config: EvalConfig, dates: list[int]) -> pd.DataFrame:
    """Load price/liquidity fields for the requested trade dates from stock1m."""
    # Read Close/Amount and DateTime for each date and concatenate into one table.
    parts: list[pd.DataFrame] = []
    for d in list(dates):
        # Resolve file path by year folder convention.
        year = int(d) // 10000
        path = Path(config.stock1m_dir) / str(year) / f"{int(d)}.feather"
        day = pd.read_feather(path, columns=["StockCode", "DateTime", "Close", "Amount", "Date"])
        day = day.sort_values(["StockCode", "DateTime"], kind="stable").reset_index(drop=True)
        parts.append(day)
    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


def _forward_vol_label(day: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    """Compute per-row forward volatility label using next-horizon 1m returns std."""
    # Build log close and 1m returns per stock with NaN for invalid prices.
    close = day["Close"].to_numpy(dtype=float)
    m = np.isfinite(close) & (close > 0.0)
    log_close = np.full_like(close, np.nan, dtype=float)
    log_close[m] = np.log(close[m])
    day = day.copy()
    day["log_close"] = log_close
    day["r1"] = day.groupby("StockCode", sort=False)["log_close"].diff(1)

    # Compute forward std of r1[t+1:t+h] using groupby+rolling vectorization.
    h = int(horizon_minutes)
    g = day.groupby("StockCode", sort=False)["r1"]
    r_next = g.shift(-1)
    vol = (
        r_next.groupby(day["StockCode"], sort=False)
        .rolling(window=h, min_periods=2)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
        .shift(-(h - 1))
    )
    return vol.astype(np.float32, copy=False).rename("volatility_label")


def attach_labels(pred_df: pd.DataFrame, config: EvalConfig) -> pd.DataFrame:
    """Attach volatility_label and price_label to the prediction dataframe."""
    # Resolve trade_date list from pred_df by converting yymmdd to yyyymmdd.
    yymmdd = pred_df["date"].astype(int).unique().tolist()
    yyyymmdd = [20000000 + int(d) for d in yymmdd]

    # Merge per-day labels in a loop to keep peak memory bounded on multi-year spans.
    parts: list[pd.DataFrame] = []
    for d_yymmdd, d_yyyymmdd in zip(yymmdd, yyyymmdd):
        # Select prediction rows for one trade date to minimize merge payload.
        day_pred = pred_df.loc[pred_df["date"].astype(int) == int(d_yymmdd)].copy()

        # Load price panel for this date and compute same-day labels.
        panel = _load_price_panel_for_dates(config, [int(d_yyyymmdd)])
        panel["price_label"] = panel["Close"].astype(float)
        panel["liquidity_label"] = panel["Amount"].astype(float)
        panel["volatility_label"] = _forward_vol_label(panel, int(config.horizon_minutes)).astype(float)

        # Merge labels onto prediction rows using (StockCode, DateTime) keys.
        key_cols = ["StockCode", "DateTime"]
        day_merged = day_pred.merge(panel[key_cols + ["price_label", "liquidity_label", "volatility_label"]], on=key_cols, how="left", validate="many_to_one")
        parts.append(day_merged)

    # Concatenate day merges back into one dataframe in stable original order.
    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


def _attach_labeled_day_for_date_task(d_yymmdd: int) -> pd.DataFrame:
    """Attach same-day price and volatility labels for one trade date."""
    # Resolve globals populated by the parallel rolling driver.
    if _ROLLING_GLOBAL_PRED_DF is None or _ROLLING_GLOBAL_IDX_BY_DATE is None or _ROLLING_GLOBAL_EVAL_CFG is None:
        raise RuntimeError("Rolling globals not initialized.")

    # Select the day's prediction rows by precomputed indices.
    idx = _ROLLING_GLOBAL_IDX_BY_DATE[int(d_yymmdd)]
    day_pred = _ROLLING_GLOBAL_PRED_DF.iloc[idx].copy()

    # Load the day's minute-bar panel and restrict to the union of sampled stocks.
    cfg = _ROLLING_GLOBAL_EVAL_CFG
    d_yyyymmdd = 20000000 + int(d_yymmdd)
    panel = _load_price_panel_for_dates(cfg, [int(d_yyyymmdd)])
    codes = day_pred["StockCode"].unique().tolist()
    panel = panel.loc[panel["StockCode"].isin(list(codes))].reset_index(drop=True)

    # Compute price and forward-volatility labels on the filtered panel.
    panel["price_label"] = panel["Close"].astype(float)
    panel["liquidity_label"] = panel["Amount"].astype(float)
    panel["volatility_label"] = _forward_vol_label(panel, int(cfg.horizon_minutes)).astype(float)

    # Merge labels onto prediction rows using (StockCode, DateTime) keys.
    key_cols = ["StockCode", "DateTime"]
    day_merged = day_pred.merge(panel[key_cols + ["price_label", "liquidity_label", "volatility_label"]], on=key_cols, how="left", validate="many_to_one")
    return day_merged


def rolling_group_ic_parallel(pred_df: pd.DataFrame, config: EvalConfig, label_col: str, workers: int) -> pd.DataFrame:
    """Compute pooled rolling-window IC by label via per-date multiprocessing."""
    # Build a fast index map from date to row indices so workers avoid full scans.
    dates = pred_df["date"].astype(int)
    idx_by_date = pred_df.groupby(dates, sort=False).indices

    # Publish the prediction dataframe and index map into module globals for forked workers.
    global _ROLLING_GLOBAL_PRED_DF, _ROLLING_GLOBAL_IDX_BY_DATE, _ROLLING_GLOBAL_EVAL_CFG
    _ROLLING_GLOBAL_PRED_DF = pred_df
    _ROLLING_GLOBAL_IDX_BY_DATE = {int(k): np.asarray(v, dtype=np.int64) for k, v in idx_by_date.items()}
    _ROLLING_GLOBAL_EVAL_CFG = config

    # Run one task per trade date and concatenate labeled rows in the parent process.
    tasks = [int(d) for d in sorted(_ROLLING_GLOBAL_IDX_BY_DATE.keys())]
    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=int(workers))
    try:
        parts: list[pd.DataFrame] = []
        for rows in pool.imap(_attach_labeled_day_for_date_task, tasks, chunksize=1):
            if int(rows.shape[0]) == 0:
                continue
            parts.append(rows)
    finally:
        pool.close()
        pool.join()

    # Concatenate per-date labeled rows and compute one pooled rolling curve.
    if len(parts) == 0:
        return _empty_group_schema()
    labeled = pd.concat(parts, axis=0).reset_index(drop=True)
    out = _rolling_group_ic(labeled, str(label_col), int(config.window_size), int(config.step_size))
    return out


def annual_pooled_ic(df: pd.DataFrame, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute pooled IC per calendar year and persist a CSV plus bar plot."""
    # Compute year integer from yymmdd date and keep only finite prediction/target rows.
    tmp = df[["date", "prediction", "target"]].dropna(subset=["prediction", "target"]).copy()
    tmp["year"] = (2000 + (tmp["date"].astype(int) // 10000)).astype(int)

    # Aggregate pooled Pearson/Spearman IC per year.
    rows: list[dict[str, object]] = []
    for y, g in tmp.groupby("year", sort=True):
        # Compute correlations using the shared correlation helpers.
        pred = g["prediction"].to_numpy(dtype=float)
        tgt = g["target"].to_numpy(dtype=float)
        rows.append({"year": int(y), "pearson_ic": _pearson(pred, tgt), "rank_ic": _spearman(pred, tgt), "count": int(np.isfinite(pred).sum())})
    out = pd.DataFrame(rows).sort_values("year", kind="stable").reset_index(drop=True)
    out.to_csv(out_csv, index=False)

    # Skip a one-point figure because the table is the clearer representation.
    if int(out.shape[0]) <= 1:
        Path(out_png).unlink(missing_ok=True)
        return out

    # Plot yearly IC bars for Pearson and Rank IC.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    xs = out["year"].to_numpy(dtype=int)
    ax.bar(xs - 0.15, out["pearson_ic"].to_numpy(dtype=float), width=0.3, label="Pearson IC")
    ax.bar(xs + 0.15, out["rank_ic"].to_numpy(dtype=float), width=0.3, label="Rank IC (Spearman)")
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title("Annual pooled IC (prediction vs target)")
    ax.set_xlabel("year")
    ax.set_ylabel("IC")
    ax.set_xticks(xs, [str(int(x)) for x in xs])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out


def _rolling_group_ic(
    df: pd.DataFrame,
    label_col: str,
    window_size: int,
    step_size: int,
) -> pd.DataFrame:
    """Compute aggregated rolling-window IC over cross-sections sorted by a label."""
    # Aggregate rolling windows into compact rank bins instead of materializing every window row.
    bins = _rolling_group_ic_bins(df, label_col, int(window_size), int(step_size))
    return _rolling_bins_to_curve(bins)


def _rolling_group_ic_rows(
    df: pd.DataFrame,
    label_col: str,
    window_size: int,
    step_size: int,
) -> pd.DataFrame:
    """Compute pooled raw rolling-window IC rows on one globally sorted label axis."""
    # Define helpers to compute Pearson correlation from windowed prefix sums.
    def _corr_from_sums(sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float, w: int) -> float:
        """Compute Pearson correlation from raw sums on a fixed window."""
        # Compute covariance and variances with stable float math.
        wf = float(w)
        cov = float(sum_xy - (sum_x * sum_y) / wf)
        var_x = float(sum_x2 - (sum_x * sum_x) / wf)
        var_y = float(sum_y2 - (sum_y * sum_y) / wf)
        if not (np.isfinite(cov) and np.isfinite(var_x) and np.isfinite(var_y)):
            return float("nan")
        if float(var_x) <= 0.0 or float(var_y) <= 0.0:
            return float("nan")
        return float(cov / float(np.sqrt(var_x * var_y)))

    # Sort the full pooled sample by the label and keep only finite rows.
    gg = df[["prediction", "target", label_col]].dropna(subset=["prediction", "target", label_col]).sort_values(label_col, kind="stable").reset_index(drop=True)
    n = int(gg.shape[0])
    w = int(window_size)
    step = int(step_size)
    if int(n) < int(w):
        return pd.DataFrame([])

    # Extract pooled arrays and cast once for stable prefix-sum accumulation.
    pred = gg["prediction"].to_numpy(dtype=np.float64, copy=False)
    tgt = gg["target"].to_numpy(dtype=np.float64, copy=False)
    label = gg[label_col].to_numpy(dtype=np.float64, copy=False)

    # Build prefix sums for Pearson IC computation on raw values.
    ps = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pred, dtype=np.float64)])
    ts = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tgt, dtype=np.float64)])
    p2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pred * pred, dtype=np.float64)])
    t2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tgt * tgt, dtype=np.float64)])
    pts = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pred * tgt, dtype=np.float64)])

    # Build pooled ranks once and reuse for the rolling-window rank IC approximation.
    pr = stats.rankdata(pred, method="average").astype(np.float64, copy=False)
    tr = stats.rankdata(tgt, method="average").astype(np.float64, copy=False)
    prs = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pr, dtype=np.float64)])
    trs = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tr, dtype=np.float64)])
    pr2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pr * pr, dtype=np.float64)])
    tr2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tr * tr, dtype=np.float64)])
    prts = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pr * tr, dtype=np.float64)])

    # Collect one output row per rolling window on the pooled sorted sequence.
    rows: list[dict[str, object]] = []
    for st in range(0, int(n - w + 1), int(step)):
        # Compute the pooled window's center rank percentile on the globally sorted sequence.
        center = float(st + w * 0.5)
        center_rank = float(center / float(n))

        # Compute Pearson IC from prefix sums on raw prediction/target.
        ed = int(st + w)
        sum_p = float(ps[ed] - ps[st])
        sum_t = float(ts[ed] - ts[st])
        sum_p2 = float(p2s[ed] - p2s[st])
        sum_t2 = float(t2s[ed] - t2s[st])
        sum_pt = float(pts[ed] - pts[st])
        ic = _corr_from_sums(sum_p, sum_t, sum_p2, sum_t2, sum_pt, w)

        # Compute rank IC from prefix sums on pooled ranks restricted to the same window.
        sum_pr = float(prs[ed] - prs[st])
        sum_tr = float(trs[ed] - trs[st])
        sum_pr2 = float(pr2s[ed] - pr2s[st])
        sum_tr2 = float(tr2s[ed] - tr2s[st])
        sum_prt = float(prts[ed] - prts[st])
        rank_ic = _corr_from_sums(sum_pr, sum_tr, sum_pr2, sum_tr2, sum_prt, w)

        # Emit pooled-window diagnostics on the globally sorted label axis.
        rows.append(
            {
                "window_start": int(st),
                "window_end": int(ed - 1),
                "group_center_rank": float(center_rank),
                "label_left": float(label[st]),
                "label_center": float(label[min(st + w // 2, n - 1)]),
                "label_right": float(label[ed - 1]),
                "mean_ic": float(ic),
                "std_ic": float("nan"),
                "mean_rank_ic": float(rank_ic),
                "std_rank_ic": float("nan"),
                "count": int(w),
            }
        )

    # Return windows in percentile order so plotting follows the sorted label axis.
    out = pd.DataFrame(rows).sort_values(["group_center_rank", "window_start"], kind="stable").reset_index(drop=True)
    return out


def _rolling_group_ic_bins(
    df: pd.DataFrame,
    label_col: str,
    window_size: int,
    step_size: int,
) -> dict[float, dict[str, float]]:
    """Compute rolling-window IC bins aggregated by center-rank."""
    # Define helpers to compute Pearson correlation from windowed prefix sums.
    def _corr_from_sums(sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float, w: int) -> float:
        """Compute Pearson correlation from raw sums on a fixed window."""
        # Compute covariance and variances with stable float math.
        wf = float(w)
        cov = float(sum_xy - (sum_x * sum_y) / wf)
        var_x = float(sum_x2 - (sum_x * sum_x) / wf)
        var_y = float(sum_y2 - (sum_y * sum_y) / wf)
        if not (np.isfinite(cov) and np.isfinite(var_x) and np.isfinite(var_y)):
            return float("nan")
        if float(var_x) <= 0.0 or float(var_y) <= 0.0:
            return float("nan")
        return float(cov / float(np.sqrt(var_x * var_y)))

    # Accumulate aggregated moments per rank bin to avoid materializing millions of window rows.
    bins: dict[float, dict[str, float]] = {}
    w = int(window_size)
    step = int(step_size)
    for (_d, _t), g in df.groupby(["date", "time"], sort=True):
        # Sort by the grouping label and drop missing rows.
        gg = g[["prediction", "target", label_col]].dropna(subset=["prediction", "target", label_col]).sort_values(label_col, kind="stable")
        n = int(gg.shape[0])
        if int(n) < int(w):
            continue

        # Extract prediction/target arrays and cast once for stable prefix-sum accumulation.
        pred = gg["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = gg["target"].to_numpy(dtype=np.float64, copy=False)

        # Build prefix sums for Pearson IC computation on raw values.
        ps = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pred, dtype=np.float64)])
        ts = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tgt, dtype=np.float64)])
        p2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pred * pred, dtype=np.float64)])
        t2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tgt * tgt, dtype=np.float64)])
        pts = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pred * tgt, dtype=np.float64)])

        # Build global ranks once and reuse for the rolling-window rank IC approximation.
        pr = stats.rankdata(pred, method="average").astype(np.float64, copy=False)
        tr = stats.rankdata(tgt, method="average").astype(np.float64, copy=False)
        prs = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pr, dtype=np.float64)])
        trs = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tr, dtype=np.float64)])
        pr2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pr * pr, dtype=np.float64)])
        tr2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tr * tr, dtype=np.float64)])
        prts = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pr * tr, dtype=np.float64)])

        # Slide windows along sorted rows and accumulate moments per rank bin.
        for st in range(0, int(n - w + 1), int(step)):
            # Compute the window's center rank percentile for cross-date binning.
            center = float(st + w * 0.5)
            center_rank = float(center / float(n))
            rank_bin = float(round(center_rank, 3))

            # Compute Pearson IC from prefix sums on raw prediction/target.
            ed = int(st + w)
            sum_p = float(ps[ed] - ps[st])
            sum_t = float(ts[ed] - ts[st])
            sum_p2 = float(p2s[ed] - p2s[st])
            sum_t2 = float(t2s[ed] - t2s[st])
            sum_pt = float(pts[ed] - pts[st])
            ic = _corr_from_sums(sum_p, sum_t, sum_p2, sum_t2, sum_pt, w)

            # Compute rank IC from prefix sums on global ranks restricted to the same window.
            sum_pr = float(prs[ed] - prs[st])
            sum_tr = float(trs[ed] - trs[st])
            sum_pr2 = float(pr2s[ed] - pr2s[st])
            sum_tr2 = float(tr2s[ed] - tr2s[st])
            sum_prt = float(prts[ed] - prts[st])
            rank_ic = _corr_from_sums(sum_pr, sum_tr, sum_pr2, sum_tr2, sum_prt, w)

            # Initialize accumulator buckets lazily per observed rank bin.
            if rank_bin not in bins:
                bins[rank_bin] = {
                    "sum_center_rank": 0.0,
                    "sum_ic": 0.0,
                    "sum_ic2": 0.0,
                    "n_ic": 0.0,
                    "sum_rank_ic": 0.0,
                    "sum_rank_ic2": 0.0,
                    "n_rank_ic": 0.0,
                    "count": 0.0,
                }
            acc = bins[rank_bin]

            # Accumulate first and second moments for mean/std computation.
            acc["sum_center_rank"] += float(center_rank)
            if np.isfinite(ic):
                acc["sum_ic"] += float(ic)
                acc["sum_ic2"] += float(ic * ic)
                acc["n_ic"] += 1.0
            if np.isfinite(rank_ic):
                acc["sum_rank_ic"] += float(rank_ic)
                acc["sum_rank_ic2"] += float(rank_ic * rank_ic)
                acc["n_rank_ic"] += 1.0
            acc["count"] += 1.0

    return bins


def _plot_group_curve(df: pd.DataFrame, title: str, out_png: Path) -> None:
    """Plot mean IC curves against group_center_rank."""
    # Render a simple 2-line plot for Pearson and Rank IC.
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df["group_center_rank"].to_numpy(dtype=float), df["mean_ic"].to_numpy(dtype=float), label="Pearson IC", linewidth=1.8)
    ax.plot(df["group_center_rank"].to_numpy(dtype=float), df["mean_rank_ic"].to_numpy(dtype=float), label="Rank IC", linewidth=1.8)
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("group_center_rank (percentile)")
    ax.set_ylabel("mean IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _empty_group_schema() -> pd.DataFrame:
    """Return an empty rolling-group IC dataframe with a stable schema."""
    # Define a stable column order so downstream report rendering is predictable.
    cols = ["group_center_rank", "mean_ic", "std_ic", "mean_rank_ic", "std_rank_ic", "count"]
    out = pd.DataFrame({c: pd.Series([], dtype=float) for c in cols})
    out["count"] = out["count"].astype(int)
    return out


def _plot_empty_group_curve(title: str, out_png: Path) -> None:
    """Write a placeholder plot when rolling groups are empty."""
    # Render a simple figure with an explanatory text to avoid downstream missing files.
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.text(0.5, 0.5, "Empty rolling groups: n < window_size", ha="center", va="center", fontsize=12)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def volatility_rolling_ic(pred_df: pd.DataFrame, config: EvalConfig, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute volatility rolling-window IC and persist CSV/plot."""
    # Compute rolling-window IC assuming labels are already attached.
    agg = _rolling_group_ic(pred_df, "volatility_label", int(config.window_size), int(config.step_size))
    out_yaml = Path(out_csv).with_suffix(".yaml")
    if agg.shape[0] == 0:
        warnings.warn("Empty volatility rolling IC: valid stock count < window_size.", RuntimeWarning)
        agg = _empty_group_schema()
        agg.to_csv(out_csv, index=False)
        _plot_empty_group_curve("Volatility rolling IC", out_png)
        _write_group_curve_summary(agg, out_yaml, "volatility")
        return agg

    # Persist the aggregated curve and emit the plot.
    agg.to_csv(out_csv, index=False)
    _plot_group_curve(agg, "Volatility rolling IC", out_png)
    _write_group_curve_summary(agg, out_yaml, "volatility")
    return agg


def volatility_rolling_ic_parallel(pred_df: pd.DataFrame, config: EvalConfig, out_csv: Path, out_png: Path, workers: int) -> pd.DataFrame:
    """Compute volatility rolling-window IC via per-date multiprocessing and persist CSV/plot."""
    # Compute aggregated rolling-window IC using per-date label joins in worker processes.
    agg = rolling_group_ic_parallel(pred_df, config, "volatility_label", int(workers))
    out_yaml = Path(out_csv).with_suffix(".yaml")
    if agg.shape[0] == 0:
        warnings.warn("Empty volatility rolling IC: valid stock count < window_size.", RuntimeWarning)
        agg = _empty_group_schema()
        agg.to_csv(out_csv, index=False)
        _plot_empty_group_curve("Volatility rolling IC", out_png)
        _write_group_curve_summary(agg, out_yaml, "volatility")
        return agg

    # Persist the aggregated curve and emit the plot.
    agg.to_csv(out_csv, index=False)
    _plot_group_curve(agg, "Volatility rolling IC", out_png)
    _write_group_curve_summary(agg, out_yaml, "volatility")
    return agg


def price_rolling_ic(pred_df: pd.DataFrame, config: EvalConfig, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute price rolling-window IC and persist CSV/plot."""
    # Compute rolling-window IC assuming labels are already attached.
    agg = _rolling_group_ic(pred_df, "price_label", int(config.window_size), int(config.step_size))
    out_yaml = Path(out_csv).with_suffix(".yaml")
    if agg.shape[0] == 0:
        warnings.warn("Empty price rolling IC: valid stock count < window_size.", RuntimeWarning)
        agg = _empty_group_schema()
        agg.to_csv(out_csv, index=False)
        _plot_empty_group_curve("Price rolling IC", out_png)
        _write_group_curve_summary(agg, out_yaml, "price")
        return agg

    # Persist the aggregated curve and emit the plot.
    agg.to_csv(out_csv, index=False)
    _plot_group_curve(agg, "Price rolling IC", out_png)
    _write_group_curve_summary(agg, out_yaml, "price")
    return agg


def price_rolling_ic_parallel(pred_df: pd.DataFrame, config: EvalConfig, out_csv: Path, out_png: Path, workers: int) -> pd.DataFrame:
    """Compute price rolling-window IC via per-date multiprocessing and persist CSV/plot."""
    # Compute aggregated rolling-window IC using per-date label joins in worker processes.
    agg = rolling_group_ic_parallel(pred_df, config, "price_label", int(workers))
    out_yaml = Path(out_csv).with_suffix(".yaml")
    if agg.shape[0] == 0:
        warnings.warn("Empty price rolling IC: valid stock count < window_size.", RuntimeWarning)
        agg = _empty_group_schema()
        agg.to_csv(out_csv, index=False)
        _plot_empty_group_curve("Price rolling IC", out_png)
        _write_group_curve_summary(agg, out_yaml, "price")
        return agg

    # Persist the aggregated curve and emit the plot.
    agg.to_csv(out_csv, index=False)
    _plot_group_curve(agg, "Price rolling IC", out_png)
    _write_group_curve_summary(agg, out_yaml, "price")
    return agg


def score_ret_rank_plot(pred_df: pd.DataFrame, out_png: Path) -> pd.DataFrame:
    """Plot predicted-score rank bins against realized target return and win-rate."""
    # Build rank-percentile bins by prediction within each timestamp cross-section.
    tmp = pred_df[["date", "time", "prediction", "target"]].dropna(subset=["prediction", "target"]).copy()
    tmp["pred_rank_pct"] = tmp.groupby(["date", "time"], sort=False)["prediction"].rank(method="average", pct=True)

    # Bin rank into deciles and aggregate realized return and sign win rate.
    tmp["decile"] = np.minimum((tmp["pred_rank_pct"] * 10.0).astype(int), 9)
    agg = tmp.groupby("decile", sort=True).agg(
        mean_target=("target", "mean"),
        win_rate=("target", lambda x: float((np.asarray(x, dtype=float) > 0.0).mean())),
        count=("target", "size"),
    )
    agg = agg.reset_index()

    # Plot mean return and win-rate on dual axes.
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    xs = agg["decile"].to_numpy(dtype=int)
    ax1.plot(xs, agg["mean_target"].to_numpy(dtype=float), color="#4c72b0", linewidth=2.0, label="mean target")
    ax2.plot(xs, agg["win_rate"].to_numpy(dtype=float), color="#dd8452", linewidth=2.0, label="win rate")
    ax1.set_xlabel("prediction rank decile (0=low, 9=high)")
    ax1.set_ylabel("mean target")
    ax2.set_ylabel("win rate (target>0)")
    ax1.set_title("Prediction vs target: rank curve")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return agg
