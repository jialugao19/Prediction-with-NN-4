"""Prepare NPZ datasets from /data/ashare/market/stock1m and emit data-clean artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing as mp
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_STOCK_FEATURES = [
    "ret_1m",
    "ret_5m",
    "ret_30m",
    "ret_60m",
    "hl",
    "oc",
    "log_vol",
    "log_amount",
    "vol_30m",
    "vol_60m",
    "log_vol_30m_mean",
    "log_amount_30m_mean",
]
_TIME_FEATURES = ["minute_norm", "session_id", "session_minute_norm"]


@dataclass(frozen=True)
class DataPrepConfig:
    """Define IO paths and sampling knobs for data preparation."""

    stock1m_dir: Path
    out_dir: Path
    start_trade_date: int
    end_trade_date: int
    train_days: int
    val_days: int
    test_days: int
    seed: int
    horizon_minutes: int
    sample_stocks_per_minute: int
    workers: int


def list_trade_dates(config: DataPrepConfig) -> list[int]:
    """List trade dates in [start,end] available under stock1m_dir."""
    # Scan year subfolders for feather files and filter by date range.
    dates: list[int] = []
    for year_dir in sorted(Path(config.stock1m_dir).glob("*")):
        if not year_dir.is_dir():
            continue
        for p in year_dir.glob("*.feather"):
            d = int(p.stem)
            if int(d) < int(config.start_trade_date) or int(d) > int(config.end_trade_date):
                continue
            dates.append(int(d))

    # Sort and return full date list; split policy should decide how to consume them.
    dates = sorted(set(dates))
    return dates


def _read_stock1m_day(trade_date: int, config: DataPrepConfig) -> pd.DataFrame:
    """Load one trade_date minute-bar panel with required columns."""
    # Resolve file path by year folder convention.
    year = int(trade_date) // 10000
    path = Path(config.stock1m_dir) / str(year) / f"{int(trade_date)}.feather"

    # Read minimal columns to compute features and labels.
    cols = ["StockCode", "DateTime", "Open", "Close", "High", "Low", "Vol", "Amount", "Date", "MinuteIndex"]
    df = pd.read_feather(path, columns=cols)
    df = df.sort_values(["StockCode", "DateTime"], kind="stable").reset_index(drop=True)
    return df


def _add_features_and_label(df: pd.DataFrame, config: DataPrepConfig) -> pd.DataFrame:
    """Compute non-rank feature transforms and a forward return label."""
    # Drop invalid price rows so log/ratio transforms are well-defined.
    df = df.copy()
    close = df["Close"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    vol = df["Vol"].to_numpy(dtype=float)
    amount = df["Amount"].to_numpy(dtype=float)
    m_price = np.isfinite(close) & np.isfinite(open_) & np.isfinite(high) & np.isfinite(low)
    m_price &= np.isfinite(vol) & np.isfinite(amount)
    m_price &= (close > 0.0) & (open_ > 0.0) & (high > 0.0) & (low > 0.0)
    m_price &= (vol >= 0.0) & (amount >= 0.0)
    df = df.loc[m_price].reset_index(drop=True)

    # Compute log prices to define return-like features and labels.
    df["log_close"] = np.log(df["Close"].to_numpy(dtype=float))

    # Compute within-stock intraday returns as simple derived features.
    g = df.groupby("StockCode", sort=False)
    df["ret_1m"] = g["log_close"].diff(1)
    df["ret_5m"] = g["log_close"].diff(5)
    df["ret_30m"] = g["log_close"].diff(30)
    df["ret_60m"] = g["log_close"].diff(60)

    # Compute price-shape and activity features without rank transforms.
    df["hl"] = np.log(df["High"].to_numpy(dtype=float) / df["Low"].to_numpy(dtype=float))
    df["oc"] = np.log(df["Close"].to_numpy(dtype=float) / df["Open"].to_numpy(dtype=float))
    df["log_vol"] = np.log(df["Vol"].to_numpy(dtype=float) + 1.0)
    df["log_amount"] = np.log(df["Amount"].to_numpy(dtype=float) + 1.0)
    df["minute_norm"] = df["MinuteIndex"].to_numpy(dtype=float) / 240.0
    df["session_id"] = (df["MinuteIndex"].to_numpy(dtype=int) >= 120).astype(float)
    df["session_minute_norm"] = (df["MinuteIndex"].to_numpy(dtype=int) % 120).astype(float) / 120.0

    # Compute longer-window volatility and activity summaries per stock.
    df["vol_30m"] = g["ret_1m"].rolling(window=30, min_periods=2).std(ddof=0).reset_index(level=0, drop=True)
    df["vol_60m"] = g["ret_1m"].rolling(window=60, min_periods=2).std(ddof=0).reset_index(level=0, drop=True)
    df["log_vol_30m_mean"] = g["log_vol"].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)
    df["log_amount_30m_mean"] = g["log_amount"].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)

    # Compute forward log return as the supervised label.
    h = int(config.horizon_minutes)
    df["label_ret"] = g["log_close"].shift(-h) - df["log_close"]

    # Keep only rows with finite features and label.
    feat_cols = [
        "ret_1m",
        "ret_5m",
        "ret_30m",
        "ret_60m",
        "hl",
        "oc",
        "log_vol",
        "log_amount",
        "vol_30m",
        "vol_60m",
        "log_vol_30m_mean",
        "log_amount_30m_mean",
        "minute_norm",
        "session_id",
        "session_minute_norm",
    ]
    need = feat_cols + ["label_ret"]
    m = np.ones((df.shape[0],), dtype=bool)
    for c in need:
        v = df[c].to_numpy(dtype=float)
        m &= np.isfinite(v)
    df = df.loc[m, ["StockCode", "DateTime", "Date", "MinuteIndex"] + feat_cols + ["label_ret"]].reset_index(drop=True)
    return df


def _sample_per_minute(df: pd.DataFrame, config: DataPrepConfig) -> pd.DataFrame:
    """Sample a fixed number of stocks per minute to control dataset size."""
    # Short-circuit when no sampling limit is requested.
    k = int(config.sample_stocks_per_minute)
    if k <= 0:
        return df

    # Sample codes per minute with a deterministic RNG seed per trade date.
    trade_date = int(df["Date"].iloc[0])
    seed = (int(config.seed) * 1_000_003 + trade_date) % (2**32)
    rng = np.random.default_rng(int(seed))
    rows: list[pd.DataFrame] = []
    for _minute, g in df.groupby("MinuteIndex", sort=True):
        # Use stable sampling by taking a random subset of row indices.
        if int(g.shape[0]) <= k:
            rows.append(g)
            continue
        pick = rng.choice(g.index.to_numpy(), size=k, replace=False)
        rows.append(g.loc[pick])
    out = pd.concat(rows, axis=0).sort_values(["StockCode", "DateTime"], kind="stable").reset_index(drop=True)
    return out


def _cross_sectional_gaussianize(df: pd.DataFrame, stock_features: list[str]) -> pd.DataFrame:
    """Apply robust cross-sectional Gaussianization to stock-varying features."""
    # Compute per-minute percentile ranks and map them onto a standard normal.
    keys = ["Date", "MinuteIndex"]
    out = df.copy()
    for name in list(stock_features):
        # Build clipped percentile ranks to avoid +/-inf from norm.ppf.
        pct = out.groupby(keys, sort=False)[name].rank(method="average", pct=True)
        pct = pct.clip(lower=1e-3, upper=1.0 - 1e-3)
        out[f"{name}_cs"] = stats.norm.ppf(pct.to_numpy(dtype=float, copy=False)).astype(np.float32, copy=False)
    return out


def _date_time_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DateTime into qmodel-style int date/time columns."""
    # Compute yymmdd date int to match qmodel merge_date_time_dataframe convention.
    dt = pd.to_datetime(df["DateTime"])
    date_yymmdd = (dt.dt.year % 100) * 10000 + dt.dt.month * 100 + dt.dt.day
    time_hhmmss = dt.dt.hour * 10000 + dt.dt.minute * 100 + dt.dt.second

    # Attach columns and return a trimmed dataframe.
    out = df.copy()
    out["date_int"] = date_yymmdd.astype(np.int64)
    out["time_int"] = time_hhmmss.astype(np.int64)
    return out


def _compute_norm_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std for standardization."""
    # Compute mean and std in float64 for numerical stability.
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64, ddof=0)
    if bool((std <= 0.0).any()):
        raise RuntimeError("Feature std contains non-positive values.")
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply per-feature standardization."""
    # Standardize values into approximately N(0,1)-like marginals.
    return ((x - mean) / std).astype(np.float32, copy=False)


def _write_feature_distribution_artifacts(x: np.ndarray, feature_names: list[str], out_dir: Path) -> pd.DataFrame:
    """Compute distribution stats and write per-feature histogram plots."""
    # Ensure output directory exists before writing artifacts.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute mean/std/skew/kurtosis and store in a table.
    rows: list[dict[str, object]] = []
    for j, name in enumerate(list(feature_names)):
        # Compute summary moments on finite values only.
        v = x[:, j].astype(np.float64, copy=False)
        v = v[np.isfinite(v)]
        rows.append(
            {
                "feature": str(name),
                "mean": float(v.mean()),
                "std": float(v.std(ddof=0)),
                "skew": float(stats.skew(v, bias=False)),
                "kurtosis": float(stats.kurtosis(v, fisher=True, bias=False)),
                "count": int(v.shape[0]),
            }
        )

        # Plot histogram with an overlaid standard normal curve.
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(v, bins=80, density=True, alpha=0.6, color="#4c72b0")
        grid = np.linspace(-5.0, 5.0, 400)
        ax.plot(grid, stats.norm.pdf(grid, loc=0.0, scale=1.0), color="#dd8452", linewidth=2)
        ax.set_title(f"Feature dist: {name}")
        ax.set_xlabel("standardized value")
        ax.set_ylabel("density")
        fig.tight_layout()
        fig.savefig(out_dir / f"dist_{name}.png", dpi=160)
        plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "feature_moments.csv", index=False)
    return table


def _write_feature_distribution_artifacts_from_bin(
    x_path: Path, train_rows: int, feature_dim: int, feature_names: list[str], out_dir: Path
) -> pd.DataFrame:
    """Compute distribution stats from a raw float32 matrix and write histogram plots."""
    # Ensure output directory exists before writing artifacts.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Memory-map the raw feature matrix so multi-year datasets do not require RAM materialization.
    x = np.memmap(Path(x_path), mode="r", dtype=np.float32, shape=(int(train_rows), int(feature_dim)))

    # Predefine histogram bins so we can update counts in a streaming pass.
    edges = np.linspace(-5.0, 5.0, 81, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_counts = np.zeros((int(feature_dim), int(edges.shape[0] - 1)), dtype=np.int64)

    # Accumulate raw moments (sum, sum2, sum3, sum4) for exact mean/std/skew/kurtosis.
    count = np.zeros((int(feature_dim),), dtype=np.int64)
    s1 = np.zeros((int(feature_dim),), dtype=np.float64)
    s2 = np.zeros((int(feature_dim),), dtype=np.float64)
    s3 = np.zeros((int(feature_dim),), dtype=np.float64)
    s4 = np.zeros((int(feature_dim),), dtype=np.float64)

    # Stream over the memmap in chunks to bound peak memory while keeping IO sequential.
    chunk_rows = 1_000_000
    for st in range(0, int(train_rows), int(chunk_rows)):
        # Slice a contiguous block and upcast to float64 for stable moment accumulation.
        ed = min(int(train_rows), int(st + chunk_rows))
        blk = np.asarray(x[int(st) : int(ed)], dtype=np.float64)

        # Update moments and histograms feature-by-feature to avoid large intermediates.
        for j in range(int(feature_dim)):
            # Filter finite values to match the original artifact semantics.
            v = blk[:, int(j)]
            v = v[np.isfinite(v)]
            if int(v.shape[0]) == 0:
                continue

            # Update raw-moment accumulators for this feature.
            n = int(v.shape[0])
            count[int(j)] += int(n)
            s1[int(j)] += float(v.sum(dtype=np.float64))
            s2[int(j)] += float((v * v).sum(dtype=np.float64))
            s3[int(j)] += float((v * v * v).sum(dtype=np.float64))
            s4[int(j)] += float((v * v * v * v).sum(dtype=np.float64))

            # Update histogram counts using the fixed edges.
            idx = np.searchsorted(edges, v, side="right") - 1
            idx = np.clip(idx, 0, int(edges.shape[0] - 2))
            hist_counts[int(j)] += np.bincount(idx.astype(np.int64, copy=False), minlength=int(edges.shape[0] - 1)).astype(
                np.int64, copy=False
            )

    # Convert raw moments into mean/std/skew/kurtosis and render plots per feature.
    rows: list[dict[str, object]] = []
    for j, name in enumerate(list(feature_names)):
        # Compute derived moments using aggregated raw sums.
        n = float(count[int(j)])
        mean = float(s1[int(j)] / n)
        m2 = float(s2[int(j)] / n)
        m3 = float(s3[int(j)] / n)
        m4 = float(s4[int(j)] / n)
        mu2 = float(m2 - mean * mean)
        mu3 = float(m3 - 3.0 * m2 * mean + 2.0 * mean * mean * mean)
        mu4 = float(m4 - 4.0 * m3 * mean + 6.0 * m2 * mean * mean - 3.0 * mean**4)
        std = float(np.sqrt(mu2))
        skew = float(mu3 / (std**3))
        kurtosis = float(mu4 / (mu2 * mu2) - 3.0)

        # Append per-feature moment summary for downstream report linkage.
        rows.append(
            {
                "feature": str(name),
                "mean": float(mean),
                "std": float(std),
                "skew": float(skew),
                "kurtosis": float(kurtosis),
                "count": int(count[int(j)]),
            }
        )

        # Plot histogram with an overlaid standard normal curve.
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        dens = hist_counts[int(j)].astype(np.float64) / float(max(int(count[int(j)]), 1))
        dens = dens / float((edges[1] - edges[0]))
        ax.bar(centers, dens, width=float(edges[1] - edges[0]), alpha=0.6, color="#4c72b0", align="center")
        grid = np.linspace(-5.0, 5.0, 400)
        ax.plot(grid, stats.norm.pdf(grid, loc=0.0, scale=1.0), color="#dd8452", linewidth=2)
        ax.set_title(f"Feature dist: {name}")
        ax.set_xlabel("standardized value")
        ax.set_ylabel("density")
        fig.tight_layout()
        fig.savefig(out_dir / f"dist_{name}.png", dpi=160)
        plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "feature_moments.csv", index=False)
    return table


def _build_single_day_table(trade_date: int, config: DataPrepConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build one day dataframe for NPZ export and return a per-day audit dict."""
    # Track raw row count to quantify feature/label missing filters.
    day_raw = _read_stock1m_day(int(trade_date), config)
    raw_rows = int(day_raw.shape[0])

    # Compute raw features/label and then sample within each minute.
    day_feat = _add_features_and_label(day_raw, config)
    kept_rows = int(day_feat.shape[0])
    day_samp = _sample_per_minute(day_feat, config)
    sampled_rows = int(day_samp.shape[0])

    # Apply cross-sectional normalization on stock-varying features only.
    day_norm = _cross_sectional_gaussianize(day_samp, list(_STOCK_FEATURES))

    # Attach date/time integer columns and keep only schema columns.
    day_norm = _date_time_int_columns(day_norm)
    keep_cols = (
        ["StockCode", "DateTime", "Date", "MinuteIndex"]
        + [f"{c}_cs" for c in list(_STOCK_FEATURES)]
        + list(_TIME_FEATURES)
        + ["label_ret", "date_int", "time_int"]
    )
    day_out = day_norm.loc[:, keep_cols].reset_index(drop=True)

    # Assemble an explicit per-day audit record for cross-period aggregation.
    audit = {
        "trade_date": int(trade_date),
        "raw_rows": int(raw_rows),
        "kept_rows": int(kept_rows),
        "sampled_rows": int(sampled_rows),
        "out_rows": int(day_out.shape[0]),
    }
    return day_out, audit


def _build_single_day_table_task(args: tuple[int, DataPrepConfig]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Unpack starmap-style args so Pool.imap can stream day results."""
    # Unpack task tuple and delegate into the core per-day builder.
    trade_date, config = args
    return _build_single_day_table(int(trade_date), config)


def prepare_npz_splits(config: DataPrepConfig) -> dict[str, object]:
    """Prepare train/val/test NPZ datasets and data-clean artifacts."""
    # Resolve IO paths and short-circuit when a completed meta.yaml already exists.
    npz_dir = Path(config.out_dir) / "npz"
    meta_path = npz_dir / "meta.yaml"
    npz_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    if meta_path.exists():
        # Load the persisted metadata so repeated invocations can resume at training without rebuilding.
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        feature_names = list(meta["feature_names"])
        storage = dict(meta["storage"])
        groups = dict(storage["groups"])
        stats_dir = Path(config.out_dir) / "data_clean"
        moment_table = pd.read_csv(stats_dir / "feature_moments.csv")
        return {
            "done": True,
            "feature_names": feature_names,
            "train_rows": int(groups["train"]["rows"]),
            "val_rows": int(groups["val"]["rows"]),
            "test_rows": int(groups["test"]["rows"]),
            "predict_rows": int(groups["predict"]["rows"]),
            "moment_table": moment_table,
            "meta_path": meta_path,
            "audit": dict(meta["audit"]),
            "audit_rates": dict(meta["audit_rates"]),
        }

    # Resolve trade dates and validate that splits are feasible.
    dates = list_trade_dates(config)
    n_train = int(config.train_days)
    n_val = int(config.val_days)
    n_test = int(config.test_days)
    need = n_train + n_val + n_test
    if len(dates) < need:
        raise RuntimeError(f"Not enough trade dates: got={len(dates)} need={need}")
    train_dates = dates[:n_train]
    val_dates = dates[n_train : n_train + n_val]
    test_dates = dates[n_train + n_val : n_train + n_val + n_test]

    # Define the stable feature ordering early so array materialization is consistent.
    stock_cs_features = [f"{c}_cs" for c in list(_STOCK_FEATURES)]
    time_features = list(_TIME_FEATURES)
    feature_names = list(stock_cs_features) + list(time_features)

    # Create a shared worker pool once so per-day processing runs in parallel.
    workers = int(config.workers)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=int(workers))

    # Load panels, compute features/labels, and sample within each day using multiprocessing map.
    def _build_for_dates(
        tag: str,
        date_list: list[int],
        *,
        x_f,
        y_f,
        meta_f,
        predict_x_f,
        predict_y_f,
        predict_meta_f,
    ) -> tuple[int, list[dict[str, object]], dict[str, object]]:
        """Stream one split into raw binary arrays and return (rows, daily_audits, split_audit)."""
        # Stream per-day processing inside worker processes while writing arrays sequentially in the main process.
        t0 = time.time()
        tasks = [(int(d), config) for d in list(date_list)]
        it = pool.imap(_build_single_day_table_task, tasks, chunksize=1)

        # Accumulate audits and row counts without materializing the full split in memory.
        rows = 0
        audits: list[dict[str, object]] = []
        for i, (day_df, audit) in enumerate(it, start=1):
            # Convert the per-day dataframe into arrays and append them to raw binary files.
            x_day = day_df[feature_names].to_numpy(dtype=np.float32, copy=False)
            y_day = day_df[["label_ret"]].to_numpy(dtype=np.float32, copy=False)
            meta_day = day_df[["StockCode", "date_int", "time_int"]].to_numpy(dtype=np.int64, copy=False)
            x_day.tofile(x_f)
            y_day.tofile(y_f)
            meta_day.tofile(meta_f)

            # Append the same arrays into the predict split so long-horizon evaluation reuses the prep pass.
            x_day.tofile(predict_x_f)
            y_day.tofile(predict_y_f)
            meta_day.tofile(predict_meta_f)

            # Track split-level counters for audit aggregation.
            rows += int(x_day.shape[0])
            audits.append(dict(audit))

            # Print coarse-grained progress so long multi-year runs are observable in logs.
            if int(i) == 1 or int(i) % 20 == 0 or int(i) == int(len(date_list)):
                print(f"[data_prep] tag={tag} day={i}/{len(date_list)} rows={rows}", flush=True)
        t1 = time.time()

        # Build split-level audit summary for metadata and report rendering.
        raw_rows = int(sum(int(a["raw_rows"]) for a in audits))
        kept_rows = int(sum(int(a["kept_rows"]) for a in audits))
        sampled_rows = int(sum(int(a["sampled_rows"]) for a in audits))
        split_audit = {
            "tag": str(tag),
            "workers": int(workers),
            "days": int(len(date_list)),
            "raw_rows": int(raw_rows),
            "kept_rows": int(kept_rows),
            "sampled_rows": int(sampled_rows),
            "out_rows": int(rows),
            "elapsed_seconds": float(t1 - t0),
        }
        return int(rows), audits, split_audit

    # Load or initialize incremental progress tracking so repeated calls can finish multi-year prep.
    progress_path = npz_dir / "progress.yaml"
    if progress_path.exists():
        # Load prior progress to continue appending to the binary arrays.
        progress = yaml.safe_load(progress_path.read_text(encoding="utf-8"))
    else:
        # Initialize progress from scratch so the first invocation truncates any stale partial outputs.
        progress = {"stage": "train", "index": 0, "elapsed_seconds": {"train": 0.0, "val": 0.0, "test": 0.0}}

    # Decide which split to advance in this invocation and slice a bounded batch of dates.
    stage = str(progress["stage"])
    stage_dates = {"train": train_dates, "val": val_dates, "test": test_dates}[stage]
    start_i = int(progress["index"])
    days_per_call = 20
    todo = list(stage_dates[int(start_i) : int(start_i + days_per_call)])
    if len(todo) == 0:
        raise RuntimeError(f"progress points past end of stage: stage={stage} index={start_i} total={len(stage_dates)}")

    # Open raw array files in append-or-truncate mode depending on the stage progress.
    stage_paths = {
        "train": (npz_dir / "train_x.f32", npz_dir / "train_y.f32", npz_dir / "train_meta.i64"),
        "val": (npz_dir / "val_x.f32", npz_dir / "val_y.f32", npz_dir / "val_meta.i64"),
        "test": (npz_dir / "test_x.f32", npz_dir / "test_y.f32", npz_dir / "test_meta.i64"),
    }
    x_path, y_path, meta_bin_path = stage_paths[stage]
    mode = "ab" if int(start_i) > 0 else "wb"
    predict_x_path = npz_dir / "predict_x.f32"
    predict_y_path = npz_dir / "predict_y.f32"
    predict_meta_path = npz_dir / "predict_meta.i64"
    predict_mode = "ab" if predict_x_path.exists() else "wb"

    # Run one chunk and persist both binary arrays and daily audits.
    try:
        with open(predict_x_path, predict_mode) as px, open(predict_y_path, predict_mode) as py, open(predict_meta_path, predict_mode) as pm:
            with open(x_path, mode) as xf, open(y_path, mode) as yf, open(meta_bin_path, mode) as mf:
                _rows, audits, split_audit = _build_for_dates(
                    stage,
                    todo,
                    x_f=xf,
                    y_f=yf,
                    meta_f=mf,
                    predict_x_f=px,
                    predict_y_f=py,
                    predict_meta_f=pm,
                )
    finally:
        pool.close()
        pool.join()

    # Append daily audits into a YAML list so final meta.yaml can embed the full audit table.
    audit_path = npz_dir / f"audit_daily_{stage}.yaml"
    if audit_path.exists():
        daily = list(yaml.safe_load(audit_path.read_text(encoding="utf-8")))
    else:
        daily = []
    daily.extend(list(audits))
    audit_path.write_text(yaml.safe_dump(daily, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Advance progress and persist it before potentially returning early.
    progress["index"] = int(start_i + len(todo))
    progress["elapsed_seconds"][stage] = float(progress["elapsed_seconds"][stage]) + float(split_audit["elapsed_seconds"])
    if int(progress["index"]) >= int(len(stage_dates)):
        if stage == "train":
            progress["stage"] = "val"
        elif stage == "val":
            progress["stage"] = "test"
        else:
            progress["stage"] = "done"
        progress["index"] = 0
    progress_path.write_text(yaml.safe_dump(progress, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Return early when data prep is still incomplete so the outer pipeline can be re-invoked.
    if str(progress["stage"]) != "done":
        return {
            "done": False,
            "feature_names": feature_names,
            "meta_path": meta_path,
            "stage": str(progress["stage"]),
            "index": int(progress["index"]),
        }

    # Load full daily audit tables now that all splits are complete.
    train_daily_audits = list(yaml.safe_load((npz_dir / "audit_daily_train.yaml").read_text(encoding="utf-8")))
    val_daily_audits = list(yaml.safe_load((npz_dir / "audit_daily_val.yaml").read_text(encoding="utf-8")))
    test_daily_audits = list(yaml.safe_load((npz_dir / "audit_daily_test.yaml").read_text(encoding="utf-8")))

    # Compute split-level audit summaries from the daily audit tables.
    def _sum_audits(tag: str, xs: list[dict[str, object]], elapsed: float) -> dict[str, object]:
        """Aggregate per-day audit dicts into a split-level counter summary."""
        # Sum counters across all days and attach the measured elapsed seconds for the split.
        raw_rows = int(sum(int(a["raw_rows"]) for a in xs))
        kept_rows = int(sum(int(a["kept_rows"]) for a in xs))
        sampled_rows = int(sum(int(a["sampled_rows"]) for a in xs))
        out_rows = int(sum(int(a["out_rows"]) for a in xs))
        return {
            "tag": str(tag),
            "workers": int(workers),
            "days": int(len(xs)),
            "raw_rows": int(raw_rows),
            "kept_rows": int(kept_rows),
            "sampled_rows": int(sampled_rows),
            "out_rows": int(out_rows),
            "elapsed_seconds": float(elapsed),
        }

    train_audit = _sum_audits("train", train_daily_audits, float(progress["elapsed_seconds"]["train"]))
    val_audit = _sum_audits("val", val_daily_audits, float(progress["elapsed_seconds"]["val"]))
    test_audit = _sum_audits("test", test_daily_audits, float(progress["elapsed_seconds"]["test"]))
    predict_audit = {
        "tag": "predict",
        "workers": int(workers),
        "days": int(train_audit["days"]) + int(val_audit["days"]) + int(test_audit["days"]),
        "raw_rows": int(train_audit["raw_rows"]) + int(val_audit["raw_rows"]) + int(test_audit["raw_rows"]),
        "kept_rows": int(train_audit["kept_rows"]) + int(val_audit["kept_rows"]) + int(test_audit["kept_rows"]),
        "sampled_rows": int(train_audit["sampled_rows"]) + int(val_audit["sampled_rows"]) + int(test_audit["sampled_rows"]),
        "out_rows": int(train_audit["out_rows"]) + int(val_audit["out_rows"]) + int(test_audit["out_rows"]),
        "elapsed_seconds": float(train_audit["elapsed_seconds"]) + float(val_audit["elapsed_seconds"]) + float(test_audit["elapsed_seconds"]),
    }

    # Compute data-clean distribution artifacts from the full training matrix via memmap scanning.
    stats_dir = Path(config.out_dir) / "data_clean"
    train_rows = int(train_audit["out_rows"])
    moment_table = _write_feature_distribution_artifacts_from_bin(npz_dir / "train_x.f32", int(train_rows), int(len(feature_names)), list(feature_names), stats_dir)

    # Convert raw/kept/sampled counters into float rates for report consumption.
    def _audit_rates(a: dict[str, object]) -> dict[str, float]:
        """Convert raw/kept/sampled counters into float rates."""
        # Compute kept and sampled rates with explicit float conversions.
        raw = float(a["raw_rows"])
        kept = float(a["kept_rows"])
        sampled = float(a["sampled_rows"])
        return {
            "kept_rate": float(kept / raw),
            "sampled_rate_vs_kept": float(sampled / kept),
            "sampled_rate_vs_raw": float(sampled / raw),
        }

    # Record the raw binary storage layout so the dataset can memory-map without loading full arrays.
    storage = {
        "format": "raw_bin_v1",
        "dtype": {"x": "float32", "y": "float32", "meta": "int64"},
        "groups": {
            "train": {"rows": int(train_audit["out_rows"]), "feature_dim": int(len(feature_names)), "x": "train_x.f32", "y": "train_y.f32", "meta": "train_meta.i64"},
            "val": {"rows": int(val_audit["out_rows"]), "feature_dim": int(len(feature_names)), "x": "val_x.f32", "y": "val_y.f32", "meta": "val_meta.i64"},
            "test": {"rows": int(test_audit["out_rows"]), "feature_dim": int(len(feature_names)), "x": "test_x.f32", "y": "test_y.f32", "meta": "test_meta.i64"},
            "predict": {"rows": int(predict_audit["out_rows"]), "feature_dim": int(len(feature_names)), "x": "predict_x.f32", "y": "predict_y.f32", "meta": "predict_meta.i64"},
        },
    }

    # Persist split metadata and preprocessing audit for reproducibility.
    meta = {
        "feature_names": list(feature_names),
        "storage": storage,
        "feature_transform": {
            "stock_features": list(_STOCK_FEATURES),
            "stock_norm": {"type": "cross_sectional_gaussianize", "rank_clip": 1e-3},
            "time_features": list(time_features),
        },
        "dates": {"train": train_dates, "val": val_dates, "test": test_dates, "predict": dates},
        "label": {"type": "forward_log_return", "horizon_minutes": int(config.horizon_minutes)},
        "sampling": {"sample_stocks_per_minute": int(config.sample_stocks_per_minute)},
        "audit": {"train": train_audit, "val": val_audit, "test": test_audit, "predict": predict_audit},
        "audit_rates": {
            "train": _audit_rates(train_audit),
            "val": _audit_rates(val_audit),
            "test": _audit_rates(test_audit),
            "predict": _audit_rates(predict_audit),
        },
        "audit_daily": {"train": train_daily_audits, "val": val_daily_audits, "test": test_daily_audits},
    }
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")
    progress_path.unlink()

    # Return a small in-memory summary for the pipeline.
    return {
        "done": True,
        "feature_names": feature_names,
        "train_rows": int(train_audit["out_rows"]),
        "val_rows": int(val_audit["out_rows"]),
        "test_rows": int(test_audit["out_rows"]),
        "predict_rows": int(predict_audit["out_rows"]),
        "moment_table": moment_table,
        "meta_path": meta_path,
        "audit": meta["audit"],
        "audit_rates": meta["audit_rates"],
    }
