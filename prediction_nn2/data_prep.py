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

from prediction_nn2.html_report import build_page, render_section, render_table, render_value_rows


_STOCK_FEATURES = [
    "ret_1m",
    "ret_5m",
    "ret_2m",
    "ret_3m",
    "ret_10m",
    "hl",
    "oc",
    "log_vol",
    "log_amount",
    "vol_10m",
    "log_vol_5m_mean",
    "log_amount_5m_mean",
]
_TIME_FEATURES = ["minute_norm", "session_minute_norm"]

_RET_FEATURES = ["ret_1m", "ret_2m", "ret_3m", "ret_5m", "ret_10m"]
_HEAVY_TAIL_FEATURES = ["hl", "oc", "vol_10m"]
_IS_ZERO_SOURCE_FEATURES: list[str] = []
_FEATURE_WARMUP_MINUTES = 10


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
    use_cross_sectional_gaussianize: bool
    use_ret_signed_log1p: bool
    use_heavy_tanh: bool
    heavy_tanh_c: float
    add_is_zero_features: bool
    include_session_id: bool
    include_inference_splits: bool
    norm_fit_scope: str
    label_norm: str
    label_norm_fit_scope: str
    days_per_call: int
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


def _resolve_split_dates(config: DataPrepConfig) -> tuple[list[int], list[int], list[int], list[int]]:
    """Resolve available trade dates and split them into train/val/test lists."""
    # List the available trade dates and validate the requested split length.
    dates = list_trade_dates(config)
    n_train = int(config.train_days)
    n_val = int(config.val_days)
    n_test = int(config.test_days)
    need = n_train + n_val + n_test
    if len(dates) < need:
        raise RuntimeError(f"Not enough trade dates: got={len(dates)} need={need}")

    # Slice the available trade-date list into contiguous train/val/test segments.
    train_dates = dates[:n_train]
    val_dates = dates[n_train : n_train + n_val]
    test_dates = dates[n_train + n_val : n_train + n_val + n_test]
    return dates, train_dates, val_dates, test_dates


def _prep_config_contract(config: DataPrepConfig) -> dict[str, object]:
    """Return the output-defining preprocessing contract stored in meta.yaml."""
    # Keep only fields that change the base train/val/test dataset contents.
    return {
        "start_trade_date": int(config.start_trade_date),
        "end_trade_date": int(config.end_trade_date),
        "train_days": int(config.train_days),
        "val_days": int(config.val_days),
        "test_days": int(config.test_days),
        "seed": int(config.seed),
        "horizon_minutes": int(config.horizon_minutes),
        "sample_stocks_per_minute": int(config.sample_stocks_per_minute),
        "use_cross_sectional_gaussianize": bool(config.use_cross_sectional_gaussianize),
        "use_ret_signed_log1p": bool(config.use_ret_signed_log1p),
        "use_heavy_tanh": bool(config.use_heavy_tanh),
        "heavy_tanh_c": float(config.heavy_tanh_c),
        "add_is_zero_features": bool(config.add_is_zero_features),
        "include_session_id": bool(config.include_session_id),
        "norm_fit_scope": str(config.norm_fit_scope),
        "label_norm": str(config.label_norm),
        "label_norm_fit_scope": str(config.label_norm_fit_scope),
    }


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


def _add_features_and_label(df: pd.DataFrame, config: DataPrepConfig) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    """Compute non-rank feature transforms and a forward return label."""
    # Precompute raw forward-minute availability to audit label invalid sources.
    h = int(config.horizon_minutes)
    raw_keys = ["StockCode", "Date", "MinuteIndex"]
    raw_fwd = df.loc[:, raw_keys].copy()
    raw_fwd["MinuteIndex"] = raw_fwd["MinuteIndex"].to_numpy(dtype=int) - int(h)
    raw_fwd["raw_has_fwd_minute"] = 1
    raw_fwd = raw_fwd.drop_duplicates(list(raw_keys), keep="first").reset_index(drop=True)

    # Compute feature columns before applying any label-based filtering.
    df, feat_cols = _compute_feature_frame(df, config)

    # Align forward log_close by joining on (StockCode, Date, MinuteIndex+h) to avoid shift jumps on missing minutes.
    keys = ["StockCode", "Date", "MinuteIndex"]
    fwd = df.loc[:, keys + ["log_close"]].copy()
    fwd["MinuteIndex"] = fwd["MinuteIndex"].to_numpy(dtype=int) - int(h)
    fwd = fwd.rename(columns={"log_close": "log_close_fwd"})
    df = df.merge(fwd, on=list(keys), how="left", sort=False, validate="m:1")
    df["label_ret"] = df["log_close_fwd"] - df["log_close"]

    # Drop the last horizon window minutes explicitly so downstream invalid ratios reflect real data gaps.
    pre_drop_rows = int(df.shape[0])
    df = df.loc[df["MinuteIndex"].to_numpy(dtype=int) <= int(239 - h)].reset_index(drop=True)
    dropped_last_h_rows = int(pre_drop_rows - int(df.shape[0]))

    # Summarize NaN/inf statistics before dropping invalid rows.
    need = feat_cols + ["label_ret"]
    invalid_feature_stats = _collect_invalid_feature_stats(df, feat_cols, "label_ret")

    # Audit label invalid decomposition after horizon dropping to separate missing minutes from price filtering.
    label_invalid_mask = ~np.isfinite(df["label_ret"].to_numpy(dtype=float, copy=False))
    missing_fwd_mask = df["log_close_fwd"].isna().to_numpy(dtype=bool, copy=False)
    raw_has_fwd_mask = (
        df.loc[:, raw_keys]
        .merge(raw_fwd, on=list(raw_keys), how="left", sort=False)["raw_has_fwd_minute"]
        .fillna(0)
        .to_numpy(dtype=np.int8, copy=False)
        == 1
    )
    raw_missing_mask = missing_fwd_mask & ~raw_has_fwd_mask
    filtered_missing_mask = missing_fwd_mask & raw_has_fwd_mask
    other_invalid_mask = label_invalid_mask & ~missing_fwd_mask

    # Keep only rows with finite features and label.
    m = np.ones((df.shape[0],), dtype=bool)
    for c in need:
        v = df[c].to_numpy(dtype=float)
        m &= np.isfinite(v)
    df = df.loc[m, ["StockCode", "DateTime", "Date", "MinuteIndex"] + feat_cols + ["label_ret"]].reset_index(drop=True)
    audit = {
        "label_alignment": "minute_index_join",
        "dropped_last_h_rows": int(dropped_last_h_rows),
        "horizon_minutes": int(h),
        "label_invalid_rows": int(label_invalid_mask.sum()),
        "label_invalid_ratio": float(label_invalid_mask.mean()) if int(df.shape[0]) > 0 else float("nan"),
        "label_invalid_missing_fwd_raw_absent_rows": int(raw_missing_mask.sum()),
        "label_invalid_missing_fwd_raw_absent_ratio": float(raw_missing_mask.mean()) if int(df.shape[0]) > 0 else float("nan"),
        "label_invalid_missing_fwd_filtered_rows": int(filtered_missing_mask.sum()),
        "label_invalid_missing_fwd_filtered_ratio": float(filtered_missing_mask.mean()) if int(df.shape[0]) > 0 else float("nan"),
        "label_invalid_other_rows": int(other_invalid_mask.sum()),
        "label_invalid_other_ratio": float(other_invalid_mask.mean()) if int(df.shape[0]) > 0 else float("nan"),
        "feature_warmup_minutes": int(_FEATURE_WARMUP_MINUTES),
    }
    return df, invalid_feature_stats, audit


def _compute_feature_frame(df: pd.DataFrame, config: DataPrepConfig) -> tuple[pd.DataFrame, list[str]]:
    """Compute model feature columns before any label-based row filtering."""
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

    # Compute log prices to define return-like features.
    df["log_close"] = np.log(df["Close"].to_numpy(dtype=float))

    # Compute within-stock intraday returns as simple derived features.
    g = df.groupby("StockCode", sort=False)
    df["ret_1m"] = g["log_close"].diff(1)
    df["ret_5m"] = g["log_close"].diff(5)
    df["ret_2m"] = g["log_close"].diff(2)
    df["ret_3m"] = g["log_close"].diff(3)
    df["ret_10m"] = g["log_close"].diff(10)

    # Compute price-shape and activity features without rank transforms.
    df["hl"] = np.log(df["High"].to_numpy(dtype=float) / df["Low"].to_numpy(dtype=float))
    df["oc"] = np.log(df["Close"].to_numpy(dtype=float) / df["Open"].to_numpy(dtype=float))
    df["log_vol"] = np.log(df["Vol"].to_numpy(dtype=float) + 1.0)
    df["log_amount"] = np.log(df["Amount"].to_numpy(dtype=float) + 1.0)
    df["minute_norm"] = df["MinuteIndex"].to_numpy(dtype=float) / 240.0
    df["session_minute_norm"] = (df["MinuteIndex"].to_numpy(dtype=int) % 120).astype(float) / 120.0

    # Compute longer-window volatility and activity summaries per stock.
    df["vol_10m"] = g["ret_1m"].rolling(window=10, min_periods=2).std(ddof=0).reset_index(level=0, drop=True)
    df["log_vol_5m_mean"] = g["log_vol"].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    df["log_amount_5m_mean"] = g["log_amount"].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

    # Optionally attach the session_id feature, which is removed by default per spec.
    if bool(config.include_session_id):
        df["session_id"] = (df["MinuteIndex"].to_numpy(dtype=int) >= 120).astype(float)

    # Apply monotonic heavy-tail transforms so NN training is numerically stable.
    if bool(config.use_ret_signed_log1p):
        for name in list(_RET_FEATURES):
            v = df[str(name)].to_numpy(dtype=float, copy=False)
            df[str(name)] = (np.sign(v) * np.log1p(np.abs(v))).astype(np.float32, copy=False)
    if bool(config.use_heavy_tanh):
        c = float(config.heavy_tanh_c)
        for name in list(_HEAVY_TAIL_FEATURES):
            v = df[str(name)].to_numpy(dtype=float, copy=False)
            df[str(name)] = np.tanh(v / c).astype(np.float32, copy=False)

    # Optionally add mass-at-zero indicator features without zscore participation.
    is_zero_cols: list[str] = []
    if bool(config.add_is_zero_features):
        for src in list(_IS_ZERO_SOURCE_FEATURES):
            out_name = f"{str(src)}_is_zero"
            is_zero_cols.append(str(out_name))
            df[str(out_name)] = (df[str(src)].to_numpy(dtype=float, copy=False) == 0.0).astype(np.float32, copy=False)

    # Resolve the ordered feature list used by downstream filtering and export.
    feat_cols = list(_STOCK_FEATURES) + list(is_zero_cols) + list(_TIME_FEATURES)
    if bool(config.include_session_id):
        feat_cols = list(_STOCK_FEATURES) + list(is_zero_cols) + ["minute_norm", "session_id", "session_minute_norm"]
    return df, feat_cols


def _build_single_day_inference_table(trade_date: int, config: DataPrepConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build one inference day dataframe without label-based filtering."""
    # Track raw row count to quantify feature-only filtering.
    day_raw = _read_stock1m_day(int(trade_date), config)
    raw_rows = int(day_raw.shape[0])

    # Compute feature columns without requiring any future label availability.
    day_feat, feat_cols = _compute_feature_frame(day_raw, config)

    # Keep only rows whose feature vector is fully finite.
    m = np.ones((day_feat.shape[0],), dtype=bool)
    for c in list(feat_cols):
        v = day_feat[str(c)].to_numpy(dtype=float, copy=False)
        m &= np.isfinite(v)
    day_feat = day_feat.loc[m].reset_index(drop=True)
    kept_rows = int(day_feat.shape[0])

    # Apply the configured stock-feature transform on each minute cross-section.
    if bool(config.use_cross_sectional_gaussianize):
        day_feat_out = _cross_sectional_gaussianize(day_feat, list(_STOCK_FEATURES))
    else:
        day_feat_out = day_feat

    # Attach date/time integer columns and keep only schema columns.
    day_norm = _date_time_int_columns(day_feat_out)
    stock_feature_names = _stock_feature_output_names(config)
    is_zero_feature_names = _is_zero_feature_output_names(config)
    time_feature_names = _time_feature_output_names(config)
    keep_cols = (
        ["StockCode", "DateTime", "Date", "MinuteIndex"]
        + list(stock_feature_names)
        + list(is_zero_feature_names)
        + list(time_feature_names)
        + ["date_int", "time_int"]
    )
    day_out = day_norm.loc[:, keep_cols].reset_index(drop=True)

    # Assemble an explicit per-day audit record for cross-period aggregation.
    audit = {
        "trade_date": int(trade_date),
        "raw_rows": int(raw_rows),
        "kept_rows": int(kept_rows),
        "sampled_rows": int(day_out.shape[0]),
        "out_rows": int(day_out.shape[0]),
    }
    return day_out, audit


def _collect_invalid_feature_stats(df: pd.DataFrame, feat_cols: list[str], label_col: str) -> list[dict[str, object]]:
    """Collect pooled NaN/inf counters and finite-value moments for features and label."""
    # Define the ordered fields once so output tables stay stable.
    fields = [(str(name), "feature") for name in list(feat_cols)] + [(str(label_col), "label")]
    rows: list[dict[str, object]] = []
    for name, field_type in fields:
        # Compute invalid counters and finite moments for one field.
        v = df[str(name)].to_numpy(dtype=np.float64, copy=False)
        total = int(v.shape[0])
        nan_mask = np.isnan(v)
        posinf_mask = np.isposinf(v)
        neginf_mask = np.isneginf(v)
        finite_mask = np.isfinite(v)
        finite = v[finite_mask]
        if int(finite.shape[0]) > 0:
            s1 = float(finite.sum(dtype=np.float64))
            s2 = float((finite * finite).sum(dtype=np.float64))
            s3 = float((finite * finite * finite).sum(dtype=np.float64))
            s4 = float((finite * finite * finite * finite).sum(dtype=np.float64))
            vmin = float(finite.min())
            vmax = float(finite.max())
        else:
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            s4 = 0.0
            vmin = float("nan")
            vmax = float("nan")
        rows.append(
            {
                "field": str(name),
                "field_type": str(field_type),
                "total_count": int(total),
                "finite_count": int(finite.shape[0]),
                "nan_count": int(nan_mask.sum()),
                "posinf_count": int(posinf_mask.sum()),
                "neginf_count": int(neginf_mask.sum()),
                "sum1": float(s1),
                "sum2": float(s2),
                "sum3": float(s3),
                "sum4": float(s4),
                "min_finite": float(vmin),
                "max_finite": float(vmax),
            }
        )
    return rows


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


def _stock_feature_output_names(config: DataPrepConfig) -> list[str]:
    """Return the persisted stock-feature column names for the current config."""
    # Switch between raw stock features and gaussianized columns.
    if bool(config.use_cross_sectional_gaussianize):
        return [f"{c}_cs" for c in list(_STOCK_FEATURES)]
    return list(_STOCK_FEATURES)


def _is_zero_feature_output_names(config: DataPrepConfig) -> list[str]:
    """Return the persisted is_zero feature column names for the current config."""
    # Append is_zero indicators only when the optional mass-at-zero features are enabled.
    if not bool(config.add_is_zero_features):
        return []
    return [f"{str(src)}_is_zero" for src in list(_IS_ZERO_SOURCE_FEATURES)]


def _time_feature_output_names(config: DataPrepConfig) -> list[str]:
    """Return the persisted time-feature column names for the current config."""
    # Include session_id only when explicitly enabled by config.
    if bool(config.include_session_id):
        return ["minute_norm", "session_id", "session_minute_norm"]
    return list(_TIME_FEATURES)


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


def _standardize_label(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply scalar standardization to one label array."""
    # Standardize label values into a stable zero-mean unit-std scale.
    return ((y - float(mean)) / float(std)).astype(np.float32, copy=False)


def _compute_norm_stats_from_bin(x_path: Path, rows: int, feature_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute pooled mean/std from a raw float32 feature matrix on disk."""
    # Memory-map the raw feature matrix so pooled moments can be streamed.
    x = np.memmap(Path(x_path), mode="r", dtype=np.float32, shape=(int(rows), int(feature_dim)))

    # Accumulate first and second moments in float64 for numerical stability.
    count = np.zeros((int(feature_dim),), dtype=np.int64)
    s1 = np.zeros((int(feature_dim),), dtype=np.float64)
    s2 = np.zeros((int(feature_dim),), dtype=np.float64)
    chunk_rows = 1_000_000
    for st in range(0, int(rows), int(chunk_rows)):
        # Slice one chunk and upcast once before moment accumulation.
        ed = min(int(rows), int(st + chunk_rows))
        blk = np.asarray(x[int(st) : int(ed)], dtype=np.float64)

        # Update pooled sums feature-by-feature to keep the logic explicit.
        for j in range(int(feature_dim)):
            v = blk[:, int(j)]
            v = v[np.isfinite(v)]
            if int(v.shape[0]) == 0:
                continue
            count[int(j)] += int(v.shape[0])
            s1[int(j)] += float(v.sum(dtype=np.float64))
            s2[int(j)] += float((v * v).sum(dtype=np.float64))

    # Convert pooled sums into mean/std vectors.
    mean = s1 / count.astype(np.float64)
    var = s2 / count.astype(np.float64) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    if bool((std <= 0.0).any()):
        raise RuntimeError("Pooled zscore std contains non-positive values.")
    return mean.astype(np.float32), std.astype(np.float32)


def _compute_norm_stats_from_bins(x_paths: list[Path], rows_list: list[int], feature_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute pooled mean/std from multiple raw float32 matrices on disk."""
    # Validate aligned path/row inputs so accumulated moments are meaningful.
    if int(len(x_paths)) != int(len(rows_list)):
        raise RuntimeError(f"x_paths and rows_list length mismatch: {len(x_paths)} vs {len(rows_list)}")

    # Accumulate first and second moments in float64 for numerical stability.
    count = np.zeros((int(feature_dim),), dtype=np.int64)
    s1 = np.zeros((int(feature_dim),), dtype=np.float64)
    s2 = np.zeros((int(feature_dim),), dtype=np.float64)
    chunk_rows = 1_000_000
    for x_path, rows in zip(list(x_paths), list(rows_list)):
        # Memory-map each matrix so pooled moments can stream without materializing all splits.
        x = np.memmap(Path(x_path), mode="r", dtype=np.float32, shape=(int(rows), int(feature_dim)))
        for st in range(0, int(rows), int(chunk_rows)):
            # Slice one chunk and upcast once before moment accumulation.
            ed = min(int(rows), int(st + chunk_rows))
            blk = np.asarray(x[int(st) : int(ed)], dtype=np.float64)

            # Update pooled sums feature-by-feature to keep the logic explicit.
            for j in range(int(feature_dim)):
                v = blk[:, int(j)]
                v = v[np.isfinite(v)]
                if int(v.shape[0]) == 0:
                    continue
                count[int(j)] += int(v.shape[0])
                s1[int(j)] += float(v.sum(dtype=np.float64))
                s2[int(j)] += float((v * v).sum(dtype=np.float64))

    # Convert pooled sums into mean/std vectors.
    mean = s1 / count.astype(np.float64)
    var = s2 / count.astype(np.float64) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    if bool((std <= 0.0).any()):
        raise RuntimeError("Pooled zscore std contains non-positive values.")
    return mean.astype(np.float32), std.astype(np.float32)


def _compute_norm_stats_from_bins_excluding_features(
    x_paths: list[Path],
    rows_list: list[int],
    feature_names: list[str],
    excluded_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pooled mean/std for zscore while excluding configured features."""
    # Convert excluded feature names into a stable include mask aligned with feature_names.
    excluded = set(str(name) for name in list(excluded_features))
    include_mask = np.asarray([str(name) not in excluded for name in list(feature_names)], dtype=bool)

    # Accumulate first and second moments only for included features.
    feature_dim = int(len(feature_names))
    count = np.zeros((int(feature_dim),), dtype=np.int64)
    s1 = np.zeros((int(feature_dim),), dtype=np.float64)
    s2 = np.zeros((int(feature_dim),), dtype=np.float64)
    chunk_rows = 1_000_000
    for x_path, rows in zip(list(x_paths), list(rows_list)):
        # Memory-map each matrix so pooled moments can stream without full materialization.
        x = np.memmap(Path(x_path), mode="r", dtype=np.float32, shape=(int(rows), int(feature_dim)))
        for st in range(0, int(rows), int(chunk_rows)):
            # Slice one chunk and upcast once before moment accumulation.
            ed = min(int(rows), int(st + chunk_rows))
            blk = np.asarray(x[int(st) : int(ed)], dtype=np.float64)

            # Update pooled sums feature-by-feature to keep the logic explicit.
            for j in range(int(feature_dim)):
                # Skip excluded features so their columns remain unchanged by zscore.
                if not bool(include_mask[int(j)]):
                    continue
                v = blk[:, int(j)]
                v = v[np.isfinite(v)]
                if int(v.shape[0]) == 0:
                    continue
                count[int(j)] += int(v.shape[0])
                s1[int(j)] += float(v.sum(dtype=np.float64))
                s2[int(j)] += float((v * v).sum(dtype=np.float64))

    # Convert pooled sums into mean/std vectors and keep excluded columns as identity.
    mean = np.zeros((int(feature_dim),), dtype=np.float64)
    std = np.ones((int(feature_dim),), dtype=np.float64)
    for j in range(int(feature_dim)):
        # Convert included-feature raw sums into pooled mean/std.
        if not bool(include_mask[int(j)]):
            continue
        if int(count[int(j)]) == 0:
            raise RuntimeError(f"No finite values available for zscore feature: {feature_names[int(j)]}")
        mean[int(j)] = float(s1[int(j)] / float(count[int(j)]))
        var = float(s2[int(j)] / float(count[int(j)]) - mean[int(j)] * mean[int(j)])
        std[int(j)] = float(np.sqrt(max(var, 0.0)))
        if float(std[int(j)]) <= 0.0:
            raise RuntimeError(f"Pooled zscore std contains non-positive values: {feature_names[int(j)]}")
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize_bin_inplace(x_path: Path, rows: int, feature_dim: int, mean: np.ndarray, std: np.ndarray) -> None:
    """Apply pooled zscore to a raw float32 feature matrix on disk in place."""
    # Memory-map the matrix in read-write mode so normalization can stream in place.
    x = np.memmap(Path(x_path), mode="r+", dtype=np.float32, shape=(int(rows), int(feature_dim)))

    # Rewrite one chunk at a time to bound peak memory.
    chunk_rows = 1_000_000
    for st in range(0, int(rows), int(chunk_rows)):
        # Standardize one chunk and write it back to disk.
        ed = min(int(rows), int(st + chunk_rows))
        blk = np.asarray(x[int(st) : int(ed)], dtype=np.float32)
        x[int(st) : int(ed)] = _standardize(blk, mean, std)
    x.flush()


def _compute_label_norm_stats_from_bin(y_path: Path, rows: int) -> tuple[float, float]:
    """Compute pooled mean/std from one raw float32 label column on disk."""
    # Memory-map the raw label vector so pooled moments can be streamed.
    y = np.memmap(Path(y_path), mode="r", dtype=np.float32, shape=(int(rows), 1))

    # Accumulate first and second moments in float64 for numerical stability.
    count = 0
    s1 = 0.0
    s2 = 0.0
    chunk_rows = 1_000_000
    for st in range(0, int(rows), int(chunk_rows)):
        # Slice one chunk and keep only finite values.
        ed = min(int(rows), int(st + chunk_rows))
        blk = np.asarray(y[int(st) : int(ed), 0], dtype=np.float64)
        blk = blk[np.isfinite(blk)]
        if int(blk.shape[0]) == 0:
            continue
        count += int(blk.shape[0])
        s1 += float(blk.sum(dtype=np.float64))
        s2 += float((blk * blk).sum(dtype=np.float64))

    # Convert pooled sums into scalar mean/std parameters.
    mean = float(s1 / float(count))
    var = float(s2 / float(count) - mean * mean)
    std = float(np.sqrt(max(var, 0.0)))
    if float(std) <= 0.0:
        raise RuntimeError("Label zscore std contains non-positive values.")
    return float(mean), float(std)


def _compute_label_norm_stats_from_bins(y_paths: list[Path], rows_list: list[int]) -> tuple[float, float]:
    """Compute pooled mean/std from multiple raw float32 label columns on disk."""
    # Validate aligned path/row inputs so accumulated moments are meaningful.
    if int(len(y_paths)) != int(len(rows_list)):
        raise RuntimeError(f"y_paths and rows_list length mismatch: {len(y_paths)} vs {len(rows_list)}")

    # Accumulate first and second moments in float64 for numerical stability.
    count = 0
    s1 = 0.0
    s2 = 0.0
    chunk_rows = 1_000_000
    for y_path, rows in zip(list(y_paths), list(rows_list)):
        # Memory-map each label vector so pooled moments can stream without full materialization.
        y = np.memmap(Path(y_path), mode="r", dtype=np.float32, shape=(int(rows), 1))
        for st in range(0, int(rows), int(chunk_rows)):
            # Slice one chunk and keep only finite values.
            ed = min(int(rows), int(st + chunk_rows))
            blk = np.asarray(y[int(st) : int(ed), 0], dtype=np.float64)
            blk = blk[np.isfinite(blk)]
            if int(blk.shape[0]) == 0:
                continue
            count += int(blk.shape[0])
            s1 += float(blk.sum(dtype=np.float64))
            s2 += float((blk * blk).sum(dtype=np.float64))

    # Convert pooled sums into scalar mean/std parameters.
    mean = float(s1 / float(count))
    var = float(s2 / float(count) - mean * mean)
    std = float(np.sqrt(max(var, 0.0)))
    if float(std) <= 0.0:
        raise RuntimeError("Label zscore std contains non-positive values.")
    return float(mean), float(std)


def _standardize_label_bin_inplace(y_path: Path, rows: int, mean: float, std: float) -> None:
    """Apply pooled zscore to a raw float32 label column on disk in place."""
    # Memory-map the label column in read-write mode so normalization can stream in place.
    y = np.memmap(Path(y_path), mode="r+", dtype=np.float32, shape=(int(rows), 1))

    # Rewrite one chunk at a time to bound peak memory.
    chunk_rows = 1_000_000
    for st in range(0, int(rows), int(chunk_rows)):
        # Standardize one chunk and write it back to disk.
        ed = min(int(rows), int(st + chunk_rows))
        blk = np.asarray(y[int(st) : int(ed), 0], dtype=np.float32)
        y[int(st) : int(ed), 0] = _standardize_label(blk, float(mean), float(std))
    y.flush()


def _write_pooled_zscore_artifacts(
    stats_path: Path,
    scope: str,
    feature_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    rows: int,
    excluded_features: list[str],
) -> None:
    """Write pooled zscore parameters to YAML for reproducibility."""
    # Persist pooled mean/std vectors in YAML for reproducibility.
    import yaml

    rows_yaml = []
    for j, name in enumerate(list(feature_names)):
        # Store one feature's pooled parameters with stable scalar conversions.
        rows_yaml.append({"feature": str(name), "mean": float(mean[int(j)]), "std": float(std[int(j)])})
    payload = {"scope": str(scope), "rows": int(rows), "excluded_features": [str(x) for x in list(excluded_features)], "features": rows_yaml}
    stats_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_label_zscore_artifacts(stats_path: Path, scope: str, mean: float, std: float, rows: int) -> None:
    """Write label pooled zscore parameters to YAML for reproducibility."""
    # Persist scalar label mean/std in YAML for reproducibility.
    import yaml

    payload = {
        "scope": str(scope),
        "rows": int(rows),
        "label": {"feature": "label_ret", "mean": float(mean), "std": float(std)},
    }
    stats_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_feature_distribution_artifacts_from_bins(
    x_paths: list[Path], rows_list: list[int], feature_dim: int, feature_names: list[str], out_dir: Path
) -> pd.DataFrame:
    """Compute distribution stats from multiple raw float32 matrices and write histogram plots."""
    # Validate aligned path/row inputs so pooled moments are meaningful.
    if int(len(x_paths)) != int(len(rows_list)):
        raise RuntimeError(f"x_paths and rows_list length mismatch: {len(x_paths)} vs {len(rows_list)}")

    # Ensure output directory exists before writing artifacts.
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Stream over each memmap in chunks to bound peak memory while keeping IO sequential.
    chunk_rows = 1_000_000
    for x_path, rows in zip(list(x_paths), list(rows_list)):
        # Memory-map the raw feature matrix so multi-split datasets do not require RAM materialization.
        x = np.memmap(Path(x_path), mode="r", dtype=np.float32, shape=(int(rows), int(feature_dim)))
        for st in range(0, int(rows), int(chunk_rows)):
            # Slice a contiguous block and upcast to float64 for stable moment accumulation.
            ed = min(int(rows), int(st + chunk_rows))
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
    _write_feature_distribution_overview(table, hist_counts, centers, edges, list(feature_names), out_dir / "pooled_feature_grid.png")
    return table



def _aggregate_invalid_feature_stats(daily_audits: list[dict[str, object]]) -> pd.DataFrame:
    """Aggregate daily invalid-value stats into a pooled dataframe."""
    # Merge per-day field statistics by summing counters and raw moments.
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for audit in list(daily_audits):
        # Read the per-day invalid stats list from the audit payload.
        for row in list(audit["invalid_feature_stats"]):
            key = (str(row["field"]), str(row["field_type"]))
            if key not in merged:
                merged[key] = {
                    "field": str(row["field"]),
                    "field_type": str(row["field_type"]),
                    "total_count": 0,
                    "finite_count": 0,
                    "nan_count": 0,
                    "posinf_count": 0,
                    "neginf_count": 0,
                    "sum1": 0.0,
                    "sum2": 0.0,
                    "sum3": 0.0,
                    "sum4": 0.0,
                    "min_finite": float("nan"),
                    "max_finite": float("nan"),
                }
            acc = merged[key]
            acc["total_count"] = int(acc["total_count"]) + int(row["total_count"])
            acc["finite_count"] = int(acc["finite_count"]) + int(row["finite_count"])
            acc["nan_count"] = int(acc["nan_count"]) + int(row["nan_count"])
            acc["posinf_count"] = int(acc["posinf_count"]) + int(row["posinf_count"])
            acc["neginf_count"] = int(acc["neginf_count"]) + int(row["neginf_count"])
            acc["sum1"] = float(acc["sum1"]) + float(row["sum1"])
            acc["sum2"] = float(acc["sum2"]) + float(row["sum2"])
            acc["sum3"] = float(acc["sum3"]) + float(row["sum3"])
            acc["sum4"] = float(acc["sum4"]) + float(row["sum4"])
            if np.isfinite(float(row["min_finite"])):
                acc["min_finite"] = (
                    float(row["min_finite"])
                    if not np.isfinite(float(acc["min_finite"]))
                    else float(min(float(acc["min_finite"]), float(row["min_finite"])))
                )
            if np.isfinite(float(row["max_finite"])):
                acc["max_finite"] = (
                    float(row["max_finite"])
                    if not np.isfinite(float(acc["max_finite"]))
                    else float(max(float(acc["max_finite"]), float(row["max_finite"])))
                )

    # Convert merged counters and moments into a stable summary table.
    rows: list[dict[str, object]] = []
    for key in sorted(merged.keys()):
        # Compute descriptive statistics from pooled finite moments.
        acc = merged[key]
        total = int(acc["total_count"])
        finite = int(acc["finite_count"])
        nan_count = int(acc["nan_count"])
        posinf_count = int(acc["posinf_count"])
        neginf_count = int(acc["neginf_count"])
        inf_count = int(posinf_count + neginf_count)
        invalid_count = int(nan_count + inf_count)
        if int(finite) > 0:
            mean = float(acc["sum1"]) / float(finite)
            m2 = float(acc["sum2"]) / float(finite)
            m3 = float(acc["sum3"]) / float(finite)
            m4 = float(acc["sum4"]) / float(finite)
            mu2 = float(m2 - mean * mean)
            std = float(np.sqrt(max(mu2, 0.0)))
            if float(std) > 0.0:
                mu3 = float(m3 - 3.0 * m2 * mean + 2.0 * mean * mean * mean)
                mu4 = float(m4 - 4.0 * m3 * mean + 6.0 * m2 * mean * mean - 3.0 * mean**4)
                skew = float(mu3 / (std**3))
                kurtosis = float(mu4 / (mu2 * mu2) - 3.0) if float(mu2) > 0.0 else float("nan")
            else:
                skew = float("nan")
                kurtosis = float("nan")
        else:
            mean = float("nan")
            std = float("nan")
            skew = float("nan")
            kurtosis = float("nan")
        rows.append(
            {
                "field": str(acc["field"]),
                "field_type": str(acc["field_type"]),
                "total_count": int(total),
                "finite_count": int(finite),
                "nan_count": int(nan_count),
                "posinf_count": int(posinf_count),
                "neginf_count": int(neginf_count),
                "inf_count": int(inf_count),
                "invalid_count": int(invalid_count),
                "nan_ratio": float(nan_count / total),
                "inf_ratio": float(inf_count / total),
                "invalid_ratio": float(invalid_count / total),
                "mean_finite": float(mean),
                "std_finite": float(std),
                "skew_finite": float(skew),
                "kurtosis_finite": float(kurtosis),
                "min_finite": float(acc["min_finite"]),
                "max_finite": float(acc["max_finite"]),
            }
        )
    return pd.DataFrame(rows)


def _write_invalid_feature_artifacts(invalid_table: pd.DataFrame, out_dir: Path) -> None:
    """Write invalid-value statistics to CSV and self-contained HTML report."""
    # Ensure the data-clean output directory exists before writing files.
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "invalid_feature_stats.csv"
    html_path = out_dir / "invalid_feature_report.html"
    invalid_table.to_csv(csv_path, index=False)

    # Summarize the most problematic fields for fast inspection.
    top_invalid = invalid_table.sort_values(["invalid_ratio", "field"], ascending=[False, True], kind="stable").reset_index(drop=True)
    top_nan = invalid_table.sort_values(["nan_ratio", "field"], ascending=[False, True], kind="stable").reset_index(drop=True)
    top_inf = invalid_table.sort_values(["inf_ratio", "field"], ascending=[False, True], kind="stable").reset_index(drop=True)

    # Build the scalar summary block for the invalid-value report.
    summary_rows = [
        ("stats_csv", csv_path.as_posix()),
        ("field_count", str(int(invalid_table.shape[0]))),
        ("max_invalid_ratio", f"{float(top_invalid.iloc[0]['invalid_ratio']):.6f} ({str(top_invalid.iloc[0]['field'])})"),
        ("max_nan_ratio", f"{float(top_nan.iloc[0]['nan_ratio']):.6f} ({str(top_nan.iloc[0]['field'])})"),
        ("max_inf_ratio", f"{float(top_inf.iloc[0]['inf_ratio']):.6f} ({str(top_inf.iloc[0]['field'])})"),
    ]

    # Build the top-invalid HTML table in one vertical section.
    top_table = render_table(
        ["field", "type", "nan_ratio", "inf_ratio", "invalid_ratio", "mean_finite", "std_finite", "skew_finite", "kurtosis_finite"],
        [
            [
                str(row["field"]),
                str(row["field_type"]),
                f"{float(row['nan_ratio']):.6f}",
                f"{float(row['inf_ratio']):.6f}",
                f"{float(row['invalid_ratio']):.6f}",
                f"{float(row['mean_finite']):.6f}",
                f"{float(row['std_finite']):.6f}",
                f"{float(row['skew_finite']):.6f}",
                f"{float(row['kurtosis_finite']):.6f}",
            ]
            for row in top_invalid.head(8).to_dict(orient="records")
        ],
    )

    # Render the self-contained HTML document.
    html = build_page(
        "Data Clean NaN/inf Report",
        "Self-contained HTML generated from pooled invalid-value statistics.",
        [
            render_section("Summary", render_value_rows(summary_rows)),
            render_section("Invalid Ratio Top 8", top_table),
        ],
    )
    html_path.write_text(html, encoding="utf-8")


def _write_label_audit_artifacts(daily_audits: list[dict[str, object]], out_dir: Path) -> pd.DataFrame:
    """Write label alignment and invalid decomposition audits to CSV."""
    # Ensure the output directory exists before writing files.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flatten one CSV row per trade_date so label invalid sources are easy to inspect.
    rows: list[dict[str, object]] = []
    for a in list(daily_audits):
        # Read the label_audit payload and attach the parent trade_date.
        la = dict(a.get("label_audit", {}))
        la["trade_date"] = int(a["trade_date"])
        rows.append(dict(la))
    table = pd.DataFrame(rows).sort_values(["trade_date"], kind="stable").reset_index(drop=True)
    table.to_csv(out_dir / "label_audit.csv", index=False)
    return table


def _write_kept_rows_by_minute_artifacts(npz_dir: Path, trade_dates: list[int], horizon_minutes: int, out_dir: Path) -> pd.DataFrame:
    """Aggregate per-day minute counts and write kept_rows_by_minute CSV+PNG."""
    # Ensure the output directory exists before writing files.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sum per-day per-minute row counts so minute-level cliffs are visible.
    minute_dir = Path(npz_dir) / "audit_aux" / "minute_counts"
    total = np.zeros((240,), dtype=np.int64)
    for d in list(trade_dates):
        # Load one day's counts and update the global accumulator.
        p = minute_dir / f"{int(d)}.npy"
        total += np.load(p).astype(np.int64, copy=False)

    # Persist the aggregated minute table and a lightweight line plot.
    n_minutes = int(240 - int(horizon_minutes))
    table = pd.DataFrame({"MinuteIndex": np.arange(n_minutes, dtype=np.int64), "kept_rows": total[:n_minutes].astype(np.int64, copy=False)})
    table.to_csv(out_dir / "kept_rows_by_minute.csv", index=False)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(table["MinuteIndex"].to_numpy(dtype=int), table["kept_rows"].to_numpy(dtype=float), linewidth=1.5)
    ax.set_title("Kept rows by MinuteIndex")
    ax.set_xlabel("MinuteIndex")
    ax.set_ylabel("kept_rows")
    fig.tight_layout()
    fig.savefig(out_dir / "kept_rows_by_minute.png", dpi=160)
    plt.close(fig)
    return table


def _write_drop_top_stocks_artifacts(npz_dir: Path, trade_dates: list[int], out_dir: Path) -> pd.DataFrame:
    """Aggregate per-day stock minute counts and write drop_top_stocks CSV."""
    # Ensure the output directory exists before writing files.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate kept/expected minute totals by StockCode across all processed days.
    stock_dir = Path(npz_dir) / "audit_aux" / "stock_counts"
    kept_by_stock: dict[int, int] = {}
    expected_by_stock: dict[int, int] = {}
    for d in list(trade_dates):
        # Load one day's stock-minute counts and update global dictionaries.
        p = stock_dir / f"{int(d)}.npz"
        blob = np.load(p)
        expected = int(np.asarray(blob["expected_minutes"]).reshape(-1)[0])
        codes = np.asarray(blob["stock_code"], dtype=np.int64)
        kept = np.asarray(blob["kept_minutes"], dtype=np.int64)
        for code, k in zip(list(codes), list(kept)):
            # Add one day's expected/kept minute counters for this StockCode.
            c = int(code)
            kept_by_stock[c] = int(kept_by_stock.get(c, 0)) + int(k)
            expected_by_stock[c] = int(expected_by_stock.get(c, 0)) + int(expected)

    # Convert aggregated counters into drop ratios and missing-minute totals.
    rows: list[dict[str, object]] = []
    for code in sorted(list(expected_by_stock.keys())):
        # Compute drop ratio on the expected-minute denominator.
        expected = int(expected_by_stock[int(code)])
        kept = int(kept_by_stock.get(int(code), 0))
        missing = int(expected - kept)
        drop_ratio = float(missing / expected)
        rows.append({"StockCode": int(code), "drop_ratio": float(drop_ratio), "missing_minutes": int(missing)})
    table = pd.DataFrame(rows).sort_values(["drop_ratio", "missing_minutes", "StockCode"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
    table.to_csv(out_dir / "drop_top_stocks.csv", index=False)
    return table


def _write_value_transform_params_artifact(
    out_dir: Path,
    *,
    ret_features: list[str],
    heavy_tail_features: list[str],
    is_zero_features: list[str],
    time_features: list[str],
    use_ret_signed_log1p: bool,
    use_heavy_tanh: bool,
    heavy_tanh_c: float,
    feature_warmup_minutes: int,
) -> None:
    """Write value-transform parameters to YAML so train/apply stays explicit."""
    # Ensure the output directory exists before writing the YAML payload.
    out_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    # Persist the full transform contract so downstream stages can audit settings.
    payload = {
        "ret_signed_log1p": {"enabled": bool(use_ret_signed_log1p), "features": [str(x) for x in list(ret_features)]},
        "heavy_tanh": {"enabled": bool(use_heavy_tanh), "c": float(heavy_tanh_c), "features": [str(x) for x in list(heavy_tail_features)]},
        "is_zero": {"enabled": int(len(is_zero_features)) > 0, "features": [str(x) for x in list(is_zero_features)], "zscore_excluded": True},
        "time_features": [str(x) for x in list(time_features)],
        "feature_warmup_minutes": int(feature_warmup_minutes),
    }
    (out_dir / "value_transform.yaml").write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_feature_distribution_artifacts(x: np.ndarray, feature_names: list[str], out_dir: Path) -> pd.DataFrame:
    """Compute distribution stats and write per-feature histogram plots."""
    # Ensure output directory exists before writing artifacts.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predefine histogram bins so the combined overview shares one stable x-axis.
    edges = np.linspace(-5.0, 5.0, 81, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_counts = np.zeros((int(len(feature_names)), int(edges.shape[0] - 1)), dtype=np.int64)

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

        # Update histogram counts for the pooled overview figure.
        idx = np.searchsorted(edges, v, side="right") - 1
        idx = np.clip(idx, 0, int(edges.shape[0] - 2))
        hist_counts[int(j)] = np.bincount(idx.astype(np.int64, copy=False), minlength=int(edges.shape[0] - 1)).astype(np.int64, copy=False)

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
    _write_feature_distribution_overview(table, hist_counts, centers, edges, list(feature_names), out_dir / "pooled_feature_grid.png")
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
    _write_feature_distribution_overview(table, hist_counts, centers, edges, list(feature_names), out_dir / "pooled_feature_grid.png")
    return table


def _write_feature_distribution_overview(
    moment_table: pd.DataFrame,
    hist_counts: np.ndarray,
    centers: np.ndarray,
    edges: np.ndarray,
    feature_names: list[str],
    out_path: Path,
) -> None:
    """Write one pooled feature-distribution overview grid."""
    # Ensure the output directory exists before writing the combined figure.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a compact subplot grid that covers all pooled cleaned features.
    n_feat = int(len(feature_names))
    n_cols = 4
    n_rows = int(np.ceil(float(n_feat) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    # Precompute the standard normal reference curve once for all subplots.
    grid = np.linspace(-5.0, 5.0, 400)
    ref_pdf = stats.norm.pdf(grid, loc=0.0, scale=1.0)
    bin_width = float(edges[1] - edges[0])
    moments = moment_table.set_index("feature")

    # Render one histogram-plus-moments panel per feature.
    for j, name in enumerate(list(feature_names)):
        ax = axes_flat[int(j)]
        dens = hist_counts[int(j)].astype(np.float64) / float(max(int(hist_counts[int(j)].sum()), 1))
        dens = dens / float(bin_width)
        row = moments.loc[str(name)]
        ax.bar(centers, dens, width=bin_width, alpha=0.6, color="#4c72b0", align="center")
        ax.plot(grid, ref_pdf, color="#dd8452", linewidth=1.6)
        ax.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_xlim(-5.0, 5.0)
        ax.set_title(
            f"{name}\nμ={float(row['mean']):.2f}, σ={float(row['std']):.2f}, "
            f"skew={float(row['skew']):.2f}, kurt={float(row['kurtosis']):.2f}",
            fontsize=9,
        )
        ax.set_xlabel("zscore")
        ax.set_ylabel("density")

    # Hide unused subplots so the overview stays visually clean.
    for j in range(int(n_feat), int(len(axes_flat))):
        axes_flat[int(j)].axis("off")

    # Save the combined pooled overview figure.
    fig.suptitle("Data Clean pooled feature distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _build_single_day_table(trade_date: int, config: DataPrepConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build one day dataframe for NPZ export and return a per-day audit dict."""
    # Track raw row count to quantify feature/label missing filters.
    day_raw = _read_stock1m_day(int(trade_date), config)
    raw_rows = int(day_raw.shape[0])

    # Compute raw features/label and then sample within each minute.
    day_feat, invalid_feature_stats, label_audit = _add_features_and_label(day_raw, config)
    kept_rows = int(day_feat.shape[0])

    # Persist per-day missingness auxiliaries so global artifacts can be built without re-reading raw data.
    npz_dir = Path(config.out_dir) / "npz"
    aux_dir = npz_dir / "audit_aux"
    (aux_dir / "minute_counts").mkdir(parents=True, exist_ok=True)
    (aux_dir / "stock_counts").mkdir(parents=True, exist_ok=True)
    expected_minutes = int(240 - int(config.horizon_minutes) - int(_FEATURE_WARMUP_MINUTES))
    minute_counts = np.bincount(day_feat["MinuteIndex"].to_numpy(dtype=np.int64, copy=False), minlength=240).astype(np.int64, copy=False)
    np.save(aux_dir / "minute_counts" / f"{int(trade_date)}.npy", minute_counts)
    stock_kept = day_feat.groupby("StockCode", sort=False)["MinuteIndex"].nunique()
    np.savez_compressed(
        aux_dir / "stock_counts" / f"{int(trade_date)}.npz",
        trade_date=np.asarray([int(trade_date)], dtype=np.int32),
        expected_minutes=np.asarray([int(expected_minutes)], dtype=np.int32),
        stock_code=stock_kept.index.to_numpy(dtype=np.int64, copy=False),
        kept_minutes=stock_kept.to_numpy(dtype=np.int32, copy=False),
    )

    day_samp = _sample_per_minute(day_feat, config)
    sampled_rows = int(day_samp.shape[0])

    # Apply the configured stock-feature transform on each minute cross-section.
    if bool(config.use_cross_sectional_gaussianize):
        day_feat_out = _cross_sectional_gaussianize(day_samp, list(_STOCK_FEATURES))
    else:
        day_feat_out = day_samp

    # Attach date/time integer columns and keep only schema columns.
    day_norm = _date_time_int_columns(day_feat_out)
    stock_feature_names = _stock_feature_output_names(config)
    is_zero_feature_names = _is_zero_feature_output_names(config)
    time_feature_names = _time_feature_output_names(config)
    keep_cols = (
        ["StockCode", "DateTime", "Date", "MinuteIndex"]
        + list(stock_feature_names)
        + list(is_zero_feature_names)
        + list(time_feature_names)
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
        "invalid_feature_stats": list(invalid_feature_stats),
        "label_audit": dict(label_audit),
    }
    return day_out, audit


def _build_single_day_table_task(args: tuple[int, DataPrepConfig]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Unpack starmap-style args so Pool.imap can stream day results."""
    # Unpack task tuple and delegate into the core per-day builder.
    trade_date, config = args
    return _build_single_day_table(int(trade_date), config)


def _build_single_day_inference_table_task(args: tuple[int, DataPrepConfig]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Unpack starmap-style args so Pool.imap can stream inference day results."""
    # Unpack task tuple and delegate into the core inference per-day builder.
    trade_date, config = args
    return _build_single_day_inference_table(int(trade_date), config)


def _data_prep_iterator(
    pool: mp.pool.Pool | None,
    tasks: list[tuple[int, DataPrepConfig]],
    task_fn,
):
    """Build the day-result iterator used by data prep."""
    # Run sequentially when workers=1 so debugging and local repro stay simple.
    if pool is None:
        return map(task_fn, list(tasks))

    # Stream one day at a time from worker processes to bound parent memory.
    return pool.imap(task_fn, list(tasks), chunksize=1)


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


def _inference_group_contract(group_name: str, source_split: str, date_list: list[int]) -> dict[str, object]:
    """Describe one formal inference group contract."""
    # Record the sample definition so old label-filtered groups can be detected.
    return {
        "type": "feature_only_inference_split",
        "group": str(group_name),
        "source_split": str(source_split),
        "dates": [int(d) for d in list(date_list)],
        "uses_label_filter": False,
        "uses_sampling": False,
        "target_placeholder": "nan",
    }


def _inference_split_specs(train_dates: list[int], val_dates: list[int], test_dates: list[int]) -> list[tuple[str, list[int], str]]:
    """Return the formal inference split specification list."""
    # Define the stable inference split ordering once.
    return [
        ("inference_train", [int(d) for d in list(train_dates)], "train"),
        ("inference_val", [int(d) for d in list(val_dates)], "val"),
        ("inference_test", [int(d) for d in list(test_dates)], "test"),
    ]


def _load_pooled_zscore_params(stats_path: Path, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load pooled feature zscore parameters from YAML."""
    # Parse the YAML payload and validate the stored feature order.
    import yaml

    payload = yaml.safe_load(Path(stats_path).read_text(encoding="utf-8"))
    rows = list(payload["features"])
    stored_names = [str(row["feature"]) for row in list(rows)]
    if list(stored_names) != list(feature_names):
        raise RuntimeError(f"Pooled zscore feature order mismatch: stored={stored_names} expected={feature_names}")

    # Materialize the mean/std vectors aligned with feature_names.
    mean = np.asarray([float(row["mean"]) for row in list(rows)], dtype=np.float32)
    std = np.asarray([float(row["std"]) for row in list(rows)], dtype=np.float32)
    return mean, std


def _materialize_inference_group(
    config: DataPrepConfig,
    group_name: str,
    date_list: list[int],
    npz_dir: Path,
    feature_names: list[str],
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
) -> tuple[int, list[dict[str, object]], dict[str, object]]:
    """Materialize one formal inference group from raw split dates."""
    # Resolve worker setup before launching a bounded second pass over test dates.
    workers = int(config.workers)
    start_method = "spawn"
    ctx = mp.get_context(str(start_method))
    pool = ctx.Pool(processes=int(workers)) if int(workers) > 1 else None
    try:
        # Stream day-level inference tables and write one raw binary group.
        t0 = time.time()
        tasks = [(int(d), config) for d in list(date_list)]
        it = _data_prep_iterator(pool, tasks, _build_single_day_inference_table_task)
        infer_x = Path(npz_dir) / f"{str(group_name)}_x.f32"
        infer_y = Path(npz_dir) / f"{str(group_name)}_y.f32"
        infer_m = Path(npz_dir) / f"{str(group_name)}_meta.i64"
        rows = 0
        audits: list[dict[str, object]] = []
        with open(infer_x, "wb") as out_x, open(infer_y, "wb") as out_y, open(infer_m, "wb") as out_m:
            for i, (day_df, audit) in enumerate(it, start=1):
                # Convert the per-day dataframe into arrays and apply the training zscore when needed.
                x_day = day_df[feature_names].to_numpy(dtype=np.float32, copy=False)
                if feature_mean is not None and feature_std is not None:
                    x_day = _standardize(x_day, feature_mean, feature_std)
                y_day = np.full((int(x_day.shape[0]), 1), np.nan, dtype=np.float32)
                meta_day = day_df[["StockCode", "date_int", "time_int"]].to_numpy(dtype=np.int64, copy=False)
                x_day.tofile(out_x)
                y_day.tofile(out_y)
                meta_day.tofile(out_m)

                # Track split-level counters for audit aggregation.
                rows += int(x_day.shape[0])
                audits.append(dict(audit))

                # Print coarse-grained progress so long inference builds are observable in logs.
                if int(i) == 1 or int(i) % 20 == 0 or int(i) == int(len(date_list)):
                    print(f"[data_prep] tag={group_name} day={i}/{len(date_list)} rows={rows}", flush=True)
        t1 = time.time()
    finally:
        # Close the worker pool on both success and failure to avoid leaking processes.
        if pool is not None:
            pool.close()
            pool.join()

    # Build split-level audit summary for metadata and report rendering.
    raw_rows = int(sum(int(a["raw_rows"]) for a in list(audits)))
    kept_rows = int(sum(int(a["kept_rows"]) for a in list(audits)))
    sampled_rows = int(sum(int(a["sampled_rows"]) for a in list(audits)))
    split_audit = {
        "tag": str(group_name),
        "workers": int(workers),
        "days": int(len(date_list)),
        "raw_rows": int(raw_rows),
        "kept_rows": int(kept_rows),
        "sampled_rows": int(sampled_rows),
        "out_rows": int(rows),
        "elapsed_seconds": float(t1 - t0),
    }
    return int(rows), audits, split_audit


def _progress_metrics(
    progress: dict[str, object],
    train_dates: list[int],
    val_dates: list[int],
    test_dates: list[int],
) -> dict[str, object]:
    """Summarize chunked data-prep progress into days and time estimates."""
    # Define split order once so cumulative processed-day counting is explicit.
    stage_order = ["train", "val", "test"]
    stage_days = {"train": int(len(train_dates)), "val": int(len(val_dates)), "test": int(len(test_dates))}
    curr_stage = str(progress["stage"])

    # Count processed days by summing completed stages plus the current split index.
    days_total = int(sum(int(v) for v in stage_days.values()))
    days_done = 0
    for stage in list(stage_order):
        # Add fully completed stages before the current progress stage.
        if str(curr_stage) == "done":
            days_done += int(stage_days[stage])
            continue
        if int(stage_order.index(stage)) < int(stage_order.index(curr_stage)):
            days_done += int(stage_days[stage])
            continue
        if str(stage) == str(curr_stage):
            days_done += int(progress["index"])
            break

    # Convert elapsed split counters into aggregate elapsed and ETA estimates.
    elapsed = dict(progress["elapsed_seconds"])
    elapsed_seconds = float(sum(float(elapsed[s]) for s in list(stage_order)))
    seconds_per_day = float(elapsed_seconds / float(days_done)) if int(days_done) > 0 else float("nan")
    estimated_total_seconds = float(seconds_per_day * float(days_total)) if int(days_done) > 0 else float("nan")
    remaining_seconds = float(estimated_total_seconds - elapsed_seconds) if int(days_done) > 0 else float("nan")
    return {
        "stage": str(curr_stage),
        "index": int(progress["index"]),
        "days_total": int(days_total),
        "days_done": int(days_done),
        "seconds_per_day": float(seconds_per_day),
        "elapsed_seconds": float(elapsed_seconds),
        "estimated_total_seconds": float(estimated_total_seconds),
        "remaining_seconds": float(remaining_seconds),
    }


def prepare_npz_splits(config: DataPrepConfig) -> dict[str, object]:
    """Prepare train/val/test NPZ datasets and data-clean artifacts."""
    # Resolve IO paths and short-circuit when a completed meta.yaml already exists.
    npz_dir = Path(config.out_dir) / "npz"
    meta_path = npz_dir / "meta.yaml"
    npz_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    # Enforce train-only fit scopes so statistics cannot leak across splits.
    if str(config.norm_fit_scope) != "train_only":
        raise RuntimeError(f"norm_fit_scope must be train_only to avoid leakage, got: {config.norm_fit_scope}")
    if str(config.label_norm_fit_scope) != "train_only":
        raise RuntimeError(f"label_norm_fit_scope must be train_only to avoid leakage, got: {config.label_norm_fit_scope}")

    # Build the final feature schema once so all binaries and artifacts stay aligned.
    time_features = _time_feature_output_names(config)
    is_zero_features = _is_zero_feature_output_names(config)
    feature_names = list(_stock_feature_output_names(config)) + list(is_zero_features) + list(time_features)
    dates, train_dates, val_dates, test_dates = _resolve_split_dates(config)
    expected_stock_norm = (
        {"type": "cross_sectional_gaussianize", "rank_clip": 1e-3}
        if bool(config.use_cross_sectional_gaussianize)
        else {"type": "pooled_zscore", "scope": str(config.norm_fit_scope), "params_path": "pooled_zscore.yaml"}
    )
    expected_label = {"type": "forward_log_return", "horizon_minutes": int(config.horizon_minutes)}
    expected_label_transform = (
        {"type": "pooled_zscore", "scope": str(config.label_norm_fit_scope), "params_path": "label_zscore.yaml"}
        if str(config.label_norm) == "pooled_zscore"
        else {"type": "none"}
    )
    expected_sampling = {"sample_stocks_per_minute": int(config.sample_stocks_per_minute)}
    expected_dates = {"train": list(train_dates), "val": list(val_dates), "test": list(test_dates)}
    expected_prep_config = _prep_config_contract(config)

    if meta_path.exists():
        # Load the persisted metadata so repeated invocations can resume at training without rebuilding.
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        meta_feature_names = list(meta["feature_names"])
        meta_stock_norm = dict(meta["feature_transform"]["stock_norm"])
        meta_label_transform = dict(meta.get("label_transform", {"type": "none"}))
        if list(meta_feature_names) != list(feature_names) or dict(meta_stock_norm) != dict(expected_stock_norm):
            raise RuntimeError(
                f"Existing meta.yaml does not match current preprocessing config: expected_features={feature_names} expected_stock_norm={expected_stock_norm}"
            )
        if dict(meta_label_transform) != dict(expected_label_transform):
            raise RuntimeError(f"Existing meta.yaml does not match current label_transform config: expected_label_transform={expected_label_transform}")

        # Validate split dates, label, and sampling so old artifacts cannot be silently reused.
        meta_dates = dict(meta["dates"])
        if list(meta_dates["train"]) != list(expected_dates["train"]) or list(meta_dates["val"]) != list(expected_dates["val"]) or list(meta_dates["test"]) != list(expected_dates["test"]):
            raise RuntimeError(f"Existing meta.yaml does not match current split dates: expected_dates={expected_dates}")
        if dict(meta["label"]) != dict(expected_label):
            raise RuntimeError(f"Existing meta.yaml does not match current label config: expected_label={expected_label}")
        if dict(meta["sampling"]) != dict(expected_sampling):
            raise RuntimeError(f"Existing meta.yaml does not match current sampling config: expected_sampling={expected_sampling}")

        # Validate the stored preprocessing contract when it is available.
        meta_prep_config = dict(meta.get("prep_config", {}))
        if len(meta_prep_config) > 0 and dict(meta_prep_config) != dict(expected_prep_config):
            raise RuntimeError(f"Existing meta.yaml does not match current prep_config: expected_prep_config={expected_prep_config}")
        if len(meta_prep_config) == 0 and int(config.sample_stocks_per_minute) > 0:
            raise RuntimeError("Existing meta.yaml is missing prep_config; cannot validate sampled data seed.")
        storage = dict(meta["storage"])
        groups = dict(storage["groups"])
        inference_specs = _inference_split_specs(list(train_dates), list(val_dates), list(test_dates))
        expected_inference_contracts = {
            str(group_name): _inference_group_contract(str(group_name), str(source_split), list(date_list))
            for group_name, date_list, source_split in list(inference_specs)
        }

        # Ensure all referenced binary groups exist on disk before claiming completion.
        for group in list(groups.keys()):
            # Resolve required binary file paths for this group.
            g = dict(groups[str(group)])
            for k in ["x", "y", "meta"]:
                p = npz_dir / str(g[str(k)])
                if not p.exists():
                    raise RuntimeError(f"Missing binary artifact for existing meta.yaml: group={group} key={k} path={p}")

        # Ensure audit-daily YAML files exist so report stages can rebuild without re-running prep.
        for stage in ["train", "val", "test"]:
            # Require the daily-audit file written during incremental data prep.
            audit_path = npz_dir / f"audit_daily_{stage}.yaml"
            if not audit_path.exists():
                raise RuntimeError(f"Missing audit-daily artifact for existing meta.yaml: {audit_path}")

        # Ensure core data-clean artifacts exist so clean-report rebuild stays lightweight.
        stats_dir = Path(config.out_dir) / "data_clean"
        invalid_stats_path = stats_dir / "invalid_feature_stats.csv"
        invalid_report_path = stats_dir / "invalid_feature_report.html"
        if not invalid_stats_path.exists() or not invalid_report_path.exists():
            raise RuntimeError(f"Missing invalid-value artifacts for existing meta.yaml: {invalid_stats_path} {invalid_report_path}")

        # Ensure required audit artifacts exist so missingness and label alignment are observable.
        label_audit_path = stats_dir / "label_audit.csv"
        kept_minute_csv = stats_dir / "kept_rows_by_minute.csv"
        kept_minute_png = stats_dir / "kept_rows_by_minute.png"
        drop_top_stocks_path = stats_dir / "drop_top_stocks.csv"
        value_transform_path = stats_dir / "value_transform.yaml"
        if not label_audit_path.exists():
            raise RuntimeError(f"Missing label audit artifact for existing meta.yaml: {label_audit_path}")
        if not kept_minute_csv.exists() or not kept_minute_png.exists():
            raise RuntimeError(f"Missing kept_rows_by_minute artifacts for existing meta.yaml: {kept_minute_csv} {kept_minute_png}")
        if not drop_top_stocks_path.exists():
            raise RuntimeError(f"Missing drop_top_stocks artifact for existing meta.yaml: {drop_top_stocks_path}")
        if not value_transform_path.exists():
            raise RuntimeError(f"Missing value transform params artifact for existing meta.yaml: {value_transform_path}")

        # Ensure pooled zscore params exist when pooled normalization is requested.
        if str(meta_stock_norm["type"]) == "pooled_zscore":
            pooled_stats_path = stats_dir / str(meta_stock_norm["params_path"])
            if not pooled_stats_path.exists():
                raise RuntimeError(f"Missing pooled zscore params for existing meta.yaml: {pooled_stats_path}")
        if str(meta_label_transform["type"]) == "pooled_zscore":
            label_stats_path = stats_dir / str(meta_label_transform["params_path"])
            if not label_stats_path.exists():
                raise RuntimeError(f"Missing label zscore params for existing meta.yaml: {label_stats_path}")

        # Optionally materialize the formal inference splits from raw split dates.
        if bool(config.include_inference_splits):
            current_inference_contracts = dict(meta.get("inference_contracts", {}))
            need_inference_rebuild = False
            for group_name, _date_list, _source_split in list(inference_specs):
                if str(group_name) not in groups:
                    need_inference_rebuild = True
                    break
                if dict(current_inference_contracts.get(str(group_name), {})) != dict(expected_inference_contracts[str(group_name)]):
                    need_inference_rebuild = True
                    break
            if need_inference_rebuild:
                feature_mean: np.ndarray | None = None
                feature_std: np.ndarray | None = None
                if not bool(config.use_cross_sectional_gaussianize):
                    pooled_stats_path = stats_dir / str(meta_stock_norm["params_path"])
                    feature_mean, feature_std = _load_pooled_zscore_params(pooled_stats_path, list(feature_names))

                # Remove legacy predict artifacts before rebuilding the formal inference groups.
                groups.pop("predict", None)
                meta.get("audit", {}).pop("predict", None)
                meta.get("audit_rates", {}).pop("predict", None)
                meta.get("dates", {}).pop("predict", None)
                meta.get("audit_daily", {}).pop("predict", None)
                for legacy_name in ["predict_x.f32", "predict_y.f32", "predict_meta.i64", "audit_daily_predict.yaml"]:
                    legacy_path = npz_dir / legacy_name
                    if legacy_path.exists():
                        legacy_path.unlink()

                # Rebuild the formal inference groups so they no longer depend on label availability.
                feature_dim = int(groups["train"]["feature_dim"])
                meta["inference_contracts"] = {}
                for group_name, date_list, source_split in list(inference_specs):
                    rows, daily_audits, split_audit = _materialize_inference_group(
                        config,
                        str(group_name),
                        list(date_list),
                        npz_dir,
                        list(feature_names),
                        feature_mean,
                        feature_std,
                    )
                    groups[str(group_name)] = {
                        "rows": int(rows),
                        "feature_dim": int(feature_dim),
                        "x": f"{str(group_name)}_x.f32",
                        "y": f"{str(group_name)}_y.f32",
                        "meta": f"{str(group_name)}_meta.i64",
                    }
                    meta["audit"][str(group_name)] = dict(split_audit)
                    meta["audit_rates"][str(group_name)] = _audit_rates(dict(split_audit))
                    meta["dates"][str(group_name)] = [int(d) for d in list(date_list)]
                    meta["audit_daily"][str(group_name)] = list(daily_audits)
                    meta["inference_contracts"][str(group_name)] = _inference_group_contract(
                        str(group_name),
                        str(source_split),
                        list(date_list),
                    )
                    audit_path = npz_dir / f"audit_daily_{str(group_name)}.yaml"
                    audit_path.write_text(yaml.safe_dump(list(daily_audits), sort_keys=False, allow_unicode=True), encoding="utf-8")
                meta["storage"]["groups"] = groups
            meta_path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")

        # Ensure distribution artifacts exist; recompute them from existing bins when missing.
        moment_path = stats_dir / "feature_moments.csv"
        overview_path = stats_dir / "pooled_feature_grid.png"
        if moment_path.exists() and overview_path.exists():
            moment_table = pd.read_csv(moment_path)
        else:
            # Recompute pooled distribution artifacts from the configured normalization scope.
            scope = str(config.norm_fit_scope)
            if scope == "train_only":
                moment_paths = [npz_dir / "train_x.f32"]
                moment_rows = [int(groups["train"]["rows"])]
            elif scope in ["train_val_test", "inference_all_dates"]:
                moment_paths = [npz_dir / "train_x.f32", npz_dir / "val_x.f32", npz_dir / "test_x.f32"]
                moment_rows = [int(groups["train"]["rows"]), int(groups["val"]["rows"]), int(groups["test"]["rows"])]
            else:
                raise RuntimeError(f"Unknown norm_fit_scope: {scope}")
            moment_table = _write_feature_distribution_artifacts_from_bins(
                list(moment_paths),
                list(moment_rows),
                int(len(feature_names)),
                list(feature_names),
                stats_dir,
            )

        # Build a completed progress payload for stable downstream progress logs.
        progress = {
            "stage": "done",
            "index": 0,
            "elapsed_seconds": {
                "train": float(meta["audit"]["train"]["elapsed_seconds"]),
                "val": float(meta["audit"]["val"]["elapsed_seconds"]),
                "test": float(meta["audit"]["test"]["elapsed_seconds"]),
            },
        }
        groups = dict(meta["storage"]["groups"])
        out = {
            "done": True,
            "feature_names": meta_feature_names,
            "train_rows": int(groups["train"]["rows"]),
            "val_rows": int(groups["val"]["rows"]),
            "test_rows": int(groups["test"]["rows"]),
            "moment_table": moment_table,
            "meta_path": meta_path,
            "audit": dict(meta["audit"]),
            "audit_rates": dict(meta["audit_rates"]),
            "groups": sorted(list(groups.keys())),
            "progress": _progress_metrics(progress, list(meta["dates"]["train"]), list(meta["dates"]["val"]), list(meta["dates"]["test"])),
        }
        for group_name in ["inference_train", "inference_val", "inference_test"]:
            if group_name in groups:
                out[f"{group_name}_rows"] = int(groups[group_name]["rows"])
        return out

    # Define the stable feature ordering early so array materialization is consistent.
    # Select a multiprocessing start method that is safe after importing torch.
    workers = int(config.workers)
    if int(workers) < 1:
        raise RuntimeError(f"workers must be positive, got: {workers}")
    start_method = "spawn"
    print(f"[data_prep] start_method={start_method} workers={workers}", flush=True)

    # Create a shared worker pool once so per-day processing runs in parallel.
    ctx = mp.get_context(str(start_method))
    pool = ctx.Pool(processes=int(workers)) if int(workers) > 1 else None
    try:
        # Load panels, compute features/labels, and sample within each day using multiprocessing map.
        def _build_for_dates(
            tag: str,
            date_list: list[int],
            *,
            x_f,
            y_f,
            meta_f,
        ) -> tuple[int, list[dict[str, object]], dict[str, object]]:
            """Stream one split into raw binary arrays and return (rows, daily_audits, split_audit)."""
            # Stream per-day processing inside worker processes while writing arrays sequentially in the main process.
            t0 = time.time()
            tasks = [(int(d), config) for d in list(date_list)]
            it = _data_prep_iterator(pool, tasks, _build_single_day_table_task)

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

        # Advance through all remaining chunks in one invocation to avoid repeated Pool setup costs.
        stage_paths = {
            "train": (npz_dir / "train_x.f32", npz_dir / "train_y.f32", npz_dir / "train_meta.i64"),
            "val": (npz_dir / "val_x.f32", npz_dir / "val_y.f32", npz_dir / "val_meta.i64"),
            "test": (npz_dir / "test_x.f32", npz_dir / "test_y.f32", npz_dir / "test_meta.i64"),
        }
        while str(progress["stage"]) != "done":
            # Decide which split to advance and slice a bounded batch of dates for progress persistence.
            stage = str(progress["stage"])
            stage_dates = {"train": train_dates, "val": val_dates, "test": test_dates}[stage]
            start_i = int(progress["index"])
            step = int(config.days_per_call)
            step = int(step) if int(step) > 0 else int(len(stage_dates) - int(start_i))
            todo = list(stage_dates[int(start_i) : int(start_i + step)])
            if len(todo) == 0:
                raise RuntimeError(f"progress points past end of stage: stage={stage} index={start_i} total={len(stage_dates)}")

            # Open raw array files in append-or-truncate mode depending on stage progress.
            x_path, y_path, meta_bin_path = stage_paths[stage]
            mode = "ab" if int(start_i) > 0 else "wb"
            with open(x_path, mode) as xf, open(y_path, mode) as yf, open(meta_bin_path, mode) as mf:
                _rows, audits, split_audit = _build_for_dates(stage, todo, x_f=xf, y_f=yf, meta_f=mf)

            # Append daily audits into a YAML list so final meta.yaml can embed the full audit table.
            audit_path = npz_dir / f"audit_daily_{stage}.yaml"
            if audit_path.exists():
                daily = list(yaml.safe_load(audit_path.read_text(encoding="utf-8")))
            else:
                daily = []
            daily.extend(list(audits))
            audit_path.write_text(yaml.safe_dump(daily, sort_keys=False, allow_unicode=True), encoding="utf-8")

            # Advance progress and persist it after each chunk so long runs can resume.
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
    finally:
        # Close the worker pool on both success and failure to avoid leaking processes.
        if pool is not None:
            pool.close()
            pool.join()

    # Load full daily audit tables now that all splits are complete.
    train_daily_audits = list(yaml.safe_load((npz_dir / "audit_daily_train.yaml").read_text(encoding="utf-8")))
    val_daily_audits = list(yaml.safe_load((npz_dir / "audit_daily_val.yaml").read_text(encoding="utf-8")))
    test_daily_audits = list(yaml.safe_load((npz_dir / "audit_daily_test.yaml").read_text(encoding="utf-8")))
    combined_daily_audits = list(train_daily_audits) + list(val_daily_audits) + list(test_daily_audits)
    predict_daily_audits: list[dict[str, object]] = []

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
    inference_audits: dict[str, dict[str, object]] = {}
    inference_daily_audits: dict[str, list[dict[str, object]]] = {}

    # Resolve the data-clean output directory before writing aggregated reports.
    stats_dir = Path(config.out_dir) / "data_clean"

    # Aggregate invalid-value stats before any downstream normalization/reporting.
    invalid_stats_table = _aggregate_invalid_feature_stats(combined_daily_audits)
    _write_invalid_feature_artifacts(invalid_stats_table, stats_dir)

    # Write required audit artifacts so missingness and label alignment are observable.
    predict_dates = list(train_dates) + list(val_dates) + list(test_dates)
    _write_label_audit_artifacts(combined_daily_audits, stats_dir)
    _write_kept_rows_by_minute_artifacts(npz_dir, list(predict_dates), int(config.horizon_minutes), stats_dir)
    _write_drop_top_stocks_artifacts(npz_dir, list(predict_dates), stats_dir)
    _write_value_transform_params_artifact(
        stats_dir,
        ret_features=list(_RET_FEATURES),
        heavy_tail_features=list(_HEAVY_TAIL_FEATURES),
        is_zero_features=list(is_zero_features),
        time_features=list(time_features),
        use_ret_signed_log1p=bool(config.use_ret_signed_log1p),
        use_heavy_tanh=bool(config.use_heavy_tanh),
        heavy_tanh_c=float(config.heavy_tanh_c),
        feature_warmup_minutes=int(_FEATURE_WARMUP_MINUTES),
    )

    # Compute pooled zscore parameters on the requested scope when Gaussianize is disabled.
    pooled_stats_path = stats_dir / "pooled_zscore.yaml"
    label_stats_path = stats_dir / "label_zscore.yaml"
    if bool(config.use_cross_sectional_gaussianize):
        # Reuse the already transformed matrices and inspect pooled all-date feature moments.
        moment_table = _write_feature_distribution_artifacts_from_bins(
            [npz_dir / "train_x.f32", npz_dir / "val_x.f32", npz_dir / "test_x.f32"],
            [int(train_audit["out_rows"]), int(val_audit["out_rows"]), int(test_audit["out_rows"])],
            int(len(feature_names)),
            list(feature_names),
            stats_dir,
        )
    else:
        # Resolve normalization fit scope into explicit binary paths and row counts.
        scope = str(config.norm_fit_scope)
        if scope == "train_only":
            fit_paths = [npz_dir / "train_x.f32"]
            fit_rows = [int(train_audit["out_rows"])]
        elif scope == "train_val_test":
            fit_paths = [npz_dir / "train_x.f32", npz_dir / "val_x.f32", npz_dir / "test_x.f32"]
            fit_rows = [int(train_audit["out_rows"]), int(val_audit["out_rows"]), int(test_audit["out_rows"])]
        elif scope == "inference_all_dates":
            fit_paths = [npz_dir / "train_x.f32", npz_dir / "val_x.f32", npz_dir / "test_x.f32"]
            fit_rows = [int(train_audit["out_rows"]), int(val_audit["out_rows"]), int(test_audit["out_rows"])]
        else:
            raise RuntimeError(f"Unknown norm_fit_scope: {scope}")

        # Estimate pooled mean/std from the requested scope before any zscore rewrite.
        excluded_zscore_features = list(is_zero_features)
        mean, std = _compute_norm_stats_from_bins_excluding_features(
            list(fit_paths),
            list(fit_rows),
            list(feature_names),
            list(excluded_zscore_features),
        )

        # Rewrite every split with the same pooled zscore parameters.
        _standardize_bin_inplace(npz_dir / "train_x.f32", int(train_audit["out_rows"]), int(len(feature_names)), mean, std)
        _standardize_bin_inplace(npz_dir / "val_x.f32", int(val_audit["out_rows"]), int(len(feature_names)), mean, std)
        _standardize_bin_inplace(npz_dir / "test_x.f32", int(test_audit["out_rows"]), int(len(feature_names)), mean, std)

        # Inspect post-zscore pooled feature moments and persist pooled zscore parameters.
        moment_table = _write_feature_distribution_artifacts_from_bins(
            list(fit_paths),
            list(fit_rows),
            int(len(feature_names)),
            list(feature_names),
            stats_dir,
        )
        _write_pooled_zscore_artifacts(
            pooled_stats_path,
            str(scope),
            list(feature_names),
            mean,
            std,
            int(sum(int(r) for r in fit_rows)),
            list(excluded_zscore_features),
        )

    # Compute pooled label zscore parameters when the requested label normalization is enabled.
    if str(config.label_norm) == "none":
        label_transform = {"type": "none"}
    elif str(config.label_norm) == "pooled_zscore":
        # Resolve label normalization fit scope into explicit binary paths and row counts.
        label_scope = str(config.label_norm_fit_scope)
        if label_scope == "train_only":
            label_fit_paths = [npz_dir / "train_y.f32"]
            label_fit_rows = [int(train_audit["out_rows"])]
        elif label_scope == "train_val_test":
            label_fit_paths = [npz_dir / "train_y.f32", npz_dir / "val_y.f32", npz_dir / "test_y.f32"]
            label_fit_rows = [int(train_audit["out_rows"]), int(val_audit["out_rows"]), int(test_audit["out_rows"])]
        elif label_scope == "inference_all_dates":
            label_fit_paths = [npz_dir / "train_y.f32", npz_dir / "val_y.f32", npz_dir / "test_y.f32"]
            label_fit_rows = [int(train_audit["out_rows"]), int(val_audit["out_rows"]), int(test_audit["out_rows"])]
        else:
            raise RuntimeError(f"Unknown label_norm_fit_scope: {label_scope}")

        # Estimate pooled label mean/std from the requested scope before any zscore rewrite.
        label_mean, label_std = _compute_label_norm_stats_from_bins(list(label_fit_paths), list(label_fit_rows))

        # Rewrite every split label file with the same pooled zscore parameters.
        _standardize_label_bin_inplace(npz_dir / "train_y.f32", int(train_audit["out_rows"]), float(label_mean), float(label_std))
        _standardize_label_bin_inplace(npz_dir / "val_y.f32", int(val_audit["out_rows"]), float(label_mean), float(label_std))
        _standardize_label_bin_inplace(npz_dir / "test_y.f32", int(test_audit["out_rows"]), float(label_mean), float(label_std))
        _write_label_zscore_artifacts(label_stats_path, str(label_scope), float(label_mean), float(label_std), int(sum(int(r) for r in label_fit_rows)))
        label_transform = {"type": "pooled_zscore", "scope": str(label_scope), "params_path": label_stats_path.name}
    else:
        raise RuntimeError(f"Unknown label_norm: {config.label_norm}")

    # Record the raw binary storage layout so the dataset can memory-map without loading full arrays.
    storage = {
        "format": "raw_bin_v1",
        "dtype": {"x": "float32", "y": "float32", "meta": "int64"},
        "groups": {
            "train": {"rows": int(train_audit["out_rows"]), "feature_dim": int(len(feature_names)), "x": "train_x.f32", "y": "train_y.f32", "meta": "train_meta.i64"},
            "val": {"rows": int(val_audit["out_rows"]), "feature_dim": int(len(feature_names)), "x": "val_x.f32", "y": "val_y.f32", "meta": "val_meta.i64"},
            "test": {"rows": int(test_audit["out_rows"]), "feature_dim": int(len(feature_names)), "x": "test_x.f32", "y": "test_y.f32", "meta": "test_meta.i64"},
        },
    }

    # Optionally materialize formal inference split binaries after normalization.
    if bool(config.include_inference_splits):
        # Build the formal inference groups from raw split dates without label-based filtering or sampling.
        feature_mean: np.ndarray | None = None
        feature_std: np.ndarray | None = None
        if not bool(config.use_cross_sectional_gaussianize):
            feature_mean = mean
            feature_std = std
        for group_name, date_list, source_split in list(_inference_split_specs(list(train_dates), list(val_dates), list(test_dates))):
            rows, daily_audits, split_audit = _materialize_inference_group(
                config,
                str(group_name),
                list(date_list),
                npz_dir,
                list(feature_names),
                feature_mean,
                feature_std,
            )
            storage["groups"][str(group_name)] = {
                "rows": int(rows),
                "feature_dim": int(len(feature_names)),
                "x": f"{str(group_name)}_x.f32",
                "y": f"{str(group_name)}_y.f32",
                "meta": f"{str(group_name)}_meta.i64",
            }
            inference_audits[str(group_name)] = dict(split_audit)
            inference_daily_audits[str(group_name)] = list(daily_audits)
            audit_path = npz_dir / f"audit_daily_{str(group_name)}.yaml"
            audit_path.write_text(yaml.safe_dump(list(daily_audits), sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Persist split metadata and preprocessing audit for reproducibility.
    meta = {
        "prep_config": expected_prep_config,
        "feature_names": list(feature_names),
        "storage": storage,
        "feature_transform": {
            "stock_features": list(_STOCK_FEATURES),
            "is_zero_features": list(is_zero_features),
            "value_transforms": {
                "ret_signed_log1p": {"enabled": bool(config.use_ret_signed_log1p), "features": list(_RET_FEATURES)},
                "heavy_tanh": {"enabled": bool(config.use_heavy_tanh), "c": float(config.heavy_tanh_c), "features": list(_HEAVY_TAIL_FEATURES)},
                "feature_warmup_minutes": int(_FEATURE_WARMUP_MINUTES),
            },
            "stock_norm": (
                {"type": "cross_sectional_gaussianize", "rank_clip": 1e-3}
                if bool(config.use_cross_sectional_gaussianize)
                else {"type": "pooled_zscore", "scope": str(config.norm_fit_scope), "params_path": pooled_stats_path.name}
            ),
            "time_features": list(time_features),
        },
        "dates": {"train": train_dates, "val": val_dates, "test": test_dates},
        "label": {"type": "forward_log_return", "horizon_minutes": int(config.horizon_minutes)},
        "label_transform": dict(label_transform),
        "sampling": {"sample_stocks_per_minute": int(config.sample_stocks_per_minute)},
        "audit": {"train": train_audit, "val": val_audit, "test": test_audit},
        "audit_rates": {
            "train": _audit_rates(train_audit),
            "val": _audit_rates(val_audit),
            "test": _audit_rates(test_audit),
        },
        "invalid_values": {
            "stats_path": "data_clean/invalid_feature_stats.csv",
            "report_path": "data_clean/invalid_feature_report.html",
        },
        "data_clean_artifacts": {
            "report_path": "data_clean/report.html",
            "feature_moments_path": "data_clean/feature_moments.csv",
            "pooled_distribution_overview_path": "data_clean/pooled_feature_grid.png",
            "label_audit_path": "data_clean/label_audit.csv",
            "kept_rows_by_minute_csv_path": "data_clean/kept_rows_by_minute.csv",
            "kept_rows_by_minute_png_path": "data_clean/kept_rows_by_minute.png",
            "drop_top_stocks_path": "data_clean/drop_top_stocks.csv",
            "value_transform_params_path": "data_clean/value_transform.yaml",
        },
        "audit_daily": {"train": train_daily_audits, "val": val_daily_audits, "test": test_daily_audits},
    }
    if bool(config.include_inference_splits):
        meta["inference_contracts"] = {}
        for group_name, date_list, source_split in list(_inference_split_specs(list(train_dates), list(val_dates), list(test_dates))):
            split_audit = dict(inference_audits[str(group_name)])
            meta["dates"][str(group_name)] = [int(d) for d in list(date_list)]
            meta["audit"][str(group_name)] = split_audit
            meta["audit_rates"][str(group_name)] = _audit_rates(split_audit)
            meta["audit_daily"][str(group_name)] = list(inference_daily_audits[str(group_name)])
            meta["inference_contracts"][str(group_name)] = _inference_group_contract(
                str(group_name),
                str(source_split),
                list(date_list),
            )
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")
    progress_path.unlink()

    # Return a small in-memory summary for the pipeline.
    progress = {"stage": "done", "index": 0, "elapsed_seconds": dict(progress["elapsed_seconds"])}
    out = {
        "done": True,
        "feature_names": feature_names,
        "train_rows": int(train_audit["out_rows"]),
        "val_rows": int(val_audit["out_rows"]),
        "test_rows": int(test_audit["out_rows"]),
        "moment_table": moment_table,
        "meta_path": meta_path,
        "audit": meta["audit"],
        "audit_rates": meta["audit_rates"],
        "groups": sorted(list(storage["groups"].keys())),
        "progress": _progress_metrics(progress, train_dates, val_dates, test_dates),
    }
    for group_name in ["inference_train", "inference_val", "inference_test"]:
        if group_name in storage["groups"]:
            out[f"{group_name}_rows"] = int(storage["groups"][group_name]["rows"])
    return out
