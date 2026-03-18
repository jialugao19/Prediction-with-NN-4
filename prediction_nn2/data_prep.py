"""Prepare NPZ datasets from /data/ashare/market/stock1m and emit data-clean artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    m_price = np.isfinite(close) & np.isfinite(open_) & np.isfinite(high) & np.isfinite(low)
    m_price &= (close > 0.0) & (open_ > 0.0) & (high > 0.0) & (low > 0.0)
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


def prepare_npz_splits(config: DataPrepConfig) -> dict[str, object]:
    """Prepare train/val/test NPZ datasets and data-clean artifacts."""
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

    # Load panels, compute features/labels, and sample within each day.
    def _build_for_dates(tag: str, date_list: list[int]) -> pd.DataFrame:
        # Concatenate per-day processed tables into one split dataframe.
        parts: list[pd.DataFrame] = []
        raw_rows = 0
        kept_rows = 0
        sampled_rows = 0
        for d in list(date_list):
            # Track raw row count to quantify feature/label missing filters.
            day_raw = _read_stock1m_day(int(d), config)
            raw_rows += int(day_raw.shape[0])

            # Compute raw features/label and then sample within each minute.
            day_feat = _add_features_and_label(day_raw, config)
            kept_rows += int(day_feat.shape[0])
            day_samp = _sample_per_minute(day_feat, config)
            sampled_rows += int(day_samp.shape[0])

            # Apply cross-sectional normalization on stock-varying features only.
            stock_feats = [
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
            time_feats = ["minute_norm", "session_id", "session_minute_norm"]
            day_norm = _cross_sectional_gaussianize(day_samp, stock_feats)

            # Attach date/time integer columns and keep only schema columns.
            day_norm = _date_time_int_columns(day_norm)
            keep_cols = (
                ["StockCode", "DateTime", "Date", "MinuteIndex"]
                + [f"{c}_cs" for c in stock_feats]
                + time_feats
                + ["label_ret", "date_int", "time_int"]
            )
            parts.append(day_norm.loc[:, keep_cols])

        # Concatenate day tables and store audit counters on the dataframe attrs.
        out = pd.concat(parts, axis=0).reset_index(drop=True)
        out.attrs["audit"] = {"tag": str(tag), "raw_rows": raw_rows, "kept_rows": kept_rows, "sampled_rows": sampled_rows}
        return out

    train_df = _build_for_dates("train", train_dates)
    val_df = _build_for_dates("val", val_dates)
    test_df = _build_for_dates("test", test_dates)

    # Materialize numpy arrays and keep a stable feature ordering.
    stock_cs_features = [
        "ret_1m_cs",
        "ret_5m_cs",
        "ret_30m_cs",
        "ret_60m_cs",
        "hl_cs",
        "oc_cs",
        "log_vol_cs",
        "log_amount_cs",
        "vol_30m_cs",
        "vol_60m_cs",
        "log_vol_30m_mean_cs",
        "log_amount_30m_mean_cs",
    ]
    time_features = ["minute_norm", "session_id", "session_minute_norm"]
    feature_names = list(stock_cs_features) + list(time_features)

    # Save NPZ splits in a schema expected by Stock1mNpzDataset.
    def _write_split(tag: str, df: pd.DataFrame) -> None:
        # Build x/y/meta arrays in the schema expected by Stock1mNpzDataset.
        x = df[feature_names].to_numpy(dtype=np.float32, copy=False)
        y = df[["label_ret"]].to_numpy(dtype=np.float32, copy=False)
        meta = df[["StockCode", "date_int", "time_int"]].to_numpy(dtype=np.int64, copy=False)
        out_path = Path(config.out_dir) / "npz" / f"{tag}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, x=x, y=y, meta=meta)

    _write_split("train", train_df)
    _write_split("val", val_df)
    _write_split("test", test_df)

    # Write data-clean distribution artifacts from cross-section normalized training features.
    x_train = train_df[feature_names].to_numpy(dtype=np.float32, copy=False)
    stats_dir = Path(config.out_dir) / "data_clean"
    moment_table = _write_feature_distribution_artifacts(x_train, feature_names, stats_dir)

    # Persist split metadata and preprocessing audit for reproducibility.
    train_audit = dict(train_df.attrs.get("audit", {}))
    val_audit = dict(val_df.attrs.get("audit", {}))
    test_audit = dict(test_df.attrs.get("audit", {}))
    meta = {
        "feature_names": list(feature_names),
        "feature_transform": {
            "stock_features": [c.replace("_cs", "") for c in stock_cs_features],
            "stock_norm": {"type": "cross_sectional_gaussianize", "rank_clip": 1e-3},
            "time_features": list(time_features),
        },
        "dates": {"train": train_dates, "val": val_dates, "test": test_dates},
        "label": {"type": "forward_log_return", "horizon_minutes": int(config.horizon_minutes)},
        "sampling": {"sample_stocks_per_minute": int(config.sample_stocks_per_minute)},
        "audit": {"train": train_audit, "val": val_audit, "test": test_audit},
    }
    meta_path = Path(config.out_dir) / "npz" / "meta.yaml"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Return a small in-memory summary for the pipeline.
    return {
        "feature_names": feature_names,
        "train_rows": int(train_df.shape[0]),
        "val_rows": int(val_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "moment_table": moment_table,
        "meta_path": meta_path,
        "audit": meta["audit"],
    }
