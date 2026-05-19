"""Compare multi-day simple baselines from source NPZ and raw minute bars."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Bootstrap the tiny-test helper path for direct execution.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


from common import FEATURE_NAMES, SOURCE_NPZ_DIR, _pearson


STOCK1M_DIR = Path("/data/ashare/market/stock1m")
OUT_DIR = Path("/data-cache/nn/tiny-test-3/multiday_raw_baseline")
START_DATE = 20240301
END_DATE = 20241231
DATE_COUNT = 20
STOCKS_PER_DATE = 300
HORIZON_MINUTES = 10
WINDOW_SIZE = 60
SOURCE_TEST_ROWS = 241788800
SOURCE_TEST_FEATURE_DIM = 14
FEATURES = ["ret_1m", "ret_5m", "ret_10m"]
BASELINES = {
    "ret_1m": {"ret_1m": 1.0},
    "ret_5m": {"ret_5m": 1.0},
    "ret_10m": {"ret_10m": 1.0},
    "ret_combo": {"ret_1m": 1.0, "ret_5m": 0.5, "ret_10m": 0.25},
}


def main() -> None:
    """Run the multi-day raw-vs-NPZ baseline comparison."""
    # Resolve dates and selected stock codes per date.
    dates = _list_dates(int(START_DATE), int(END_DATE), int(DATE_COUNT))
    selected_codes = _select_codes_by_date(list(dates), int(STOCKS_PER_DATE))

    # Build raw and source-NPZ samples for the same date/code universe.
    raw_df = _build_raw_sample(list(dates), dict(selected_codes))
    npz_df = _build_npz_sample(list(dates), dict(selected_codes))

    # Evaluate baseline IC under all-row and seq60 endpoint scopes.
    rows = []
    for source_name, df in [("raw_feather", raw_df), ("source_npz", npz_df)]:
        scoped_frames = [
            ("all_rows", df),
            ("seq60_endpoint", df.loc[df["is_seq60_endpoint"].to_numpy(dtype=bool)].copy()),
        ]
        for scope_name, scoped_df in scoped_frames:
            for baseline_name, weights in dict(BASELINES).items():
                rows.append(_baseline_summary(str(source_name), str(scope_name), str(baseline_name), dict(weights), scoped_df))

    # Persist detailed samples and compact summary artifacts.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_DIR / "multiday_baseline_summary.csv", index=False)
    payload = {
        "config": {
            "dates": [int(d) for d in list(dates)],
            "date_count": int(len(dates)),
            "stocks_per_date": int(STOCKS_PER_DATE),
            "horizon_minutes": int(HORIZON_MINUTES),
            "window_size": int(WINDOW_SIZE),
        },
        "rows": rows,
    }
    (OUT_DIR / "multiday_baseline_summary.yaml").write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _list_dates(start_date: int, end_date: int, count: int) -> list[int]:
    """List the first available trade dates in the requested range."""
    # Scan stock1m feather files and keep dates inside the requested interval.
    dates: list[int] = []
    for year_dir in sorted(STOCK1M_DIR.glob("*")):
        if not year_dir.is_dir():
            continue
        for path in sorted(year_dir.glob("*.feather")):
            trade_date = int(path.stem)
            if int(start_date) <= int(trade_date) <= int(end_date):
                dates.append(int(trade_date))
            if len(dates) >= int(count):
                return list(dates)

    # Require enough dates so the test has the intended statistical width.
    if len(dates) < int(count):
        raise RuntimeError(f"Not enough dates: got={len(dates)} need={count}")
    return list(dates)


def _select_codes_by_date(dates: list[int], stocks_per_date: int) -> dict[int, list[int]]:
    """Select a deterministic code universe for each date."""
    # Use the first stock codes present in each raw date file.
    selected: dict[int, list[int]] = {}
    for trade_date in list(dates):
        path = STOCK1M_DIR / str(int(trade_date) // 10000) / f"{int(trade_date)}.feather"
        df = pd.read_feather(path, columns=["StockCode"])
        codes = sorted(int(code) for code in df["StockCode"].drop_duplicates().to_numpy())
        selected[int(trade_date)] = list(codes[: int(stocks_per_date)])
    return selected


def _build_raw_sample(dates: list[int], selected_codes: dict[int, list[int]]) -> pd.DataFrame:
    """Build baseline rows by recomputing features and labels from raw feather files."""
    # Recompute one date at a time to keep peak memory bounded.
    parts: list[pd.DataFrame] = []
    for trade_date in list(dates):
        day = _read_raw_day(int(trade_date), list(selected_codes[int(trade_date)]))
        parts.append(day)

    # Concatenate all days into one evaluation sample.
    out = pd.concat(parts, axis=0, ignore_index=True)
    out["source"] = "raw_feather"
    return out


def _read_raw_day(trade_date: int, codes: list[int]) -> pd.DataFrame:
    """Read one raw day and recompute baseline features plus forward labels."""
    # Load raw bars for selected stocks and sort exactly like data prep.
    path = STOCK1M_DIR / str(int(trade_date) // 10000) / f"{int(trade_date)}.feather"
    cols = ["StockCode", "DateTime", "Open", "Close", "High", "Low", "Vol", "Amount", "Date", "MinuteIndex"]
    df = pd.read_feather(path, columns=cols)
    df = df.loc[df["StockCode"].astype(np.int64).isin([int(code) for code in list(codes)])].copy()
    df = df.sort_values(["StockCode", "DateTime"], kind="stable").reset_index(drop=True)

    # Compute past returns from current and previous closes.
    log_close = np.log(df["Close"].to_numpy(dtype=np.float64))
    df["log_close"] = log_close
    grouped = df.groupby("StockCode", sort=False)
    df["ret_1m_raw"] = grouped["log_close"].diff(1)
    df["ret_5m_raw"] = grouped["log_close"].diff(5)
    df["ret_10m_raw"] = grouped["log_close"].diff(10)

    # Compute forward return by joining t+h close back to t.
    keys = ["StockCode", "Date", "MinuteIndex"]
    fwd = df.loc[:, keys + ["log_close"]].copy()
    fwd["MinuteIndex"] = fwd["MinuteIndex"].to_numpy(dtype=np.int64) - int(HORIZON_MINUTES)
    fwd = fwd.rename(columns={"log_close": "label_log_close_fwd"})
    df = df.merge(fwd, on=keys, how="left", sort=False, validate="m:1")
    df["target"] = df["label_log_close_fwd"] - df["log_close"]

    # Apply the same monotonic return transform used before zscore in data prep.
    feature_zscore = _load_feature_zscore()
    for feature_name in list(FEATURES):
        raw_col = f"{feature_name}_raw"
        transformed = _signed_log1p(df[raw_col].to_numpy(dtype=np.float64))
        mean, std = feature_zscore[str(feature_name)]
        df[feature_name] = (transformed - float(mean)) / float(std)

    # Keep rows with finite features and label inside the horizon-valid interval.
    keep_cols = ["StockCode", "Date", "MinuteIndex", "DateTime"] + list(FEATURES) + ["target"]
    out = df.loc[:, keep_cols].copy()
    finite = np.isfinite(out["target"].to_numpy(dtype=np.float64))
    finite &= out["MinuteIndex"].to_numpy(dtype=np.int64) <= int(239 - HORIZON_MINUTES)
    for feature_name in list(FEATURES):
        finite &= np.isfinite(out[feature_name].to_numpy(dtype=np.float64))
    out = out.loc[finite].reset_index(drop=True)

    # Mark sequence-valid endpoints using the same trailing-window length as the NN.
    out["is_seq60_endpoint"] = _mark_seq_endpoint(out, int(WINDOW_SIZE))
    return out


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    """Apply signed log1p to return-like values."""
    # Use the same monotonic transform as data prep for return features.
    return np.sign(values) * np.log1p(np.abs(values))


def _load_feature_zscore() -> dict[str, tuple[float, float]]:
    """Load source feature zscore parameters for baseline features."""
    # Parse the small pooled-zscore sidecar and index it by feature name.
    stats_path = SOURCE_NPZ_DIR.parent / "data_clean" / "pooled_zscore.yaml"
    payload = yaml.safe_load(stats_path.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for row in list(payload["features"]):
        out[str(row["feature"])] = (float(row["mean"]), float(row["std"]))
    return out


def _mark_seq_endpoint(df: pd.DataFrame, window_size: int) -> np.ndarray:
    """Mark rows whose stock-day history is long enough for one sequence endpoint."""
    # Compute within-stock-date row number after feature/label filtering.
    run_pos = df.groupby(["StockCode", "Date"], sort=False).cumcount().to_numpy(dtype=np.int64)
    return run_pos >= int(window_size) - 1


def _build_npz_sample(dates: list[int], selected_codes: dict[int, list[int]]) -> pd.DataFrame:
    """Build baseline rows from the source test NPZ for the same date/code universe."""
    # Memory-map source test arrays.
    x = np.memmap(SOURCE_NPZ_DIR / "test_x.f32", mode="r", dtype=np.float32, shape=(int(SOURCE_TEST_ROWS), int(SOURCE_TEST_FEATURE_DIM)))
    y = np.memmap(SOURCE_NPZ_DIR / "test_y.f32", mode="r", dtype=np.float32, shape=(int(SOURCE_TEST_ROWS), 1))
    meta = np.memmap(SOURCE_NPZ_DIR / "test_meta.i64", mode="r", dtype=np.int64, shape=(int(SOURCE_TEST_ROWS), 3))

    # Scan chunks until all requested dates have been passed.
    date_yy = np.asarray([int(date) % 1000000 for date in list(dates)], dtype=np.int64)
    max_date = int(date_yy.max())
    selected_code_sets = {int(date): set(int(code) for code in codes) for date, codes in dict(selected_codes).items()}
    parts: list[pd.DataFrame] = []
    chunk_rows = 2_000_000
    for start in range(0, int(SOURCE_TEST_ROWS), int(chunk_rows)):
        stop = min(int(SOURCE_TEST_ROWS), int(start + chunk_rows))
        meta_block = np.asarray(meta[int(start) : int(stop)], dtype=np.int64)
        if int(meta_block[0, 1]) > int(max_date):
            break
        date_mask = np.isin(meta_block[:, 1], date_yy)
        if not bool(date_mask.any()):
            continue
        idx_local = np.where(date_mask)[0]
        keep_local = []
        for local_idx in idx_local:
            yy = int(meta_block[int(local_idx), 1])
            full_date = 20000000 + int(yy)
            code = int(meta_block[int(local_idx), 0])
            if code in selected_code_sets[int(full_date)]:
                keep_local.append(int(local_idx))
        if len(keep_local) == 0:
            continue
        keep = np.asarray(keep_local, dtype=np.int64)
        parts.append(_npz_block_to_df(x[int(start) + keep], y[int(start) + keep, 0], meta_block[keep]))

    # Concatenate rows and mark sequence endpoints after selected-row filtering.
    out = pd.concat(parts, axis=0, ignore_index=True)
    out["Date"] = 20000000 + out["date"].astype(np.int64)
    out["is_seq60_endpoint"] = _mark_seq_endpoint(out.rename(columns={"code": "StockCode"}), int(WINDOW_SIZE))
    out["source"] = "source_npz"
    return out


def _npz_block_to_df(x_block: np.ndarray, y_block: np.ndarray, meta_block: np.ndarray) -> pd.DataFrame:
    """Convert selected NPZ rows into the baseline dataframe schema."""
    # Pull only return features needed by the simple baselines.
    data = {
        "code": meta_block[:, 0].astype(np.int64, copy=False),
        "date": meta_block[:, 1].astype(np.int64, copy=False),
        "time": meta_block[:, 2].astype(np.int64, copy=False),
        "target": y_block.astype(np.float64, copy=False),
    }
    for feature_name in list(FEATURES):
        data[feature_name] = x_block[:, int(FEATURE_NAMES.index(str(feature_name)))].astype(np.float64, copy=False)
    return pd.DataFrame(data)


def _baseline_summary(source_name: str, scope_name: str, baseline_name: str, weights: dict[str, float], df: pd.DataFrame) -> dict[str, object]:
    """Compute pooled and daily IC for one baseline row."""
    # Build the weighted score and target arrays.
    score = np.zeros((int(df.shape[0]),), dtype=np.float64)
    for feature_name, weight in dict(weights).items():
        score += float(weight) * df[str(feature_name)].to_numpy(dtype=np.float64)
    target = df["target"].to_numpy(dtype=np.float64)

    # Compute pooled and daily cross-sectional IC summaries.
    pooled_ic = _pearson(score, target)
    daily = _daily_summary(df, score, target)
    return {
        "source": str(source_name),
        "scope": str(scope_name),
        "baseline": str(baseline_name),
        "rows": int(df.shape[0]),
        "pooled_ic": float(pooled_ic),
        "daily_ic_mean": float(daily["mean"]),
        "daily_ic_std": float(daily["std"]),
        "daily_count": int(daily["count"]),
        "daily_positive_ratio": float(daily["positive_ratio"]),
    }


def _daily_summary(df: pd.DataFrame, score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute daily mean of minute cross-sectional IC values."""
    # Normalize date/time columns across raw and NPZ samples.
    work = pd.DataFrame(
        {
            "date": df["Date"].to_numpy(dtype=np.int64) if "Date" in df.columns else 20000000 + df["date"].to_numpy(dtype=np.int64),
            "time": df["MinuteIndex"].to_numpy(dtype=np.int64) if "MinuteIndex" in df.columns else df["time"].to_numpy(dtype=np.int64),
            "score": score,
            "target": target,
        }
    )

    # Average per-minute IC within each date.
    daily_values: list[float] = []
    for _date, day_df in work.groupby("date", sort=False):
        minute_values: list[float] = []
        for (_d, _t), minute_df in day_df.groupby(["date", "time"], sort=False):
            minute_values.append(_pearson(minute_df["score"].to_numpy(dtype=np.float64), minute_df["target"].to_numpy(dtype=np.float64)))
        vals = np.asarray(minute_values, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if int(vals.shape[0]) > 0:
            daily_values.append(float(vals.mean(dtype=np.float64)))

    # Convert the daily vector into a compact scalar summary.
    arr = np.asarray(daily_values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "count": int(arr.shape[0]),
        "positive_ratio": float(np.mean(arr > 0.0)) if int(arr.shape[0]) > 0 else float("nan"),
    }


if __name__ == "__main__":
    main()
