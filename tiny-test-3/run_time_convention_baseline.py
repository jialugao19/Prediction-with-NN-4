"""Check label timing conventions with raw multi-day baseline variants."""

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


from common import SOURCE_NPZ_DIR, _pearson
from run_multiday_raw_baseline import STOCK1M_DIR, _list_dates, _select_codes_by_date


OUT_DIR = Path("/data-cache/nn/tiny-test-3/time_convention_baseline_202408")
START_DATE = 20240801
END_DATE = 20240831
DATE_COUNT = 20
STOCKS_PER_DATE = 300
HORIZON_MINUTES = 10
SOURCE_TEST_ROWS = 241788800
SOURCE_TEST_FEATURE_DIM = 14
BASELINES = ["ret_1m", "ret_5m", "ret_10m", "ret_combo"]


def main() -> None:
    """Run label and time-convention baseline checks."""
    # Build the shared date/code universe.
    dates = _list_dates(int(START_DATE), int(END_DATE), int(DATE_COUNT))
    selected_codes = _select_codes_by_date(list(dates), int(STOCKS_PER_DATE))

    # Build raw rows with current and lagged returns plus multiple target definitions.
    raw_df = _build_raw_convention_sample(list(dates), dict(selected_codes))

    # Check that source NPZ labels match convention A on the same rows.
    label_check = _check_npz_label_matches_current_target(raw_df, list(dates), dict(selected_codes))

    # Compute A/B/C convention summaries.
    rows: list[dict[str, object]] = []
    for convention in ["A_current_close_entry", "B_next_close_entry", "C_previous_bar_features"]:
        for baseline_name in list(BASELINES):
            rows.append(_summarize_convention(raw_df, str(convention), str(baseline_name)))

    # Persist and print the result payload.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_DIR / "time_convention_baseline_summary.csv", index=False)
    payload = {
        "config": {
            "dates": [int(d) for d in list(dates)],
            "date_count": int(len(dates)),
            "stocks_per_date": int(STOCKS_PER_DATE),
            "horizon_minutes": int(HORIZON_MINUTES),
        },
        "label_check": dict(label_check),
        "rows": rows,
    }
    (OUT_DIR / "time_convention_baseline_summary.yaml").write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _build_raw_convention_sample(dates: list[int], selected_codes: dict[int, list[int]]) -> pd.DataFrame:
    """Build raw rows with A/B/C scores and labels."""
    # Recompute each date from raw bars and concatenate.
    parts: list[pd.DataFrame] = []
    for trade_date in list(dates):
        parts.append(_read_one_raw_convention_day(int(trade_date), list(selected_codes[int(trade_date)])))
    return pd.concat(parts, axis=0, ignore_index=True)


def _read_one_raw_convention_day(trade_date: int, codes: list[int]) -> pd.DataFrame:
    """Read one day and compute timing convention fields."""
    # Load selected stocks from the raw minute-bar file.
    path = STOCK1M_DIR / str(int(trade_date) // 10000) / f"{int(trade_date)}.feather"
    cols = ["StockCode", "DateTime", "Close", "Date", "MinuteIndex"]
    df = pd.read_feather(path, columns=cols)
    df = df.loc[df["StockCode"].astype(np.int64).isin([int(code) for code in list(codes)])].copy()
    df = df.sort_values(["StockCode", "DateTime"], kind="stable").reset_index(drop=True)

    # Compute log close and past-return score families.
    df["log_close"] = np.log(df["Close"].to_numpy(dtype=np.float64))
    grouped = df.groupby("StockCode", sort=False)
    for k in [1, 5, 10]:
        df[f"ret_{k}m_current"] = grouped["log_close"].diff(int(k))
        df[f"ret_{k}m_previous"] = grouped["log_close"].shift(1) - grouped["log_close"].shift(int(k) + 1)

    # Compute close[t+1] and close[t+10] labels by minute-index join.
    df = _attach_forward_log_close(df, 1, "log_close_t_plus_1")
    df = _attach_forward_log_close(df, int(HORIZON_MINUTES), "log_close_t_plus_h")
    df["target_A_current_close_entry"] = df["log_close_t_plus_h"] - df["log_close"]
    df["target_B_next_close_entry"] = df["log_close_t_plus_h"] - df["log_close_t_plus_1"]
    df["target_C_previous_bar_features"] = df["target_A_current_close_entry"]

    # Keep rows where all A/B/C fields needed for comparison are finite.
    out = df.loc[:, ["StockCode", "Date", "MinuteIndex"]].copy()
    for k in [1, 5, 10]:
        out[f"ret_{k}m_current"] = df[f"ret_{k}m_current"]
        out[f"ret_{k}m_previous"] = df[f"ret_{k}m_previous"]
    for name in ["target_A_current_close_entry", "target_B_next_close_entry", "target_C_previous_bar_features"]:
        out[str(name)] = df[str(name)]

    # Drop rows that cannot support every convention.
    mask = np.ones((int(out.shape[0]),), dtype=bool)
    mask &= out["MinuteIndex"].to_numpy(dtype=np.int64) <= int(239 - HORIZON_MINUTES)
    for col in list(out.columns):
        if str(col) in {"StockCode", "Date", "MinuteIndex"}:
            continue
        mask &= np.isfinite(out[str(col)].to_numpy(dtype=np.float64))
    out = out.loc[mask].reset_index(drop=True)
    return out


def _attach_forward_log_close(df: pd.DataFrame, horizon: int, out_col: str) -> pd.DataFrame:
    """Attach log_close[t+horizon] to each row by stock/date/minute join."""
    # Shift the forward table backward so it joins onto the current minute.
    keys = ["StockCode", "Date", "MinuteIndex"]
    fwd = df.loc[:, keys + ["log_close"]].copy()
    fwd["MinuteIndex"] = fwd["MinuteIndex"].to_numpy(dtype=np.int64) - int(horizon)
    fwd = fwd.rename(columns={"log_close": str(out_col)})
    return df.merge(fwd, on=keys, how="left", sort=False, validate="m:1")


def _check_npz_label_matches_current_target(raw_df: pd.DataFrame, dates: list[int], selected_codes: dict[int, list[int]]) -> dict[str, object]:
    """Compare source NPZ test labels with raw target A on the same universe."""
    # Build an index of raw target A values.
    raw_key = raw_df.loc[:, ["StockCode", "Date", "MinuteIndex", "target_A_current_close_entry"]].copy()
    raw_key = raw_key.rename(columns={"StockCode": "code", "target_A_current_close_entry": "raw_target_A"})

    # Load NPZ rows for the same date/code universe.
    npz_df = _load_matching_npz_targets(list(dates), dict(selected_codes))
    merged = npz_df.merge(raw_key, on=["code", "Date", "MinuteIndex"], how="inner", sort=False, validate="1:1")
    diff = merged["npz_target"].to_numpy(dtype=np.float64) - merged["raw_target_A"].to_numpy(dtype=np.float64)

    # Summarize label agreement.
    abs_diff = np.abs(diff)
    return {
        "matched_rows": int(merged.shape[0]),
        "raw_rows": int(raw_df.shape[0]),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean(dtype=np.float64)),
        "corr_npz_raw_target_A": float(_pearson(merged["npz_target"].to_numpy(dtype=np.float64), merged["raw_target_A"].to_numpy(dtype=np.float64))),
    }


def _load_matching_npz_targets(dates: list[int], selected_codes: dict[int, list[int]]) -> pd.DataFrame:
    """Load source NPZ targets for the same date/code universe."""
    # Read label zscore parameters so normalized y can be mapped back to raw returns.
    label_stats = yaml.safe_load((SOURCE_NPZ_DIR.parent / "data_clean" / "label_zscore.yaml").read_text(encoding="utf-8"))
    label_mean = float(label_stats["label"]["mean"])
    label_std = float(label_stats["label"]["std"])

    # Memory-map source test metadata and targets.
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
        target = y[int(start) + keep, 0].astype(np.float64) * float(label_std) + float(label_mean)
        parts.append(
            pd.DataFrame(
                {
                    "code": meta_block[keep, 0].astype(np.int64, copy=False),
                    "Date": 20000000 + meta_block[keep, 1].astype(np.int64, copy=False),
                    "time": meta_block[keep, 2].astype(np.int64, copy=False),
                    "npz_target": target,
                }
            )
        )

    # Convert time hhmmss back to minute index for joining with raw rows.
    out = pd.concat(parts, axis=0, ignore_index=True)
    out["MinuteIndex"] = _time_int_to_minute_index(out["time"].to_numpy(dtype=np.int64))
    return out.drop(columns=["time"])


def _time_int_to_minute_index(time_int: np.ndarray) -> np.ndarray:
    """Convert hhmmss integer time to stock1m MinuteIndex."""
    # Map morning and afternoon sessions into [0, 239].
    hour = time_int // 10000
    minute = (time_int // 100) % 100
    minute_of_day = hour * 60 + minute
    out = np.empty_like(minute_of_day, dtype=np.int64)
    morning = minute_of_day < 12 * 60
    out[morning] = minute_of_day[morning] - (9 * 60 + 30)
    out[~morning] = 120 + minute_of_day[~morning] - (13 * 60)
    return out


def _summarize_convention(df: pd.DataFrame, convention: str, baseline_name: str) -> dict[str, object]:
    """Compute one convention-baseline IC summary."""
    # Select score and target columns for this convention.
    score = _score_for_baseline(df, str(convention), str(baseline_name))
    target = df[f"target_{str(convention)}"].to_numpy(dtype=np.float64)
    daily = _daily_summary(df, score, target)
    return {
        "convention": str(convention),
        "baseline": str(baseline_name),
        "rows": int(df.shape[0]),
        "pooled_ic": float(_pearson(score, target)),
        "daily_ic_mean": float(daily["mean"]),
        "daily_ic_std": float(daily["std"]),
        "daily_count": int(daily["count"]),
        "daily_positive_ratio": float(daily["positive_ratio"]),
    }


def _score_for_baseline(df: pd.DataFrame, convention: str, baseline_name: str) -> np.ndarray:
    """Return the score vector for one baseline under one convention."""
    # Use previous-bar features only for convention C.
    suffix = "previous" if str(convention) == "C_previous_bar_features" else "current"
    if str(baseline_name) == "ret_combo":
        score = df[f"ret_1m_{suffix}"].to_numpy(dtype=np.float64)
        score += 0.5 * df[f"ret_5m_{suffix}"].to_numpy(dtype=np.float64)
        score += 0.25 * df[f"ret_10m_{suffix}"].to_numpy(dtype=np.float64)
        return score
    return df[f"{str(baseline_name)}_{suffix}"].to_numpy(dtype=np.float64)


def _daily_summary(df: pd.DataFrame, score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute daily mean of minute cross-sectional IC values."""
    # Build a compact work table for date/minute grouping.
    work = pd.DataFrame(
        {
            "Date": df["Date"].to_numpy(dtype=np.int64),
            "MinuteIndex": df["MinuteIndex"].to_numpy(dtype=np.int64),
            "score": score,
            "target": target,
        }
    )

    # Average minute-level cross-sectional IC within each day.
    daily_values: list[float] = []
    for _date, day_df in work.groupby("Date", sort=False):
        minute_values: list[float] = []
        for (_d, _minute), minute_df in day_df.groupby(["Date", "MinuteIndex"], sort=False):
            minute_values.append(_pearson(minute_df["score"].to_numpy(dtype=np.float64), minute_df["target"].to_numpy(dtype=np.float64)))
        vals = np.asarray(minute_values, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if int(vals.shape[0]) > 0:
            daily_values.append(float(vals.mean(dtype=np.float64)))
    arr = np.asarray(daily_values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "count": int(arr.shape[0]),
        "positive_ratio": float(np.mean(arr > 0.0)) if int(arr.shape[0]) > 0 else float("nan"),
    }


if __name__ == "__main__":
    main()
