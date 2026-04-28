"""Build point-in-time feature and execution inputs for portfolio backtests."""

from __future__ import annotations

from collections import OrderedDict, deque
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from portfolio_backtest.contract import (
    PortfolioBacktestConfig,
    ensure_output_dir,
    load_chunk_manifest_glob,
    load_manifest_chunk_paths,
    load_manifest_glob,
    write_chunk_manifest,
)


def short_date_to_full_date(short_date: int) -> int:
    """Convert one YYMMDD date into YYYYMMDD date."""
    # Reconstruct the full calendar date.
    return 20000000 + int(short_date)


def full_date_to_stock1m_path(raw_root: Path, full_date: int) -> Path:
    """Resolve one stock1m feather path from YYYYMMDD date."""
    # Build the year folder and daily filename.
    year = int(full_date) // 10000
    return Path(raw_root) / str(year) / f"{int(full_date)}.feather"


def full_date_to_stock1d_path(raw_root: Path, full_date: int) -> Path:
    """Resolve one stock1d feather path from YYYYMMDD date."""
    # Build the year folder and daily filename.
    year = int(full_date) // 10000
    return Path(raw_root) / str(year) / f"{int(full_date)}.feather"


def load_symbol_mapping(stock_basic_path: Path) -> pd.DataFrame:
    """Load the symbol<->ts_code mapping needed for daily limits."""
    # Read the stock basic table and normalize symbol into int.
    basic = pd.read_csv(stock_basic_path, usecols=["ts_code", "symbol"])
    basic = basic.dropna(subset=["symbol"]).copy()
    basic["symbol"] = basic["symbol"].astype(np.int32, copy=False)
    basic = basic.rename(columns={"symbol": "code"})
    return basic.loc[:, ["code", "ts_code"]].drop_duplicates().reset_index(drop=True)


def load_st_periods(namechange_path: Path, ts_code_to_code: pd.DataFrame) -> pd.DataFrame:
    """Load ST name periods as (code,start,end) rows."""
    # Read the name change history and keep only ST rows.
    raw = pd.read_csv(namechange_path, usecols=["ts_code", "name", "start_date", "end_date"])
    is_st = raw["name"].astype(str).str.contains("ST", regex=False)
    st = raw.loc[is_st].copy()

    # Normalize the start/end date fields and attach numeric codes.
    st["start_date"] = st["start_date"].astype(np.int32, copy=False)
    st["end_date"] = st["end_date"].fillna(99991231).astype(np.int32, copy=False)
    st = st.merge(ts_code_to_code, on="ts_code", how="inner")
    return st.loc[:, ["code", "start_date", "end_date"]].drop_duplicates().reset_index(drop=True)


def st_codes_for_date(st_periods: pd.DataFrame, full_date: int) -> set[int]:
    """Resolve the set of ST codes active on one trade date."""
    # Filter to active ST intervals and materialize the code set.
    active = st_periods.loc[(st_periods["start_date"] <= int(full_date)) & (st_periods["end_date"] >= int(full_date))]
    return set(active["code"].astype(int).tolist())


def load_daily_limits(stock1d_root: Path, ts_code_to_code: pd.DataFrame, full_date: int) -> pd.DataFrame:
    """Load daily limit rows and same-day amount for one date."""
    # Read the daily OHLC table for limits and amount.
    daily_path = full_date_to_stock1d_path(stock1d_root, full_date)
    daily = pd.read_feather(daily_path, columns=["ts_code", "up_limit", "down_limit", "amount"])

    # Attach numeric code and rename amount into a same-day audit field.
    daily = daily.merge(ts_code_to_code, on="ts_code", how="inner")
    daily = daily.rename(columns={"amount": "daily_amount"})
    daily["daily_amount"] = daily["daily_amount"].astype(np.float64, copy=False)
    return daily.loc[:, ["code", "up_limit", "down_limit", "daily_amount"]].drop_duplicates().reset_index(drop=True)


def build_trailing_adv_frame(adv_history: dict[int, deque[float]]) -> pd.DataFrame:
    """Build one ex-ante trailing ADV table from cached prior-day history."""
    # Aggregate the cached amount history into one mean ADV per code.
    rows: list[dict[str, float]] = []
    for code, history in adv_history.items():
        if len(history) == 0:
            continue
        rows.append({"code": float(code), "adv_amount": float(np.mean(np.asarray(history, dtype=np.float64)))})

    # Materialize the trailing ADV frame with stable dtypes.
    if len(rows) == 0:
        return pd.DataFrame(columns=["code", "adv_amount"])
    out = pd.DataFrame(rows)
    out["code"] = out["code"].astype(np.int32, copy=False)
    out["adv_amount"] = out["adv_amount"].astype(np.float64, copy=False)
    return out


def update_adv_history(adv_history: dict[int, deque[float]], daily_limits: pd.DataFrame, adv_lookback_days: int) -> None:
    """Update the trailing ADV cache with one completed trade date."""
    # Append the current-day amount into each code's bounded history.
    for row in daily_limits.loc[:, ["code", "daily_amount"]].itertuples(index=False):
        code = int(row.code)
        daily_amount = float(row.daily_amount)
        if not np.isfinite(daily_amount) or daily_amount <= 0.0:
            continue
        if code not in adv_history:
            adv_history[code] = deque(maxlen=int(adv_lookback_days))
        adv_history[code].append(daily_amount)


def load_daily_market_state(
    stock1m_root: Path,
    full_date: int,
    daily_limits: pd.DataFrame,
    trailing_adv: pd.DataFrame,
    st_codes: set[int],
    sigma_lookback_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one day stock1m data and build next-trade lookups."""
    # Load the raw stock1m columns needed for the portfolio backtest.
    day_path = full_date_to_stock1m_path(stock1m_root, full_date)
    raw = pd.read_feather(
        day_path,
        columns=["StockCode", "DateTime", "MinuteIndex", "Open", "Close", "High", "Low", "Vol", "Amount"],
    )
    raw = raw.sort_values(["StockCode", "MinuteIndex"]).reset_index(drop=True)

    # Build the integer time and minute-slot fields used by the strategy.
    raw["time"] = (
        raw["DateTime"].dt.hour.astype(np.int32) * 10000
        + raw["DateTime"].dt.minute.astype(np.int32) * 100
        + raw["DateTime"].dt.second.astype(np.int32)
    )
    raw["minute_slot"] = ((raw["time"] // 100) % 100).astype(np.int32) % 10

    # Build a per-bar VWAP proxy from amount and volume.
    raw["vwap"] = raw["Amount"].astype(np.float64) / raw["Vol"].astype(np.float64)

    # Attach daily limits and ex-ante ADV proxies.
    raw = raw.merge(daily_limits, left_on="StockCode", right_on="code", how="left")
    raw = raw.drop(columns=["code"])
    raw = raw.merge(trailing_adv, left_on="StockCode", right_on="code", how="left")
    raw = raw.drop(columns=["code"], errors="ignore")

    # Attach the day-level ST flag and current tradability filter.
    raw["is_st"] = raw["StockCode"].astype(np.int32).isin(st_codes)
    raw["tradable_base"] = (
        (raw["Open"].astype(np.float64) > 0.0)
        & (raw["Vol"].astype(np.float64) > 0.0)
        & (raw["Amount"].astype(np.float64) > 0.0)
        & (~raw["is_st"])
    )
    raw["is_one_price_limit"] = (
        (raw["High"].astype(np.float64) == raw["Low"].astype(np.float64))
        & (
            (raw["High"].astype(np.float64) == raw["up_limit"].astype(np.float64))
            | (raw["Low"].astype(np.float64) == raw["down_limit"].astype(np.float64))
        )
    )
    raw["current_tradable"] = raw["tradable_base"] & (~raw["is_one_price_limit"])

    # Build per-code "all day locked limit" flags from 1m bars for direction-aware execution constraints.
    bar_is_up_locked = (raw["High"].astype(np.float64) == raw["up_limit"].astype(np.float64)) & (
        raw["Low"].astype(np.float64) == raw["up_limit"].astype(np.float64)
    )
    bar_is_down_locked = (raw["High"].astype(np.float64) == raw["down_limit"].astype(np.float64)) & (
        raw["Low"].astype(np.float64) == raw["down_limit"].astype(np.float64)
    )
    up_all_day = bar_is_up_locked.groupby(raw["StockCode"], sort=False).all()
    down_all_day = bar_is_down_locked.groupby(raw["StockCode"], sort=False).all()
    raw["is_limit_up_all_day"] = raw["StockCode"].map(up_all_day).astype(bool)
    raw["is_limit_down_all_day"] = raw["StockCode"].map(down_all_day).astype(bool)

    # Compute one ex-ante intraday sigma from trailing 1m returns.
    ret_1m = raw.groupby("StockCode", sort=False)["Close"].pct_change().astype(np.float64)
    raw["sigma_intraday"] = ret_1m.groupby(raw["StockCode"], sort=False).transform(
        lambda series: series.rolling(window=int(sigma_lookback_bars), min_periods=5).std(ddof=1)
    )

    # Prepare the next-trade lookups with tradable bars only.
    tradable_minute = raw["MinuteIndex"].where(raw["current_tradable"])
    tradable_open = raw["Open"].where(raw["current_tradable"])
    tradable_vwap = raw["vwap"].where(raw["current_tradable"])
    raw["next_trade_minute"] = tradable_minute.groupby(raw["StockCode"], sort=False).transform(
        lambda series: series.iloc[::-1].ffill().iloc[::-1]
    )
    raw["next_trade_open"] = tradable_open.groupby(raw["StockCode"], sort=False).transform(
        lambda series: series.iloc[::-1].ffill().iloc[::-1]
    )
    raw["next_trade_vwap"] = tradable_vwap.groupby(raw["StockCode"], sort=False).transform(
        lambda series: series.iloc[::-1].ffill().iloc[::-1]
    )

    # Materialize the signal-time universe slice.
    universe = raw.rename(
        columns={
            "StockCode": "code",
            "MinuteIndex": "base_minute",
            "Open": "signal_open",
            "Close": "signal_close",
            "Vol": "signal_vol",
            "Amount": "signal_amount",
        }
    )
    universe = universe.loc[
        :,
        [
            "code",
            "time",
            "minute_slot",
            "base_minute",
            "signal_open",
            "signal_close",
            "signal_vol",
            "signal_amount",
            "adv_amount",
            "current_tradable",
            "sigma_intraday",
            "up_limit",
            "down_limit",
            "is_limit_up_all_day",
            "is_limit_down_all_day",
        ],
    ].copy()

    # Materialize the next-trade execution lookup table.
    next_trade = raw.loc[:, ["StockCode", "MinuteIndex", "next_trade_minute", "next_trade_open", "next_trade_vwap"]].copy()
    next_trade = next_trade.rename(columns={"StockCode": "code", "MinuteIndex": "schedule_minute"})
    next_trade["code"] = next_trade["code"].astype(np.int32, copy=False)
    next_trade["schedule_minute"] = next_trade["schedule_minute"].astype(np.int32, copy=False)
    return universe, next_trade


def get_cached_daily_market_state(
    cache: OrderedDict[int, tuple[pd.DataFrame, pd.DataFrame]],
    stock1m_root: Path,
    full_date: int,
    daily_limits: pd.DataFrame,
    trailing_adv: pd.DataFrame,
    st_codes: set[int],
    lookup_cache_size: int,
    sigma_lookback_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch one daily market state from a small LRU cache."""
    # Reuse the cached day when possible.
    if int(full_date) in cache:
        state = cache.pop(int(full_date))
        cache[int(full_date)] = state
        return state

    # Load the missing day from disk and maintain the bounded cache.
    universe, next_trade = load_daily_market_state(
        stock1m_root,
        int(full_date),
        daily_limits,
        trailing_adv,
        st_codes,
        sigma_lookback_bars,
    )
    cache[int(full_date)] = (universe, next_trade)
    if len(cache) > int(lookup_cache_size):
        cache.popitem(last=False)
    return universe, next_trade


def build_feature_day_frame(
    pred_day: pd.DataFrame,
    universe: pd.DataFrame,
    next_trade: pd.DataFrame,
    allowed_times: set[int],
    entry_delay_bars: int,
    holding_bars: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build one date slice of features aligned with inference signals."""
    # Restrict the signal universe to model-supported times.
    signal_universe = universe.loc[universe["time"].isin(allowed_times)].copy()
    raw_universe_rows = int(signal_universe.shape[0])

    # Attach predictions to the signal universe with a left join.
    merged = signal_universe.merge(pred_day, on=["code", "time"], how="left")
    merged["date"] = int(pred_day["date"].iloc[0])
    merged["prediction_available"] = merged["prediction"].notna()
    prediction_rows = int(merged["prediction_available"].sum())

    # Attach the tradable entry execution after the signal.
    merged["entry_schedule_minute"] = merged["base_minute"].astype(np.int32) + int(entry_delay_bars)
    entry_lookup = next_trade.rename(
        columns={
            "schedule_minute": "entry_schedule_minute",
            "next_trade_minute": "entry_exec_minute",
            "next_trade_open": "entry_open",
            "next_trade_vwap": "entry_vwap",
        }
    )
    merged = merged.merge(entry_lookup, on=["code", "entry_schedule_minute"], how="left")

    # Attach the tradable exit execution after holding horizon.
    merged["exit_schedule_minute"] = merged["entry_exec_minute"].astype(np.float64) + float(holding_bars)
    merged["exit_schedule_minute"] = merged["exit_schedule_minute"].astype("Int64")
    exit_lookup = next_trade.rename(
        columns={
            "schedule_minute": "exit_schedule_minute",
            "next_trade_minute": "exit_exec_minute",
            "next_trade_open": "exit_open",
            "next_trade_vwap": "exit_vwap",
        }
    )
    merged = merged.merge(exit_lookup, on=["code", "exit_schedule_minute"], how="left")

    # Compute the tradable execution returns and fillability flags.
    merged["ret_open_exec_10"] = merged["exit_open"].astype(np.float64) / merged["entry_open"].astype(np.float64) - 1.0
    merged["ret_vwap_exec_10"] = merged["exit_vwap"].astype(np.float64) / merged["entry_vwap"].astype(np.float64) - 1.0
    merged["fillable_open"] = merged["entry_open"].notna() & merged["exit_open"].notna()
    merged["fillable_vwap"] = merged["entry_vwap"].notna() & merged["exit_vwap"].notna()

    # Build per-leg limit-touch flags so the simulator can apply direction-aware fill constraints.
    merged["entry_open_is_up_limit"] = merged["entry_open"].astype(np.float64) >= merged["up_limit"].astype(np.float64)
    merged["entry_open_is_down_limit"] = merged["entry_open"].astype(np.float64) <= merged["down_limit"].astype(np.float64)
    merged["exit_open_is_up_limit"] = merged["exit_open"].astype(np.float64) >= merged["up_limit"].astype(np.float64)
    merged["exit_open_is_down_limit"] = merged["exit_open"].astype(np.float64) <= merged["down_limit"].astype(np.float64)
    merged["entry_vwap_is_up_limit"] = merged["entry_vwap"].astype(np.float64) >= merged["up_limit"].astype(np.float64)
    merged["entry_vwap_is_down_limit"] = merged["entry_vwap"].astype(np.float64) <= merged["down_limit"].astype(np.float64)
    merged["exit_vwap_is_up_limit"] = merged["exit_vwap"].astype(np.float64) >= merged["up_limit"].astype(np.float64)
    merged["exit_vwap_is_down_limit"] = merged["exit_vwap"].astype(np.float64) <= merged["down_limit"].astype(np.float64)
    tradable_rows = int(merged["current_tradable"].sum())

    # Emit a compact audit payload for data retention checks.
    audit = {
        "date": float(merged["date"].iloc[0]),
        "raw_universe_rows": float(raw_universe_rows),
        "prediction_rows": float(prediction_rows),
        "current_tradable_rows": float(tradable_rows),
        "fillable_open_rows": float(merged["fillable_open"].sum()),
        "fillable_vwap_rows": float(merged["fillable_vwap"].sum()),
    }
    return merged, audit


def load_allowed_times(inference_manifest_path: Path) -> list[int]:
    """Load model-supported times from the inference parquet source."""
    # Read the parquet source once and collect distinct time values.
    parquet_glob = load_manifest_glob(inference_manifest_path)
    con = duckdb.connect()
    allowed_times = con.execute(
        f"""
        SELECT DISTINCT time
        FROM read_parquet('{parquet_glob}')
        ORDER BY time
        """
    ).fetchnumpy()["time"].tolist()
    con.close()
    return [int(value) for value in allowed_times]


def materialize_feature_chunks(config: PortfolioBacktestConfig) -> str:
    """Materialize feature-enriched parquet chunks for the portfolio backtest."""
    # Prepare the output directories and mapping tables.
    ensure_output_dir(config.output_dir)
    ensure_output_dir(config.feature_chunk_dir)

    # Reuse existing feature chunks when the manifest and audits are already on disk.
    if config.feature_manifest_path.exists() and (config.output_dir / "feature_audit.csv").exists() and (config.output_dir / "feature_audit.yaml").exists():
        return load_chunk_manifest_glob(config.feature_manifest_path)

    ts_code_to_code = load_symbol_mapping(config.stock_basic_path)
    st_periods = load_st_periods(config.namechange_path, ts_code_to_code)
    allowed_times = set(load_allowed_times(config.inference_manifest_path))

    # Prepare the per-day cache for stock1m reads.
    lookup_cache: OrderedDict[int, tuple[pd.DataFrame, pd.DataFrame]] = OrderedDict()
    adv_history: dict[int, deque[float]] = {}

    # Iterate over inference chunks with boundary-date carry handling.
    chunk_paths = load_manifest_chunk_paths(config.inference_manifest_path)
    output_chunk_paths: list[Path] = []
    audit_rows: list[dict[str, float]] = []
    carry_date: int | None = None
    carry_frame: pd.DataFrame | None = None
    for idx, chunk_path in enumerate(chunk_paths):
        # Load one prediction chunk and keep the minimal columns.
        print(f"[feature] {idx}/{len(chunk_paths)} -> {chunk_path.name}", flush=True)
        pred_chunk = pd.read_parquet(chunk_path, columns=["prediction", "code", "date", "time"])

        # Merge the carried boundary date before processing the chunk.
        if carry_date is not None:
            chunk_same_date = pred_chunk.loc[pred_chunk["date"] == carry_date].copy()
            pred_chunk = pred_chunk.loc[pred_chunk["date"] != carry_date].copy()
            pred_chunk = pd.concat([carry_frame, chunk_same_date, pred_chunk], axis=0, ignore_index=True)
            carry_date = None
            carry_frame = None

        # Split out the last date to avoid boundary duplication.
        last_date = int(pred_chunk["date"].max())
        carry_frame = pred_chunk.loc[pred_chunk["date"] == last_date].copy()
        carry_date = last_date
        pred_chunk = pred_chunk.loc[pred_chunk["date"] != last_date].copy()

        # Build the feature chunk day by day.
        day_frames: list[pd.DataFrame] = []
        for short_date, pred_day in pred_chunk.groupby("date", sort=True):
            full_date = short_date_to_full_date(int(short_date))
            st_codes = st_codes_for_date(st_periods, full_date)
            daily_limits = load_daily_limits(config.raw_stock1d_root, ts_code_to_code, full_date)
            trailing_adv = build_trailing_adv_frame(adv_history)
            universe, next_trade = get_cached_daily_market_state(
                lookup_cache,
                config.raw_stock1m_root,
                full_date,
                daily_limits,
                trailing_adv,
                st_codes,
                config.lookup_cache_size,
                config.sigma_lookback_bars,
            )
            feat_day, audit = build_feature_day_frame(
                pred_day.reset_index(drop=True),
                universe,
                next_trade,
                allowed_times,
                config.entry_delay_bars,
                config.holding_bars,
            )
            day_frames.append(feat_day)
            audit_rows.append(audit)
            update_adv_history(adv_history, daily_limits, config.adv_lookback_days)

        # Write the feature chunk.
        if len(day_frames) > 0:
            feat_chunk = pd.concat(day_frames, axis=0, ignore_index=True)
            output_chunk_path = config.feature_chunk_dir / chunk_path.name
            feat_chunk.to_parquet(output_chunk_path, index=False)
            output_chunk_paths.append(output_chunk_path)

    # Process the final carried date once.
    if carry_frame is None or carry_date is None:
        raise RuntimeError("Inference manifest must contain at least one date.")
    final_full_date = short_date_to_full_date(int(carry_date))
    final_st_codes = st_codes_for_date(st_periods, final_full_date)
    final_limits = load_daily_limits(config.raw_stock1d_root, ts_code_to_code, final_full_date)
    final_trailing_adv = build_trailing_adv_frame(adv_history)
    final_universe, final_next_trade = get_cached_daily_market_state(
        lookup_cache,
        config.raw_stock1m_root,
        final_full_date,
        final_limits,
        final_trailing_adv,
        final_st_codes,
        config.lookup_cache_size,
        config.sigma_lookback_bars,
    )
    final_feat_day, final_audit = build_feature_day_frame(
        carry_frame.reset_index(drop=True),
        final_universe,
        final_next_trade,
        allowed_times,
        config.entry_delay_bars,
        config.holding_bars,
    )
    final_output_path = config.feature_chunk_dir / f"part_{len(chunk_paths):06d}_tail.parquet"
    final_feat_day.to_parquet(final_output_path, index=False)
    output_chunk_paths.append(final_output_path)
    audit_rows.append(final_audit)

    # Write the manifest and audit outputs.
    audit_df = pd.DataFrame(audit_rows).sort_values("date").reset_index(drop=True)
    audit_df.to_csv(config.output_dir / "feature_audit.csv", index=False)
    (config.output_dir / "feature_audit.yaml").write_text(
        yaml.safe_dump(
            {
                "entry_delay_bars": int(config.entry_delay_bars),
                "holding_bars": int(config.holding_bars),
                "date_count": int(audit_df.shape[0]),
                "raw_universe_rows_total": float(audit_df["raw_universe_rows"].sum()),
                "prediction_rows_total": float(audit_df["prediction_rows"].sum()),
                "current_tradable_rows_total": float(audit_df["current_tradable_rows"].sum()),
                "fillable_open_rows_total": float(audit_df["fillable_open_rows"].sum()),
                "fillable_vwap_rows_total": float(audit_df["fillable_vwap_rows"].sum()),
                "prediction_coverage_ratio_total": float(audit_df["prediction_rows"].sum() / audit_df["raw_universe_rows"].sum()),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    write_chunk_manifest(
        config.feature_manifest_path,
        output_chunk_paths,
        config.feature_chunk_dir.name,
        [
            "prediction",
            "code",
            "date",
            "time",
            "minute_slot",
            "base_minute",
            "signal_open",
            "signal_close",
            "signal_vol",
            "signal_amount",
            "adv_amount",
            "current_tradable",
            "sigma_intraday",
            "up_limit",
            "down_limit",
            "is_limit_up_all_day",
            "is_limit_down_all_day",
            "entry_schedule_minute",
            "entry_exec_minute",
            "entry_open",
            "entry_vwap",
            "exit_schedule_minute",
            "exit_exec_minute",
            "exit_open",
            "exit_vwap",
            "ret_open_exec_10",
            "ret_vwap_exec_10",
            "fillable_open",
            "fillable_vwap",
            "entry_open_is_up_limit",
            "entry_open_is_down_limit",
            "exit_open_is_up_limit",
            "exit_open_is_down_limit",
            "entry_vwap_is_up_limit",
            "entry_vwap_is_down_limit",
            "exit_vwap_is_up_limit",
            "exit_vwap_is_down_limit",
            "prediction_available",
        ],
    )
    return load_chunk_manifest_glob(config.feature_manifest_path)
