"""Run the upgraded backtest2 research pipeline with non-overlap diagnostics and spread+impact costs."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Backtest2Config:
    """Store the fixed IO layout and upgraded backtest assumptions."""

    repo_root: Path
    predict_manifest_path: Path
    output_dir: Path
    feature_db_path: Path
    feature_chunk_dir: Path
    feature_manifest_path: Path
    raw_stock1m_root: Path
    raw_stock1d_root: Path
    stock_basic_path: Path
    namechange_path: Path
    top_frac: float
    entry_delay_bars: int
    holding_bars: int
    annual_days: int
    lookup_cache_size: int
    adv_lookback_days: int
    sigma_lookback_bars: int
    impact_eta: float
    spread_bps_high: float
    spread_bps_mid: float
    spread_bps_low: float
    aum_list: list[float]
    impact_budget_bps_list: list[float]
    report_title: str


def build_config() -> Backtest2Config:
    """Build the fixed backtest2 configuration."""
    # Define the fixed repo paths.
    repo_root = Path("/home/maomao/prediction-NN-2")
    predict_manifest_path = Path(
        "/data-cache/nn/upgrade_20260328_gru_seq60_h10/date_ranges/run/eval_test/iter_5000/predict_manifest.yaml"
    )

    # Define the backtest2 output layout.
    output_dir = Path("/data-cache/nn/0418/backtest2")
    feature_db_path = output_dir / "backtest2.duckdb"
    feature_chunk_dir = output_dir / "feature_chunks"
    feature_manifest_path = output_dir / "feature_manifest.yaml"

    # Define the market data inputs.
    raw_stock1m_root = Path("/data/ashare/market/stock1m")
    raw_stock1d_root = Path("/data/ashare/market/stock1d")
    stock_basic_path = Path("/data/ashare/market/stock_basic.csv")
    namechange_path = Path("/data/ashare/market/namechange.csv")

    # Define the selection, horizon, and reporting knobs.
    top_frac = 0.10
    entry_delay_bars = 1
    holding_bars = 10
    annual_days = 252
    lookup_cache_size = 6
    adv_lookback_days = 20
    sigma_lookback_bars = 20

    # Define the cost model knobs.
    impact_eta = 0.50
    spread_bps_high = 5.0
    spread_bps_mid = 10.0
    spread_bps_low = 20.0
    aum_list = [10_000_000.0, 50_000_000.0, 100_000_000.0]
    impact_budget_bps_list = [10.0, 20.0]
    report_title = "Backtest2: Portfolio Backtest Only (Signal Evaluation Delegated to eval_ic)"
    return Backtest2Config(
        repo_root=repo_root,
        predict_manifest_path=predict_manifest_path,
        output_dir=output_dir,
        feature_db_path=feature_db_path,
        feature_chunk_dir=feature_chunk_dir,
        feature_manifest_path=feature_manifest_path,
        raw_stock1m_root=raw_stock1m_root,
        raw_stock1d_root=raw_stock1d_root,
        stock_basic_path=stock_basic_path,
        namechange_path=namechange_path,
        top_frac=top_frac,
        entry_delay_bars=entry_delay_bars,
        holding_bars=holding_bars,
        annual_days=annual_days,
        lookup_cache_size=lookup_cache_size,
        adv_lookback_days=adv_lookback_days,
        sigma_lookback_bars=sigma_lookback_bars,
        impact_eta=impact_eta,
        spread_bps_high=spread_bps_high,
        spread_bps_mid=spread_bps_mid,
        spread_bps_low=spread_bps_low,
        aum_list=aum_list,
        impact_budget_bps_list=impact_budget_bps_list,
        report_title=report_title,
    )


def ensure_output_dir(path: Path) -> None:
    """Create one output directory."""
    # Create the directory tree.
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read one YAML manifest file."""
    # Load the manifest payload.
    return yaml.safe_load(manifest_path.read_text(encoding="utf-8"))


def load_manifest_glob(manifest_path: Path) -> str:
    """Resolve the parquet glob from a manifest YAML."""
    # Read the manifest once.
    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent

    # Reconstruct the chunk folder glob.
    first_chunk_parent = Path(manifest["chunk_files"][0]).parent.as_posix()
    return (manifest_dir / first_chunk_parent / "*.parquet").as_posix()


def load_manifest_chunk_paths(manifest_path: Path) -> list[Path]:
    """Resolve concrete parquet chunk paths from a manifest YAML."""
    # Read the manifest once.
    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent

    # Resolve each relative chunk path.
    return [manifest_dir / Path(chunk_file) for chunk_file in manifest["chunk_files"]]


def write_manifest(manifest_path: Path, chunk_paths: list[Path], chunk_dir_name: str) -> None:
    """Write one parquet-chunk manifest YAML."""
    # Build relative chunk filenames.
    chunk_files = [str(Path(chunk_dir_name) / path.name) for path in chunk_paths]
    payload = {"format": "parquet_chunks", "chunk_files": chunk_files}

    # Serialize as YAML for reproducibility.
    with manifest_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, allow_unicode=True, sort_keys=False)


def connect_duckdb(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open one DuckDB connection with stable settings."""
    # Open the local database file.
    con = duckdb.connect(str(db_path))

    # Configure deterministic execution knobs.
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA enable_progress_bar=false")
    return con


def short_date_to_full_date(short_date: int) -> int:
    """Convert one YYMMDD date into YYYYMMDD date."""
    # Reconstruct the full calendar date.
    return 20000000 + int(short_date)


def full_date_to_stock1m_path(raw_root: Path, full_date: int) -> Path:
    """Resolve one stock1m feather path from YYYYMMDD date."""
    # Build the year folder and daily filename.
    year = int(full_date) // 10000
    return raw_root / str(year) / f"{int(full_date)}.feather"


def full_date_to_stock1d_path(raw_root: Path, full_date: int) -> Path:
    """Resolve one stock1d feather path from YYYYMMDD date."""
    # Build the year folder and daily filename.
    year = int(full_date) // 10000
    return raw_root / str(year) / f"{int(full_date)}.feather"


def load_symbol_mapping(stock_basic_path: Path) -> pd.DataFrame:
    """Load the symbol<->ts_code mapping needed for daily limits."""
    # Read the stock basic table.
    basic = pd.read_csv(stock_basic_path, usecols=["ts_code", "symbol"])

    # Normalize the numeric symbol into int.
    basic = basic.dropna(subset=["symbol"]).copy()
    basic["symbol"] = basic["symbol"].astype(np.int32, copy=False)
    basic = basic.rename(columns={"symbol": "code"})
    return basic.loc[:, ["code", "ts_code"]].drop_duplicates().reset_index(drop=True)


def load_st_periods(namechange_path: Path, ts_code_to_code: pd.DataFrame) -> pd.DataFrame:
    """Load ST name periods as (code,start,end) rows."""
    # Read the name change history and keep only needed columns.
    raw = pd.read_csv(namechange_path, usecols=["ts_code", "name", "start_date", "end_date"])

    # Filter to rows that indicate ST naming.
    is_st = raw["name"].astype(str).str.contains("ST", regex=False)
    st = raw.loc[is_st].copy()

    # Normalize the start/end date fields.
    st["start_date"] = st["start_date"].astype(np.int32, copy=False)
    st["end_date"] = st["end_date"].fillna(99991231).astype(np.int32, copy=False)

    # Attach the numeric code for joining with stock1m data.
    st = st.merge(ts_code_to_code, on="ts_code", how="inner")
    st = st.loc[:, ["code", "start_date", "end_date"]].drop_duplicates().reset_index(drop=True)
    return st


def st_codes_for_date(st_periods: pd.DataFrame, full_date: int) -> set[int]:
    """Resolve the set of ST codes active on one trade date."""
    # Filter to active ST intervals.
    active = st_periods.loc[(st_periods["start_date"] <= int(full_date)) & (st_periods["end_date"] >= int(full_date))]
    return set(active["code"].astype(int).tolist())


def load_daily_limits(
    stock1d_root: Path,
    ts_code_to_code: pd.DataFrame,
    full_date: int,
) -> pd.DataFrame:
    """Load daily limit rows and same-day amount for one date."""
    # Read the daily OHLC table for limits and amount.
    daily_path = full_date_to_stock1d_path(stock1d_root, full_date)
    daily = pd.read_feather(daily_path, columns=["ts_code", "up_limit", "down_limit", "amount"])

    # Attach numeric code.
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


def update_adv_history(
    adv_history: dict[int, deque[float]],
    daily_limits: pd.DataFrame,
    adv_lookback_days: int,
) -> None:
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
    # Load the raw stock1m columns needed for backtest2.
    day_path = full_date_to_stock1m_path(stock1m_root, full_date)
    raw = pd.read_feather(
        day_path,
        columns=["StockCode", "DateTime", "MinuteIndex", "Open", "Close", "High", "Low", "Vol", "Amount"],
    )

    # Sort the raw bars by stock and time.
    raw = raw.sort_values(["StockCode", "MinuteIndex"]).reset_index(drop=True)

    # Build the integer `time` and `minute_slot` fields.
    raw["time"] = (
        raw["DateTime"].dt.hour.astype(np.int32) * 10000
        + raw["DateTime"].dt.minute.astype(np.int32) * 100
        + raw["DateTime"].dt.second.astype(np.int32)
    )
    raw["minute_slot"] = ((raw["time"] // 100) % 100).astype(np.int32) % 10

    # Build a per-bar VWAP proxy from amount/volume.
    raw["vwap"] = raw["Amount"].astype(np.float64) / raw["Vol"].astype(np.float64)

    # Attach daily limits and ex-ante ADV proxies.
    raw = raw.merge(daily_limits, left_on="StockCode", right_on="code", how="left")
    raw = raw.drop(columns=["code"])
    raw = raw.merge(trailing_adv, left_on="StockCode", right_on="code", how="left")
    raw = raw.drop(columns=["code"], errors="ignore")

    # Attach the day-level ST flag.
    raw["is_st"] = raw["StockCode"].astype(np.int32).isin(st_codes)

    # Define the baseline tradability filter at signal time.
    raw["tradable_base"] = (
        (raw["Open"].astype(np.float64) > 0.0)
        & (raw["Vol"].astype(np.float64) > 0.0)
        & (raw["Amount"].astype(np.float64) > 0.0)
        & (~raw["is_st"])
    )

    # Define the one-price limit bars using daily limits.
    raw["is_one_price_limit"] = (
        (raw["High"].astype(np.float64) == raw["Low"].astype(np.float64))
        & (
            (raw["High"].astype(np.float64) == raw["up_limit"].astype(np.float64))
            | (raw["Low"].astype(np.float64) == raw["down_limit"].astype(np.float64))
        )
    )

    # Define the tradability filter used by the strategy and tradable diagnostics.
    raw["current_tradable"] = raw["tradable_base"] & (~raw["is_one_price_limit"])

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

    # Load the missing day from disk.
    universe, next_trade = load_daily_market_state(
        stock1m_root,
        int(full_date),
        daily_limits,
        trailing_adv,
        st_codes,
        sigma_lookback_bars,
    )
    cache[int(full_date)] = (universe, next_trade)

    # Evict the oldest cached day.
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
    """Build one (date) slice of features aligned with prediction signals."""
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

    # Compute the tradable open/open forward return.
    merged["ret_open_exec_10"] = merged["exit_open"].astype(np.float64) / merged["entry_open"].astype(np.float64) - 1.0

    # Compute the tradable vwap/vwap forward return.
    merged["ret_vwap_exec_10"] = merged["exit_vwap"].astype(np.float64) / merged["entry_vwap"].astype(np.float64) - 1.0

    # Define the fillability flags without feeding them back into ranking.
    merged["fillable_open"] = merged["entry_open"].notna() & merged["exit_open"].notna()
    merged["fillable_vwap"] = merged["entry_vwap"].notna() & merged["exit_vwap"].notna()
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


def load_allowed_times(predict_manifest_path: Path) -> list[int]:
    """Load model-supported times from the prediction parquet source."""
    # Read the parquet source once.
    parquet_glob = load_manifest_glob(predict_manifest_path)
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


def materialize_feature_chunks(config: Backtest2Config) -> str:
    """Materialize feature-enriched parquet chunks for backtest2."""
    # Prepare the output directories.
    ensure_output_dir(config.output_dir)
    ensure_output_dir(config.feature_chunk_dir)

    # Load the mapping tables needed for ST and limits.
    ts_code_to_code = load_symbol_mapping(config.stock_basic_path)
    st_periods = load_st_periods(config.namechange_path, ts_code_to_code)
    allowed_times = set(load_allowed_times(config.predict_manifest_path))

    # Prepare the per-day cache for stock1m reads.
    lookup_cache: OrderedDict[int, tuple[pd.DataFrame, pd.DataFrame]] = OrderedDict()
    adv_history: dict[int, deque[float]] = {}

    # Iterate over prediction chunks with boundary-date carry handling.
    chunk_paths = load_manifest_chunk_paths(config.predict_manifest_path)
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
        if day_frames:
            feat_chunk = pd.concat(day_frames, axis=0, ignore_index=True)
            output_chunk_path = config.feature_chunk_dir / chunk_path.name
            feat_chunk.to_parquet(output_chunk_path, index=False)
            output_chunk_paths.append(output_chunk_path)

    # Process the final carried date once.
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
    with (config.output_dir / "feature_audit.yaml").open("w", encoding="utf-8") as file_obj:
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
            file_obj,
            allow_unicode=True,
            sort_keys=False,
        )
    write_manifest(config.feature_manifest_path, output_chunk_paths, config.feature_chunk_dir.name)
    return load_manifest_glob(config.feature_manifest_path)


def materialize_slot_strategy_table(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    top_frac: float,
    return_col: str,
    fillable_col: str,
    extra_filter_sql: str,
    spread_bps_high: float,
    spread_bps_mid: float,
    spread_bps_low: float,
    position_table_name: str,
) -> None:
    """Materialize one non-overlapping target-position table."""
    # Build the selected positions table.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {position_table_name} AS
        WITH filtered AS (
            SELECT
                date,
                time,
                code,
                minute_slot,
                prediction,
                prediction_available,
                current_tradable,
                signal_amount,
                sigma_intraday,
                adv_amount,
                {fillable_col} AS fillable,
                {return_col} AS simple_return
            FROM read_parquet('{parquet_glob}')
            WHERE {extra_filter_sql}
        ),
        ranked AS (
            SELECT
                date,
                time,
                code,
                minute_slot,
                prediction,
                prediction_available,
                current_tradable,
                signal_amount,
                sigma_intraday,
                adv_amount,
                fillable,
                simple_return,
                row_number() OVER (PARTITION BY date, time ORDER BY prediction) AS rank_asc,
                count(*) OVER (PARTITION BY date, time) AS name_count,
                ntile(3) OVER (PARTITION BY date, time ORDER BY signal_amount DESC) AS liq_bucket
            FROM filtered
        ),
        tagged AS (
            SELECT
                date,
                time,
                code,
                minute_slot,
                prediction,
                prediction_available,
                current_tradable,
                simple_return,
                sigma_intraday,
                adv_amount,
                fillable,
                CASE
                    WHEN liq_bucket = 1 THEN {spread_bps_high:.6f}
                    WHEN liq_bucket = 2 THEN {spread_bps_mid:.6f}
                    ELSE {spread_bps_low:.6f}
                END AS spread_bps,
                rank_asc,
                name_count,
                CAST(ceil({top_frac:.8f} * name_count) AS BIGINT) AS keep_count
            FROM ranked
        )
        SELECT
            date,
            time,
            minute_slot,
            dense_rank() OVER (PARTITION BY minute_slot ORDER BY date, time) AS slot_bar_id,
            code,
            prediction,
            prediction_available,
            current_tradable,
            simple_return,
            sigma_intraday,
            adv_amount,
            fillable,
            spread_bps,
            CASE
                WHEN rank_asc <= keep_count THEN -1
                ELSE 1
            END AS side,
            CASE
                WHEN rank_asc <= keep_count THEN -0.5 / keep_count
                ELSE 0.5 / keep_count
            END AS target_weight
        FROM tagged
        WHERE rank_asc <= keep_count OR rank_asc > name_count - keep_count
        """
    )


def export_duckdb_tables(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
    table_name: str,
    output_name: str,
) -> Path:
    """Export one DuckDB table to a CSV file."""
    # Define the output path.
    out_path = output_dir / output_name

    # Write the table to disk as CSV.
    con.execute(
        f"""
        COPY {table_name}
        TO '{out_path.as_posix()}'
        (HEADER, DELIMITER ',')
        """
    )
    return out_path


def t_stat_from_series(series: pd.Series) -> float:
    """Compute one standard t-statistic from a return series."""
    # Convert to float numpy values.
    values = series.astype(float).to_numpy()
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    return float(mean / (std / np.sqrt(values.size)))


def cumulative_return_from_series(series: pd.Series) -> float:
    """Compound one return series into total return."""
    # Convert to float numpy values.
    values = series.astype(float).to_numpy()
    return float(np.cumprod(1.0 + values)[-1] - 1.0)


def max_drawdown_from_series(series: pd.Series) -> float:
    """Compute max drawdown from one return series."""
    # Convert to float numpy values.
    values = series.astype(float).to_numpy()
    wealth = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def summarize_return_series(series: pd.Series, annual_days: int) -> dict[str, float]:
    """Summarize one daily return series into performance statistics."""
    # Compute mean/std and Sharpe.
    values = series.astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    sharpe = mean / std * np.sqrt(float(annual_days))

    # Compute path-level metrics.
    terminal_wealth = float(np.cumprod(1.0 + values.to_numpy())[-1])
    cum_return = terminal_wealth - 1.0
    max_dd = max_drawdown_from_series(values)
    t_stat = t_stat_from_series(values)
    hit = float((values > 0.0).mean())
    annualized_return = float(terminal_wealth ** (float(annual_days) / float(values.shape[0])) - 1.0) if terminal_wealth > 0.0 else float("nan")
    return {
        "mean_daily_return": mean,
        "std_daily_return": std,
        "annualized_return": annualized_return,
        "annualized_sharpe": float(sharpe),
        "t_stat": t_stat,
        "cum_return": cum_return,
        "max_drawdown": max_dd,
        "positive_day_ratio": hit,
        "day_count": float(values.shape[0]),
    }


def simulate_slot_book(slot_positions: pd.DataFrame, impact_eta: float) -> pd.DataFrame:
    """Simulate one minute-slot book with explicit order, fill, and position state."""
    # Normalize the input rows into one stable slot book.
    slot_positions = slot_positions.sort_values(["slot_bar_id", "code"]).reset_index(drop=True)
    minute_slot = int(slot_positions["minute_slot"].iloc[0]) if int(slot_positions.shape[0]) > 0 else -1

    # Keep the previous end-of-bar holdings as the book state.
    position_state: dict[int, dict[str, float]] = {}
    bar_rows: list[dict[str, float]] = []
    group_cols = ["slot_bar_id", "date", "time", "minute_slot"]
    for key, bar_df in slot_positions.groupby(group_cols, sort=True):
        slot_bar_id, date, time_value, _ = key

        # Build the target orders from the current ranking result.
        target_rows: dict[int, dict[str, float]] = {}
        executed_rows: dict[int, dict[str, float]] = {}
        for row in bar_df.itertuples(index=False):
            code = int(row.code)
            target_weight = float(row.target_weight)
            simple_return = float(row.simple_return) if np.isfinite(row.simple_return) else 0.0
            spread_bps = float(row.spread_bps)
            sigma_intraday = float(row.sigma_intraday)
            adv_amount = float(row.adv_amount)
            fillable = bool(row.fillable) and np.isfinite(row.simple_return)
            target_rows[code] = {
                "target_weight": target_weight,
                "spread_bps": spread_bps,
                "sigma_intraday": sigma_intraday,
                "adv_amount": adv_amount,
            }
            executed_rows[code] = {
                "weight": target_weight if fillable else 0.0,
                "simple_return": simple_return,
                "spread_bps": spread_bps,
                "sigma_intraday": sigma_intraday,
                "adv_amount": adv_amount,
                "fillable": float(fillable),
            }

        # Translate target weights into executed orders against the prior state.
        union_codes = sorted(set(position_state.keys()) | set(target_rows.keys()))
        planned_turnover = 0.0
        turnover = 0.0
        spread_cost = 0.0
        impact_coeff = 0.0
        long_exposure = 0.0
        short_exposure = 0.0
        for code in union_codes:
            prev_after_weight = float(position_state.get(code, {}).get("weight", 0.0))
            target_weight = float(target_rows.get(code, {}).get("target_weight", 0.0))
            executed_weight = float(executed_rows.get(code, {}).get("weight", 0.0))
            meta = executed_rows.get(code, position_state.get(code))
            planned_turnover += abs(target_weight - prev_after_weight)
            abs_delta = abs(executed_weight - prev_after_weight)
            turnover += abs_delta
            if meta is not None and abs_delta > 0.0:
                spread_cost += 0.5 * (float(meta["spread_bps"]) / 10000.0) * abs_delta
                impact_coeff += (
                    float(impact_eta)
                    * float(meta["sigma_intraday"])
                    * abs_delta
                    * np.sqrt(abs_delta / float(meta["adv_amount"]))
                )
            if executed_weight > 0.0:
                long_exposure += executed_weight
            if executed_weight < 0.0:
                short_exposure += -executed_weight
        planned_turnover *= 0.5
        turnover *= 0.5

        # Accumulate one bar of realized portfolio return from executed holdings.
        gross_return = 0.0
        for payload in executed_rows.values():
            gross_return += float(payload["weight"]) * float(payload["simple_return"])

        # Roll the executed holdings forward into the next state.
        next_state: dict[int, dict[str, float]] = {}
        for code, payload in executed_rows.items():
            executed_weight = float(payload["weight"])
            if executed_weight == 0.0:
                continue
            after_weight = executed_weight * (1.0 + float(payload["simple_return"])) / (1.0 + gross_return)
            next_state[int(code)] = {
                "weight": float(after_weight),
                "spread_bps": float(payload["spread_bps"]),
                "sigma_intraday": float(payload["sigma_intraday"]),
                "adv_amount": float(payload["adv_amount"]),
            }
        position_state = next_state

        # Persist one bar-level accounting row for downstream summaries.
        planned_name_count = float(bar_df.shape[0])
        filled_name_count = float(sum(int(payload["fillable"]) for payload in executed_rows.values()))
        executed_gross_exposure = float(long_exposure + short_exposure)
        cash_buffer = float(max(0.0, 1.0 - executed_gross_exposure))
        bar_rows.append(
            {
                "minute_slot": float(minute_slot),
                "slot_bar_id": float(slot_bar_id),
                "date": float(date),
                "time": float(time_value),
                "gross_return": float(gross_return),
                "planned_turnover": float(planned_turnover),
                "turnover": float(turnover),
                "spread_cost": float(spread_cost),
                "impact_coeff": float(impact_coeff),
                "planned_name_count": float(planned_name_count),
                "filled_name_count": float(filled_name_count),
                "fill_ratio": float(filled_name_count / planned_name_count) if planned_name_count > 0.0 else float("nan"),
                "long_exposure": float(long_exposure),
                "short_exposure": float(short_exposure),
                "executed_gross_exposure": executed_gross_exposure,
                "cash_buffer": cash_buffer,
            }
        )
    return pd.DataFrame(bar_rows)


def simulate_strategy_bars(position_path: Path, impact_eta: float) -> pd.DataFrame:
    """Simulate all slot books from one exported position table."""
    # Load the exported target positions once.
    position_df = pd.read_csv(position_path).sort_values(["minute_slot", "slot_bar_id", "code"]).reset_index(drop=True)
    if int(position_df.shape[0]) == 0:
        return pd.DataFrame(
            columns=[
                "minute_slot",
                "slot_bar_id",
                "date",
                "time",
                "gross_return",
                "planned_turnover",
                "turnover",
                "spread_cost",
                "impact_coeff",
                "planned_name_count",
                "filled_name_count",
                "fill_ratio",
                "long_exposure",
                "short_exposure",
                "executed_gross_exposure",
                "cash_buffer",
            ]
        )

    # Simulate each non-overlap slot independently and stack the results.
    slot_rows: list[pd.DataFrame] = []
    for _, slot_df in position_df.groupby("minute_slot", sort=True):
        slot_rows.append(simulate_slot_book(slot_df.reset_index(drop=True), impact_eta))
    return pd.concat(slot_rows, axis=0, ignore_index=True)


def build_slot_daily(slot_bar: pd.DataFrame) -> pd.DataFrame:
    """Aggregate slot bar returns and cost coefficients into slot daily series."""
    # Define the per-day compounding for each return and cost component.
    def compound(series: pd.Series) -> float:
        """Compound one per-bar series into a daily value."""
        # Convert to numpy and compound.
        values = series.astype(float).to_numpy()
        return float(np.prod(1.0 + values) - 1.0)

    # Compute compounded gross returns and summed turnover/cost coefficients.
    agg = {
        "gross_return": compound,
        "planned_turnover": "sum",
        "turnover": "sum",
        "spread_cost": "sum",
        "impact_coeff": "sum",
        "fill_ratio": "mean",
        "executed_gross_exposure": "mean",
        "cash_buffer": "mean",
    }
    return slot_bar.groupby(["minute_slot", "date"], sort=True).agg(agg).reset_index()


def build_combined_daily(slot_daily: pd.DataFrame) -> pd.DataFrame:
    """Combine ten non-overlapping slots with equal capital."""
    # Average the slot-level daily series to represent equal-capital blend.
    agg = {
        "gross_return": "mean",
        "planned_turnover": "mean",
        "turnover": "mean",
        "spread_cost": "mean",
        "impact_coeff": "mean",
        "fill_ratio": "mean",
        "executed_gross_exposure": "mean",
        "cash_buffer": "mean",
    }
    return slot_daily.groupby("date", sort=True).agg(agg).reset_index()


def summarize_slot_daily(slot_daily: pd.DataFrame, annual_days: int) -> pd.DataFrame:
    """Summarize each minute slot into mean return and Sharpe metrics."""
    # Compute per-slot headline statistics from the daily gross return.
    grouped = slot_daily.groupby("minute_slot", sort=True)["gross_return"]
    mean = grouped.mean().rename("mean_daily_return")
    std = grouped.std(ddof=1).rename("std_daily_return")
    sharpe = (mean / std * np.sqrt(float(annual_days))).rename("annualized_sharpe")
    count = grouped.count().rename("day_count")

    # Combine into one summary table.
    out = pd.concat([mean, std, sharpe, count], axis=1).reset_index()
    return out.sort_values("minute_slot").reset_index(drop=True)


def drawdown_curve_from_returns(returns: pd.Series) -> pd.Series:
    """Build one drawdown curve series from periodic returns."""
    # Convert returns to a wealth path.
    values = returns.astype(float).to_numpy()
    wealth = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return pd.Series(drawdown)


def plot_drawdown_curve(daily: pd.DataFrame, output_path: Path, aum_list: list[float]) -> None:
    """Plot drawdown curves for gross and selected net AUMs."""
    # Build the timestamp axis.
    timestamp = pd.to_datetime(daily["date"].astype(int).astype(str), format="%y%m%d")

    # Render the drawdown curves.
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(timestamp, drawdown_curve_from_returns(daily["gross_return"]), label="Gross", color="#2E86AB")
    for aum in aum_list:
        col = f"net_return_aum_{int(aum/1_000_000):d}m"
        ax.plot(timestamp, drawdown_curve_from_returns(daily[col]), label=f"Net {int(aum/1_000_000)}M")
    ax.set_title("Drawdown Curve")
    ax.set_ylabel("Drawdown (fraction)")
    ax.grid(alpha=0.25)
    ax.legend()

    # Format the x-axis as calendar dates.
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_slot_sharpe(slot_summary: pd.DataFrame, output_path: Path, title: str) -> None:
    """Plot a bar chart of per-slot annualized Sharpe."""
    # Prepare the bar inputs.
    x = slot_summary["minute_slot"].astype(int).to_numpy()
    y = slot_summary["annualized_sharpe"].astype(float).to_numpy()

    # Render the figure.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, y, color="#3D405B")
    ax.set_title(title)
    ax.set_xlabel("Minute Slot (minute % 10)")
    ax.set_ylabel("Annualized Sharpe")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def attach_net_returns_for_aums(combined_daily: pd.DataFrame, aum_list: list[float]) -> pd.DataFrame:
    """Attach net return columns for each AUM assumption."""
    # Copy the input frame to avoid mutating callers.
    out = combined_daily.copy()

    # Add net return columns with impact scaling as sqrt(AUM).
    for aum in aum_list:
        name = f"net_return_aum_{int(aum/1_000_000):d}m"
        out[name] = out["gross_return"].astype(float) - out["spread_cost"].astype(float) - out["impact_coeff"].astype(
            float
        ) * np.sqrt(float(aum))
    return out


def plot_strategy_curves(daily: pd.DataFrame, output_path: Path, aum_list: list[float]) -> None:
    """Plot cumulative gross/net curves and turnover."""
    # Build the timestamp axis.
    timestamp = pd.to_datetime(daily["date"].astype(int).astype(str), format="%y%m%d")

    # Render the figure panels.
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(timestamp, (1.0 + daily["gross_return"].astype(float)).cumprod(), label="Gross")
    for aum in aum_list:
        col = f"net_return_aum_{int(aum/1_000_000):d}m"
        axes[0].plot(timestamp, (1.0 + daily[col].astype(float)).cumprod(), label=f"Net {int(aum/1_000_000)}M")
    axes[0].set_yscale("log")
    axes[0].set_title("Combined Daily Strategy Curve (Log)")
    axes[0].set_ylabel("Wealth (log scale)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(timestamp, daily["turnover"].astype(float), color="#2E86AB")
    axes[1].set_title("Mean Daily Turnover (Equal-Capital Slots)")
    axes[1].set_ylabel("Turnover (daily fraction)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(timestamp, daily["spread_cost"].astype(float) * 1e4, label="Spread (bps)", color="#E07A5F")
    axes[2].plot(timestamp, daily["impact_coeff"].astype(float) * 1e4, label="Impact coeff (scaled)", color="#3D405B")
    axes[2].set_title("Daily Cost Components (Diagnostic)")
    axes[2].set_ylabel("Spread cost / impact coeff x1e4")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    # Format the x-axis as calendar dates.
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(formatter)
    axes[2].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_capacity_sweep(strategy_summary: dict[str, Any], output_path: Path, aum_list: list[float]) -> None:
    """Plot net return and Sharpe vs AUM for the realistic strategy."""
    # Build the sweep vectors from the summary payload.
    x = np.array([float(aum) / 1_000_000.0 for aum in aum_list], dtype=float)
    net_mean = np.array([float(strategy_summary[f"net_{int(aum/1_000_000):d}m"]["mean_daily_return"]) for aum in aum_list])
    net_sharpe = np.array([float(strategy_summary[f"net_{int(aum/1_000_000):d}m"]["annualized_sharpe"]) for aum in aum_list])

    # Render the two-panel figure.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x, net_mean * 1e4, marker="o", color="#2E86AB")
    axes[0].set_title("Net Mean Daily Return vs AUM")
    axes[0].set_ylabel("Mean daily return (bps)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, net_sharpe, marker="o", color="#E07A5F")
    axes[1].set_title("Net Sharpe vs AUM")
    axes[1].set_xlabel("AUM (million CNY)")
    axes[1].set_ylabel("Sharpe")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def estimate_capacity_from_impact_budget(daily: pd.DataFrame, impact_budget_bps: float) -> float:
    """Estimate capacity AUM from an average daily impact budget."""
    # Convert the budget to return units.
    budget = float(impact_budget_bps) / 10000.0

    # Estimate mean daily impact coefficient.
    mean_coeff = float(daily["impact_coeff"].astype(float).mean())
    if not np.isfinite(mean_coeff) or mean_coeff <= 0.0:
        return float("nan")
    return float((budget / mean_coeff) ** 2)


def build_report_text(
    config: Backtest2Config,
    strategy_summary_path: Path,
) -> str:
    """Compose one backtest2 markdown report focused on portfolio backtest outputs."""
    # Load the strategy summary payload.
    strategy_summary = yaml.safe_load(strategy_summary_path.read_text(encoding="utf-8"))
    baseline = strategy_summary["baseline_open"]
    realistic = strategy_summary["realistic_vwap"]

    # Compose the markdown lines.
    lines: list[str] = []
    lines.append(f"# {config.report_title}")
    lines.append("")
    lines.append("## 研究范围")
    lines.append("")
    lines.append(f"- 预测来源: `{config.predict_manifest_path.as_posix()}`, 但 `signal evaluation` 已迁移到 `prediction_nn2/eval_ic.py`.")
    lines.append(f"- 输出目录: `{config.output_dir.as_posix()}`.")
    lines.append("- Horizon: `hold = 10 bar`, non-overlap 使用 `minute_slot = minute % 10`.")
    lines.append("- Universe: 先从 raw `stock1m` 重建 point-in-time universe, 再 `left join prediction`, 不再让 prediction 样本行集决定股票池.")
    lines.append("- 当前明确不可实现的两项: `borrow/shortable flag` 与 `market-cap bucket`.")
    lines.append("")
    lines.append("## 信号评估")
    lines.append("")
    lines.append("- 当前脚本不再生成任何 `prediction vs target` 的信号质量结论.")
    lines.append("- IC, Rank IC, ICIR, rank curve, rank turnover, rolling group IC 应统一由 `eval_ic` 报告负责.")
    lines.append("")
    lines.append("## 组合回测")
    lines.append("")
    lines.append("- Entry/Exit: `t+1` 首次可交易 bar 入场, 持有 `10 bar` 后首次可交易 bar 出场.")
    lines.append("- 权重: top10% long / bottom10% short, 两端等权各 50%.")
    lines.append("- Ranking universe: 仅要求当前 bar 可交易且有 prediction, 不再用未来 entry/exit fillability 过滤.")
    lines.append("- 执行状态机: 先生成 target weights, 再模拟 order/fill/position. 无法完成 entry/exit 的名字在执行层回到现金, 不反向污染选股.")
    lines.append(
        f"- 成本: spread bucket(高/中/低流动) + ex-ante trailing ADV(`{int(config.adv_lookback_days)}d`) + rolling intraday sigma(`{int(config.sigma_lookback_bars)} bar`)."
    )
    lines.append("")
    lines.append(f"- 策略汇总: `{strategy_summary_path.as_posix()}`.")
    lines.append(f"- 策略曲线: `{(config.output_dir / 'strategy_curves.png').as_posix()}`.")
    lines.append(f"- Baseline(Open, no cost) curve: `{(config.output_dir / 'baseline_open_strategy.png').as_posix()}`.")
    lines.append(f"- Drawdown curve: `{(config.output_dir / 'drawdown_curve.png').as_posix()}`.")
    lines.append(f"- Slot Sharpe: `{(config.output_dir / 'slot_sharpe.png').as_posix()}`.")
    lines.append(f"- Capacity sweep: `{(config.output_dir / 'capacity_sweep.png').as_posix()}`.")
    lines.append("")
    lines.append("### Baseline(Open)")
    lines.append("")
    lines.append(f"- Mean fill ratio: {baseline['execution']['mean_fill_ratio'] * 100:.2f}%.")
    lines.append(f"- Mean executed gross exposure: {baseline['execution']['mean_executed_gross_exposure']:.4f}.")
    lines.append(f"- Mean cash buffer: {baseline['execution']['mean_cash_buffer']:.4f}.")
    lines.append("")
    lines.append("### Realistic(VWAP)")
    lines.append("")
    lines.append(f"- Mean fill ratio: {realistic['execution']['mean_fill_ratio'] * 100:.2f}%.")
    lines.append(f"- Mean executed gross exposure: {realistic['execution']['mean_executed_gross_exposure']:.4f}.")
    lines.append(f"- Mean cash buffer: {realistic['execution']['mean_cash_buffer']:.4f}.")
    lines.append("")
    lines.append("### Capacity (impact budget)")
    lines.append("")
    for budget_bps in config.impact_budget_bps_list:
        key = f"capacity_at_{int(budget_bps)}bps"
        lines.append(f"- {int(budget_bps)}bps: AUM ~= {realistic['capacity'][key] / 1_000_000:.2f}M.")
    return "\n".join(lines)


def summarize_baseline_open_strategy(
    position_path: Path,
    annual_days: int,
    impact_eta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize the baseline strategy using open/open returns without costs."""
    # Simulate the bar-level portfolio path from target positions.
    slot_bar = simulate_strategy_bars(position_path, impact_eta)

    # Build the slot and combined daily tables.
    slot_daily = build_slot_daily(slot_bar)
    combined_daily = build_combined_daily(slot_daily)
    slot_summary = summarize_slot_daily(slot_daily, annual_days)

    # Compute the baseline summary metrics.
    summary = {
        "gross": summarize_return_series(combined_daily["gross_return"], annual_days),
        "net": summarize_return_series(combined_daily["gross_return"], annual_days),
        "turnover": {
            "mean_daily_planned_turnover": float(combined_daily["planned_turnover"].astype(float).mean()),
            "mean_daily_turnover": float(combined_daily["turnover"].astype(float).mean()),
            "p50_daily_turnover": float(combined_daily["turnover"].astype(float).quantile(0.50)),
            "p95_daily_turnover": float(combined_daily["turnover"].astype(float).quantile(0.95)),
        },
        "execution": {
            "mean_fill_ratio": float(combined_daily["fill_ratio"].astype(float).mean()),
            "mean_executed_gross_exposure": float(combined_daily["executed_gross_exposure"].astype(float).mean()),
            "mean_cash_buffer": float(combined_daily["cash_buffer"].astype(float).mean()),
        },
    }
    return slot_bar, combined_daily, slot_summary, summary


def summarize_realistic_vwap_strategy(
    position_path: Path,
    annual_days: int,
    aum_list: list[float],
    impact_budget_bps_list: list[float],
    impact_eta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize the realistic strategy using vwap/vwap returns with spread+impact costs."""
    # Simulate the bar-level portfolio path from target positions.
    slot_bar = simulate_strategy_bars(position_path, impact_eta)

    # Build the slot and combined daily tables.
    slot_daily = build_slot_daily(slot_bar)
    combined_daily = build_combined_daily(slot_daily)
    combined_daily = attach_net_returns_for_aums(combined_daily, aum_list)
    slot_summary = summarize_slot_daily(slot_daily, annual_days)

    # Build the realistic summary metrics for gross and each AUM net.
    summary: dict[str, Any] = {}
    summary["gross"] = summarize_return_series(combined_daily["gross_return"], annual_days)
    for aum in aum_list:
        key = f"net_{int(aum/1_000_000):d}m"
        col = f"net_return_aum_{int(aum/1_000_000):d}m"
        summary[key] = summarize_return_series(combined_daily[col], annual_days)

    # Add turnover and capacity diagnostics.
    summary["turnover"] = {
        "mean_daily_planned_turnover": float(combined_daily["planned_turnover"].astype(float).mean()),
        "mean_daily_turnover": float(combined_daily["turnover"].astype(float).mean()),
        "p50_daily_turnover": float(combined_daily["turnover"].astype(float).quantile(0.50)),
        "p95_daily_turnover": float(combined_daily["turnover"].astype(float).quantile(0.95)),
    }
    summary["execution"] = {
        "mean_fill_ratio": float(combined_daily["fill_ratio"].astype(float).mean()),
        "mean_executed_gross_exposure": float(combined_daily["executed_gross_exposure"].astype(float).mean()),
        "mean_cash_buffer": float(combined_daily["cash_buffer"].astype(float).mean()),
    }
    summary["capacity"] = {
        f"capacity_at_{int(b)}bps": float(estimate_capacity_from_impact_budget(combined_daily, float(b)))
        for b in impact_budget_bps_list
    }
    return slot_bar, combined_daily, slot_summary, summary


def run_backtest2() -> Path:
    """Run the full backtest2 pipeline and return the report path."""
    # Build the fixed configuration and directories.
    config = build_config()
    ensure_output_dir(config.output_dir)

    # Materialize feature chunks for price-based returns and costs.
    feature_glob = materialize_feature_chunks(config)

    # Create the DuckDB target-position tables for the two execution definitions.
    con = connect_duckdb(config.feature_db_path)
    materialize_slot_strategy_table(
        con,
        feature_glob,
        config.top_frac,
        "ret_open_exec_10",
        "fillable_open",
        "current_tradable = true AND prediction_available = true AND adv_amount IS NOT NULL AND adv_amount > 0 AND sigma_intraday IS NOT NULL",
        config.spread_bps_high,
        config.spread_bps_mid,
        config.spread_bps_low,
        "open_slot_positions",
    )
    materialize_slot_strategy_table(
        con,
        feature_glob,
        config.top_frac,
        "ret_vwap_exec_10",
        "fillable_vwap",
        "current_tradable = true AND prediction_available = true AND adv_amount IS NOT NULL AND adv_amount > 0 AND sigma_intraday IS NOT NULL",
        config.spread_bps_high,
        config.spread_bps_mid,
        config.spread_bps_low,
        "vwap_slot_positions",
    )

    # Export the target-position tables to CSV for pandas simulation.
    open_position_csv = export_duckdb_tables(con, config.output_dir, "open_slot_positions", "open_slot_positions.csv")
    vwap_position_csv = export_duckdb_tables(con, config.output_dir, "vwap_slot_positions", "vwap_slot_positions.csv")
    con.close()

    # Summarize module B baseline (open) and realistic (vwap) strategies.
    baseline_bar, baseline_daily, baseline_slot_summary, baseline_summary = summarize_baseline_open_strategy(
        open_position_csv,
        config.annual_days,
        config.impact_eta,
    )
    baseline_bar.to_csv(config.output_dir / "baseline_open_slot_bar.csv", index=False)
    baseline_daily.to_csv(config.output_dir / "baseline_open_combined_daily.csv", index=False)
    baseline_slot_summary.to_csv(config.output_dir / "baseline_open_slot_summary.csv", index=False)
    plot_strategy_curves(baseline_daily, config.output_dir / "baseline_open_strategy.png", [])
    plot_slot_sharpe(baseline_slot_summary, config.output_dir / "baseline_open_slot_sharpe.png", "Baseline(Open) Slot Sharpe")

    realistic_bar, realistic_daily, realistic_slot_summary, realistic_summary = summarize_realistic_vwap_strategy(
        vwap_position_csv,
        config.annual_days,
        config.aum_list,
        config.impact_budget_bps_list,
        config.impact_eta,
    )
    realistic_bar.to_csv(config.output_dir / "realistic_vwap_slot_bar.csv", index=False)
    realistic_daily.to_csv(config.output_dir / "realistic_vwap_combined_daily.csv", index=False)
    realistic_slot_summary.to_csv(config.output_dir / "realistic_vwap_slot_summary.csv", index=False)
    plot_strategy_curves(realistic_daily, config.output_dir / "strategy_curves.png", config.aum_list)
    plot_drawdown_curve(realistic_daily, config.output_dir / "drawdown_curve.png", config.aum_list)
    plot_slot_sharpe(realistic_slot_summary, config.output_dir / "slot_sharpe.png", "Realistic(VWAP) Slot Sharpe")
    plot_capacity_sweep(realistic_summary, config.output_dir / "capacity_sweep.png", config.aum_list)

    # Build the strategy summary payload.
    strategy_payload: dict[str, Any] = {
        "baseline_open": baseline_summary,
        "realistic_vwap": realistic_summary,
        "cost_model": {
            "spread_bps_high": float(config.spread_bps_high),
            "spread_bps_mid": float(config.spread_bps_mid),
            "spread_bps_low": float(config.spread_bps_low),
            "impact_eta": float(config.impact_eta),
            "adv_lookback_days": int(config.adv_lookback_days),
            "sigma_lookback_bars": int(config.sigma_lookback_bars),
            "aum_list": [float(x) for x in config.aum_list],
        },
    }
    strategy_summary_path = config.output_dir / "strategy_summary.yaml"
    with strategy_summary_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(strategy_payload, file_obj, allow_unicode=True, sort_keys=False)

    # Compose and write the final markdown report.
    report_text = build_report_text(config, strategy_summary_path)
    report_path = config.output_dir / "research_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    """Run backtest2 as a script entry."""
    # Run the full pipeline and print the report path.
    report_path = run_backtest2()
    print(report_path.as_posix())


if __name__ == "__main__":
    main()
