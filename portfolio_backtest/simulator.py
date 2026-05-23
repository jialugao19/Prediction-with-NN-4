"""Build target portfolios and simulate execution-aware portfolio paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


def connect_duckdb(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open one DuckDB connection with stable settings."""
    # Open the local database file and pin deterministic execution knobs.
    con = duckdb.connect(str(db_path))
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA enable_progress_bar=false")
    return con



def materialize_slot_strategy_table(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    top_frac: float,
    long_enabled: bool,
    short_enabled: bool,
    max_liq_bucket: int,
    return_col: str,
    fillable_col: str,
    entry_is_up_limit_col: str,
    entry_is_down_limit_col: str,
    exit_is_up_limit_col: str,
    exit_is_down_limit_col: str,
    extra_filter_sql: str,
    spread_bps_high: float,
    spread_bps_mid: float,
    spread_bps_low: float,
    position_table_name: str,
) -> None:
    """Materialize one non-overlapping target-position table."""
    # Define side-specific SQL fragments for enabled long/short legs.
    side_cases: list[str] = []
    weight_cases: list[str] = []
    side_filters: list[str] = []
    if bool(short_enabled):
        side_cases.append("WHEN rank_asc <= keep_count THEN -1")
        weight_cases.append("WHEN rank_asc <= keep_count THEN -1.0 * short_gross / keep_count")
        side_filters.append("rank_asc <= keep_count")
    if bool(long_enabled):
        side_cases.append("WHEN rank_asc > name_count - keep_count THEN 1")
        weight_cases.append("WHEN rank_asc > name_count - keep_count THEN 1.0 * long_gross / keep_count")
        side_filters.append("rank_asc > name_count - keep_count")
    if len(side_filters) == 0:
        raise RuntimeError("At least one of long_enabled or short_enabled must be true.")
    long_gross = 0.5 if bool(short_enabled) else 1.0
    short_gross = 0.5 if bool(long_enabled) else 1.0
    side_case_sql = "\n                    ".join(side_cases)
    weight_case_sql = "\n                    ".join(weight_cases)
    side_filter_sql = " OR ".join(side_filters)

    # Build the selected positions table from the feature parquet source.
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
                is_limit_up_all_day,
                is_limit_down_all_day,
                CAST({fillable_col} AS BOOLEAN) AS fillable_base,
                CAST({return_col} AS DOUBLE) AS simple_return,
                CAST({entry_is_up_limit_col} AS BOOLEAN) AS entry_is_up_limit,
                CAST({entry_is_down_limit_col} AS BOOLEAN) AS entry_is_down_limit,
                CAST({exit_is_up_limit_col} AS BOOLEAN) AS exit_is_up_limit,
                CAST({exit_is_down_limit_col} AS BOOLEAN) AS exit_is_down_limit
            FROM read_parquet('{parquet_glob}')
            WHERE {extra_filter_sql}
        ),
        liquidity_tagged AS (
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
                is_limit_up_all_day,
                is_limit_down_all_day,
                fillable_base,
                simple_return,
                entry_is_up_limit,
                entry_is_down_limit,
                exit_is_up_limit,
                exit_is_down_limit,
                ntile(3) OVER (PARTITION BY date, time ORDER BY signal_amount DESC) AS liq_bucket
            FROM filtered
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
                simple_return,
                sigma_intraday,
                adv_amount,
                is_limit_up_all_day,
                is_limit_down_all_day,
                fillable_base,
                entry_is_up_limit,
                entry_is_down_limit,
                exit_is_up_limit,
                exit_is_down_limit,
                liq_bucket,
                row_number() OVER (PARTITION BY date, time ORDER BY prediction) AS rank_asc,
                count(*) OVER (PARTITION BY date, time) AS name_count
            FROM liquidity_tagged
            WHERE liq_bucket <= {int(max_liq_bucket)}
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
                is_limit_up_all_day,
                is_limit_down_all_day,
                fillable_base,
                entry_is_up_limit,
                entry_is_down_limit,
                exit_is_up_limit,
                exit_is_down_limit,
                CASE
                    WHEN liq_bucket = 1 THEN {spread_bps_high:.6f}
                    WHEN liq_bucket = 2 THEN {spread_bps_mid:.6f}
                    ELSE {spread_bps_low:.6f}
                END AS spread_bps,
                {float(long_gross):.8f} AS long_gross,
                {float(short_gross):.8f} AS short_gross,
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
            CASE
                WHEN rank_asc <= keep_count THEN (
                    fillable_base
                    AND (NOT is_limit_down_all_day)
                    AND (NOT entry_is_down_limit)
                    AND (NOT is_limit_up_all_day)
                    AND (NOT exit_is_up_limit)
                )
                ELSE (
                    fillable_base
                    AND (NOT is_limit_up_all_day)
                    AND (NOT entry_is_up_limit)
                    AND (NOT is_limit_down_all_day)
                    AND (NOT exit_is_down_limit)
                )
            END AS fillable,
            spread_bps,
            CASE
                    {side_case_sql}
            END AS side,
            CASE
                    {weight_case_sql}
            END AS target_weight
        FROM tagged
        WHERE {side_filter_sql}
        """
    )
def export_duckdb_table(con: duckdb.DuckDBPyConnection, output_dir: Path, table_name: str, output_name: str) -> Path:
    """Export one DuckDB table to a CSV file."""
    # Define the output path once.
    out_path = Path(output_dir) / output_name

    # Export parquet outputs when requested so large position tables stay compact.
    if str(out_path.suffix).lower() == ".parquet":
        con.execute(
            f"""
            COPY {table_name}
            TO '{out_path.as_posix()}'
            (FORMAT PARQUET)
            """
        )
        return out_path

    # Export csv outputs for small human-readable tables.
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
    # Convert the input to float numpy values and compute mean/std.
    values = series.astype(float).to_numpy()
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    return float(mean / (std / np.sqrt(values.size)))


def max_drawdown_from_series(series: pd.Series) -> float:
    """Compute max drawdown from one return series."""
    # Convert the return series into a wealth path and drawdown path.
    values = series.astype(float).to_numpy()
    wealth = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def summarize_return_series(series: pd.Series, annual_days: int) -> dict[str, float]:
    """Summarize one daily return series into performance statistics."""
    # Compute mean/std and annualized Sharpe.
    values = series.astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    sharpe = mean / std * np.sqrt(float(annual_days))

    # Compute path-level metrics from the compounded wealth path.
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
                "weight": target_weight if fillable else np.nan,
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
            raw_executed_weight = float(executed_rows.get(code, {}).get("weight", 0.0))
            executed_weight = prev_after_weight if np.isnan(raw_executed_weight) else raw_executed_weight
            if code in executed_rows:
                executed_rows[code]["weight"] = float(executed_weight)
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
    if str(Path(position_path).suffix).lower() == ".parquet":
        position_df = pd.read_parquet(position_path)
    else:
        position_df = pd.read_csv(position_path)
    position_df = position_df.sort_values(["minute_slot", "slot_bar_id", "code"]).reset_index(drop=True)
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
    # Define the per-day compounding for each return component.
    def compound(series: pd.Series) -> float:
        """Compound one per-bar series into a daily value."""
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
    return pd.concat([mean, std, sharpe, count], axis=1).reset_index().sort_values("minute_slot").reset_index(drop=True)


def attach_net_returns_for_aums(combined_daily: pd.DataFrame, aum_list: list[float]) -> pd.DataFrame:
    """Attach net return columns for each AUM assumption."""
    # Copy the input frame to avoid mutating callers.
    out = combined_daily.copy()
    for aum in list(aum_list):
        name = f"net_return_aum_{int(aum/1_000_000):d}m"
        out[name] = out["gross_return"].astype(float) - out["spread_cost"].astype(float) - out["impact_coeff"].astype(float) * np.sqrt(
            float(aum)
        )
    if len(list(aum_list)) == 1:
        only_aum = float(list(aum_list)[0])
        out["net_return"] = out[f"net_return_aum_{int(only_aum/1_000_000):d}m"]
    return out


def estimate_capacity_from_impact_budget(daily: pd.DataFrame, impact_budget_bps: float) -> float:
    """Estimate capacity AUM from an average daily impact budget."""
    # Convert the budget into return units and solve the inverse scaling.
    budget = float(impact_budget_bps) / 10000.0
    mean_coeff = float(daily["impact_coeff"].astype(float).mean())
    if not np.isfinite(mean_coeff) or mean_coeff <= 0.0:
        return float("nan")
    return float((budget / mean_coeff) ** 2)


def summarize_baseline_open_strategy(position_path: Path, annual_days: int, impact_eta: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize the baseline strategy using open/open returns without costs."""
    # Simulate the bar-level portfolio path from target positions.
    slot_bar = simulate_strategy_bars(position_path, impact_eta)
    slot_daily = build_slot_daily(slot_bar)
    combined_daily = build_combined_daily(slot_daily)
    slot_summary = summarize_slot_daily(slot_daily, annual_days)
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
    """Summarize the realistic strategy using vwap/vwap returns with spread and impact costs."""
    # Simulate the bar-level portfolio path from target positions.
    slot_bar = simulate_strategy_bars(position_path, impact_eta)
    slot_daily = build_slot_daily(slot_bar)
    combined_daily = build_combined_daily(slot_daily)
    combined_daily = attach_net_returns_for_aums(combined_daily, aum_list)
    slot_summary = summarize_slot_daily(slot_daily, annual_days)

    # Build the realistic summary metrics for gross and each AUM net.
    summary: dict[str, Any] = {}
    summary["gross"] = summarize_return_series(combined_daily["gross_return"], annual_days)
    for aum in list(aum_list):
        key = f"net_{int(aum/1_000_000):d}m"
        col = f"net_return_aum_{int(aum/1_000_000):d}m"
        summary[key] = summarize_return_series(combined_daily[col], annual_days)

    # Add turnover, execution, and capacity diagnostics.
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
        for b in list(impact_budget_bps_list)
    }
    return slot_bar, combined_daily, slot_summary, summary
