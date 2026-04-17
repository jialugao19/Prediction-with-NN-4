"""Run optimistic and execution-safe backtest research on current NN predictions."""

from __future__ import annotations

from collections import OrderedDict
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
class BacktestConfig:
    """Store the fixed file layout and trading assumptions."""

    repo_root: Path
    manifest_path: Path
    output_dir: Path
    optimistic_db_path: Path
    safe_output_dir: Path
    safe_db_path: Path
    safe_chunk_dir: Path
    safe_manifest_path: Path
    raw_root: Path
    top_frac: float
    cost_bps_list: list[float]
    annual_days: int
    report_title: str
    safe_report_title: str
    entry_delay_bars: int
    holding_bars: int
    lookup_cache_size: int
    prediction_fill_value: float


@dataclass(frozen=True)
class ResearchRunConfig:
    """Store one concrete research run definition."""

    parquet_glob: str
    output_dir: Path
    db_path: Path
    report_title: str
    return_expr_sql: str
    return_description: str
    limitation_note: str
    trading_rule: str
    rebalance_rule: str
    selection_rule: str
    pool_rule: str
    slippage_rule: str


def build_config() -> BacktestConfig:
    """Build the fixed backtest configuration."""
    # Set the fixed project paths.
    repo_root = Path("/home/maomao/prediction-NN-2")
    manifest_path = Path(
        "/data-cache/nn/upgrade_20260328_gru_seq60_h10/date_ranges/run/eval_test/iter_5000/predict_manifest.yaml"
    )
    output_dir = Path("/data-cache/nn/0416/backtest_research")
    optimistic_db_path = output_dir / "backtest_research.duckdb"

    # Set the execution-safe output layout.
    safe_output_dir = output_dir / "execution_safe_open"
    safe_db_path = safe_output_dir / "backtest_research.duckdb"
    safe_chunk_dir = safe_output_dir / "predict_chunks"
    safe_manifest_path = safe_output_dir / "predict_manifest.yaml"

    # Set the trading inputs and assumptions.
    raw_root = Path("/data/ashare/market/stock1m")
    top_frac = 0.10
    cost_bps_list = [5.0, 10.0, 20.0]
    annual_days = 252
    report_title = "NN Test Prediction Backtest Research"
    safe_report_title = "NN Test Prediction Backtest Research (Execution-Safe Open/Open)"
    entry_delay_bars = 1
    holding_bars = 10
    lookup_cache_size = 8
    prediction_fill_value = 0.0
    return BacktestConfig(
        repo_root=repo_root,
        manifest_path=manifest_path,
        output_dir=output_dir,
        optimistic_db_path=optimistic_db_path,
        safe_output_dir=safe_output_dir,
        safe_db_path=safe_db_path,
        safe_chunk_dir=safe_chunk_dir,
        safe_manifest_path=safe_manifest_path,
        raw_root=raw_root,
        top_frac=top_frac,
        cost_bps_list=cost_bps_list,
        annual_days=annual_days,
        report_title=report_title,
        safe_report_title=safe_report_title,
        entry_delay_bars=entry_delay_bars,
        holding_bars=holding_bars,
        lookup_cache_size=lookup_cache_size,
        prediction_fill_value=prediction_fill_value,
    )


def ensure_output_dir(path: Path) -> None:
    """Create one output directory."""
    # Create the output directory tree.
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read one manifest YAML file."""
    # Load the manifest payload.
    return yaml.safe_load(manifest_path.read_text(encoding="utf-8"))


def load_manifest_glob(manifest_path: Path) -> str:
    """Resolve the parquet chunk glob from a predict manifest."""
    # Read the manifest file once.
    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent

    # Build the parquet glob path.
    first_chunk_parent = Path(manifest["chunk_files"][0]).parent.as_posix()
    parquet_glob = (manifest_dir / first_chunk_parent / "*.parquet").as_posix()
    return parquet_glob


def load_manifest_chunk_paths(manifest_path: Path) -> list[Path]:
    """Resolve all concrete parquet chunk paths from a predict manifest."""
    # Read the manifest file once.
    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent

    # Resolve the relative chunk paths.
    chunk_paths = [manifest_dir / Path(chunk_file) for chunk_file in manifest["chunk_files"]]
    return chunk_paths


def write_manifest(manifest_path: Path, chunk_paths: list[Path]) -> None:
    """Write one predict manifest for generated parquet chunks."""
    # Build the relative chunk file list.
    chunk_files = [str(Path("predict_chunks") / path.name) for path in chunk_paths]
    payload = {"chunk_files": chunk_files}

    # Serialize the manifest as YAML.
    with manifest_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, allow_unicode=True, sort_keys=False)


def connect_duckdb(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open one DuckDB connection with stable settings."""
    # Open the local database file.
    con = duckdb.connect(str(db_path))

    # Configure the execution parameters.
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA enable_progress_bar=false")
    return con


def materialize_decile_table(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    return_expr_sql: str,
    table_name: str,
) -> None:
    """Materialize one all-timestamp decile return table."""
    # Build the decile aggregation table.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS
        WITH ranked AS (
            SELECT
                date,
                time,
                prediction,
                {return_expr_sql} AS simple_return,
                ntile(10) OVER (PARTITION BY date, time ORDER BY prediction) AS decile
            FROM read_parquet('{parquet_glob}')
        )
        SELECT
            date,
            time,
            decile,
            count(*) AS name_count,
            avg(prediction) AS mean_prediction,
            avg(simple_return) AS mean_return
        FROM ranked
        GROUP BY date, time, decile
        ORDER BY date, time, decile
        """
    )


def materialize_slot_strategy_tables(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    top_frac: float,
    return_expr_sql: str,
    position_table_name: str,
    metric_table_name: str,
) -> None:
    """Materialize one non-overlapping slot strategy table."""
    # Build the selected positions table.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {position_table_name} AS
        WITH ranked AS (
            SELECT
                date,
                time,
                code,
                prediction,
                {return_expr_sql} AS simple_return,
                ((time // 100) % 100) % 10 AS minute_slot,
                row_number() OVER (PARTITION BY date, time ORDER BY prediction) AS rank_asc,
                count(*) OVER (PARTITION BY date, time) AS name_count
            FROM read_parquet('{parquet_glob}')
        ),
        tagged AS (
            SELECT
                date,
                time,
                code,
                prediction,
                simple_return,
                minute_slot,
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
            simple_return,
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

    # Build the per-bar strategy table.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {metric_table_name} AS
        WITH bar_return AS (
            SELECT
                minute_slot,
                slot_bar_id,
                min(date) AS date,
                min(time) AS time,
                sum(target_weight * simple_return) AS gross_return,
                sum(CASE WHEN side = 1 THEN target_weight * simple_return ELSE 0.0 END) AS long_leg_return,
                sum(CASE WHEN side = -1 THEN target_weight * simple_return ELSE 0.0 END) AS short_leg_return
            FROM {position_table_name}
            GROUP BY minute_slot, slot_bar_id
        ),
        pair_weights AS (
            SELECT
                coalesce(curr.minute_slot, prev.minute_slot) AS minute_slot,
                coalesce(curr.slot_bar_id, prev.slot_bar_id + 1) AS slot_bar_id,
                coalesce(curr.code, prev.code) AS code,
                coalesce(curr.target_weight, 0.0) AS curr_weight,
                CASE
                    WHEN prev.slot_bar_id IS NULL THEN 0.0
                    ELSE prev.target_weight * (1.0 + prev.simple_return) / (1.0 + prev_bar.gross_return)
                END AS prev_after_weight
            FROM {position_table_name} AS prev
            FULL OUTER JOIN {position_table_name} AS curr
                ON curr.minute_slot = prev.minute_slot
               AND curr.slot_bar_id = prev.slot_bar_id + 1
               AND curr.code = prev.code
            LEFT JOIN bar_return AS prev_bar
                ON prev_bar.minute_slot = prev.minute_slot
               AND prev_bar.slot_bar_id = prev.slot_bar_id
        ),
        turnover AS (
            SELECT
                minute_slot,
                slot_bar_id,
                0.5 * sum(abs(curr_weight - prev_after_weight)) AS turnover
            FROM pair_weights
            WHERE slot_bar_id IS NOT NULL
            GROUP BY minute_slot, slot_bar_id
        )
        SELECT
            bar_return.minute_slot,
            bar_return.slot_bar_id,
            bar_return.date,
            bar_return.time,
            bar_return.gross_return,
            bar_return.long_leg_return,
            bar_return.short_leg_return,
            turnover.turnover
        FROM bar_return
        LEFT JOIN turnover
            ON turnover.minute_slot = bar_return.minute_slot
           AND turnover.slot_bar_id = bar_return.slot_bar_id
        ORDER BY bar_return.minute_slot, bar_return.slot_bar_id
        """
    )


def export_duckdb_tables(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
    decile_table_name: str,
    slot_table_name: str,
) -> dict[str, Path]:
    """Export the core tables to CSV files."""
    # Define the output file paths.
    path_map = {
        "decile_bar_returns": output_dir / "decile_bar_returns.csv",
        "slot_bar_metrics": output_dir / "slot_bar_metrics.csv",
    }

    # Write the decile table.
    con.execute(
        f"""
        COPY {decile_table_name}
        TO '{path_map["decile_bar_returns"].as_posix()}'
        (HEADER, DELIMITER ',')
        """
    )

    # Write the slot bar table.
    con.execute(
        f"""
        COPY {slot_table_name}
        TO '{path_map["slot_bar_metrics"].as_posix()}'
        (HEADER, DELIMITER ',')
        """
    )
    return path_map


def t_stat_from_series(series: pd.Series) -> float:
    """Compute the standard t-statistic for one return series."""
    # Convert the series to float values.
    values = series.astype(float).to_numpy()
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    t_stat = mean / (std / np.sqrt(values.size))
    return float(t_stat)


def cumulative_return_from_series(series: pd.Series) -> float:
    """Compound one return series into total return."""
    # Convert the series to float values.
    values = series.astype(float).to_numpy()
    wealth = np.cumprod(1.0 + values)
    return float(wealth[-1] - 1.0)


def max_drawdown_from_series(series: pd.Series) -> float:
    """Compute max drawdown from one return series."""
    # Convert the series to float values.
    values = series.astype(float).to_numpy()
    wealth = np.cumprod(1.0 + values)
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def summarize_deciles(decile_bar_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize decile returns and long-short spread."""
    # Load the per-bar decile returns.
    decile_bar = pd.read_csv(decile_bar_path)
    decile_bar = decile_bar.sort_values(["date", "time", "decile"]).reset_index(drop=True)

    # Build the decile summary table.
    decile_summary = (
        decile_bar.groupby("decile", sort=True)["mean_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_bar_return", "std": "std_bar_return", "count": "bar_count"})
    )
    decile_summary["t_stat"] = decile_bar.groupby("decile", sort=True)["mean_return"].apply(t_stat_from_series).to_numpy()
    decile_summary["cum_return"] = (
        decile_bar.groupby("decile", sort=True)["mean_return"].apply(cumulative_return_from_series).to_numpy()
    )
    decile_summary["mean_bar_return_bps"] = decile_summary["mean_bar_return"] * 1e4
    decile_summary["cum_return_pct"] = decile_summary["cum_return"] * 100.0

    # Build the spread summary table.
    pivot = decile_bar.pivot(index=["date", "time"], columns="decile", values="mean_return").reset_index()
    pivot["spread_q10_q1"] = pivot[10] - pivot[1]
    spread_summary = pd.DataFrame(
        [
            {
                "metric": "q10_minus_q1",
                "mean_bar_return": float(pivot["spread_q10_q1"].mean()),
                "std_bar_return": float(pivot["spread_q10_q1"].std(ddof=1)),
                "bar_count": int(pivot["spread_q10_q1"].shape[0]),
                "t_stat": t_stat_from_series(pivot["spread_q10_q1"]),
                "cum_return": cumulative_return_from_series(pivot["spread_q10_q1"]),
                "mean_bar_return_bps": float(pivot["spread_q10_q1"].mean() * 1e4),
                "cum_return_pct": cumulative_return_from_series(pivot["spread_q10_q1"]) * 100.0,
            }
        ]
    )
    spread_summary.to_csv(output_dir / "decile_spread_summary.csv", index=False)

    # Write the decile summary file.
    decile_summary.to_csv(output_dir / "decile_summary.csv", index=False)
    return decile_bar, pivot


def add_cost_columns(slot_bar: pd.DataFrame, cost_bps_list: list[float]) -> pd.DataFrame:
    """Attach linear trading cost columns to slot-level bar returns."""
    # Copy the source table.
    out = slot_bar.copy()

    # Add cost and net-return columns.
    for cost_bps in cost_bps_list:
        cost_name = f"cost_{int(cost_bps)}bps"
        net_name = f"net_return_{int(cost_bps)}bps"
        out[cost_name] = out["turnover"] * (cost_bps / 10000.0)
        out[net_name] = out["gross_return"] - out[cost_name]
    return out


def build_slot_daily_table(slot_bar: pd.DataFrame, cost_bps_list: list[float]) -> pd.DataFrame:
    """Aggregate slot bar returns into slot daily returns."""
    # Define the per-date compounding logic.
    group_cols = ["minute_slot", "date"]
    aggregations: dict[str, Any] = {
        "gross_return": lambda series: float(np.prod(1.0 + series.to_numpy()) - 1.0),
        "long_leg_return": lambda series: float(np.prod(1.0 + series.to_numpy()) - 1.0),
        "short_leg_return": lambda series: float(np.prod(1.0 + series.to_numpy()) - 1.0),
        "turnover": "sum",
    }
    for cost_bps in cost_bps_list:
        net_name = f"net_return_{int(cost_bps)}bps"
        aggregations[net_name] = lambda series: float(np.prod(1.0 + series.to_numpy()) - 1.0)

    # Build the daily table.
    slot_daily = slot_bar.groupby(group_cols, sort=True).agg(aggregations).reset_index()
    return slot_daily


def build_combined_daily_table(slot_daily: pd.DataFrame, cost_bps_list: list[float]) -> pd.DataFrame:
    """Combine ten non-overlapping slot books with equal capital."""
    # Build the combined daily table.
    aggregations: dict[str, Any] = {
        "gross_return": "mean",
        "long_leg_return": "mean",
        "short_leg_return": "mean",
        "turnover": "mean",
    }
    for cost_bps in cost_bps_list:
        net_name = f"net_return_{int(cost_bps)}bps"
        aggregations[net_name] = "mean"
    combined_daily = slot_daily.groupby("date", sort=True).agg(aggregations).reset_index()
    return combined_daily


def summarize_return_series(series: pd.Series, annual_days: int) -> dict[str, float]:
    """Summarize one daily return series into key backtest statistics."""
    # Convert the daily returns to float values.
    values = series.astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    sharpe = mean / std * np.sqrt(float(annual_days))

    # Compute the cumulative path metrics.
    cum_return = cumulative_return_from_series(values)
    max_drawdown = max_drawdown_from_series(values)
    t_stat = t_stat_from_series(values)
    positive_ratio = float((values > 0.0).mean())

    # Package the summary dictionary.
    return {
        "mean_daily_return": mean,
        "std_daily_return": std,
        "annualized_sharpe": float(sharpe),
        "t_stat": t_stat,
        "cum_return": cum_return,
        "max_drawdown": max_drawdown,
        "positive_day_ratio": positive_ratio,
        "day_count": float(values.shape[0]),
    }


def summarize_strategy(
    slot_bar_path: Path,
    cost_bps_list: list[float],
    annual_days: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize slot-level and combined strategy performance."""
    # Load and enrich the slot bar table.
    slot_bar = pd.read_csv(slot_bar_path).sort_values(["minute_slot", "slot_bar_id"]).reset_index(drop=True)
    slot_bar = add_cost_columns(slot_bar, cost_bps_list)

    # Build the daily tables.
    slot_daily = build_slot_daily_table(slot_bar, cost_bps_list)
    combined_daily = build_combined_daily_table(slot_daily, cost_bps_list)

    # Build the slot summary table.
    slot_summary = (
        slot_daily.groupby("minute_slot", sort=True)[["gross_return", "turnover"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    slot_summary.columns = [
        "minute_slot",
        "mean_daily_return",
        "std_daily_return",
        "day_count",
        "mean_daily_turnover",
        "std_daily_turnover",
        "turnover_day_count",
    ]
    slot_summary["t_stat"] = slot_daily.groupby("minute_slot", sort=True)["gross_return"].apply(t_stat_from_series).to_numpy()
    slot_summary["annualized_sharpe"] = (
        slot_summary["mean_daily_return"] / slot_summary["std_daily_return"] * np.sqrt(float(annual_days))
    )

    # Build the top-level strategy summary.
    strategy_summary: dict[str, Any] = {
        "gross": summarize_return_series(combined_daily["gross_return"], annual_days),
        "long_leg": summarize_return_series(combined_daily["long_leg_return"], annual_days),
        "short_leg": summarize_return_series(combined_daily["short_leg_return"], annual_days),
        "turnover": {
            "mean_daily_turnover": float(combined_daily["turnover"].mean()),
            "median_daily_turnover": float(combined_daily["turnover"].median()),
            "max_daily_turnover": float(combined_daily["turnover"].max()),
        },
    }
    for cost_bps in cost_bps_list:
        net_name = f"net_return_{int(cost_bps)}bps"
        strategy_summary[f"net_{int(cost_bps)}bps"] = summarize_return_series(combined_daily[net_name], annual_days)

    # Write the strategy data files.
    slot_bar.to_csv(output_dir / "slot_bar_metrics_enriched.csv", index=False)
    slot_daily.to_csv(output_dir / "slot_daily_metrics.csv", index=False)
    combined_daily.to_csv(output_dir / "combined_daily_metrics.csv", index=False)
    slot_summary.to_csv(output_dir / "slot_summary.csv", index=False)
    with (output_dir / "strategy_summary.yaml").open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(strategy_summary, file_obj, allow_unicode=True, sort_keys=False)
    return slot_bar, slot_daily, combined_daily, strategy_summary


def plot_decile_curves(decile_bar: pd.DataFrame, decile_pivot: pd.DataFrame, output_dir: Path) -> None:
    """Plot decile cumulative curves and long-short spread."""
    # Build the decile cumulative panel.
    ordered = decile_bar.sort_values(["date", "time", "decile"]).copy()
    ordered["timestamp"] = pd.to_datetime(
        ordered["date"].astype(str).str.zfill(6) + ordered["time"].astype(str).str.zfill(6),
        format="%y%m%d%H%M%S",
    )
    ordered["cum_wealth"] = ordered.groupby("decile", sort=True)["mean_return"].transform(lambda series: (1.0 + series).cumprod())

    # Draw the decile cumulative plot.
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    for decile in range(1, 11):
        decile_df = ordered.loc[ordered["decile"] == decile]
        axes[0].plot(decile_df["timestamp"], decile_df["cum_wealth"], label=f"Q{decile}")
    axes[0].set_title("All-Timestamp Decile Cumulative Curves")
    axes[0].set_ylabel("Wealth (log scale)")
    axes[0].set_yscale("log")
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].grid(alpha=0.25)

    # Draw the long-short spread curve.
    spread = decile_pivot.sort_values(["date", "time"]).copy()
    spread["timestamp"] = pd.to_datetime(
        spread["date"].astype(str).str.zfill(6) + spread["time"].astype(str).str.zfill(6),
        format="%y%m%d%H%M%S",
    )
    spread["spread_wealth"] = (1.0 + spread["spread_q10_q1"]).cumprod()
    axes[1].plot(spread["timestamp"], spread["spread_wealth"], color="tab:blue")
    axes[1].set_title("Q10 - Q1 Cumulative Curve")
    axes[1].set_ylabel("Wealth (log scale)")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)

    # Format both x-axes as calendar dates.
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    for axis in axes:
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)
        axis.set_xlabel("Date")

    fig.savefig(output_dir / "decile_cumulative.png", dpi=160)
    plt.close(fig)


def plot_strategy_curves(
    combined_daily: pd.DataFrame,
    slot_summary: pd.DataFrame,
    output_dir: Path,
    cost_bps_list: list[float],
) -> None:
    """Plot the daily strategy curves and slot heterogeneity."""
    # Build the cumulative daily plot.
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    daily = combined_daily.sort_values("date").copy()
    daily["timestamp"] = pd.to_datetime(daily["date"].astype(str).str.zfill(6), format="%y%m%d")
    axes[0].plot(daily["timestamp"], (1.0 + daily["gross_return"]).cumprod(), label="Gross", linewidth=2.0)
    for cost_bps in cost_bps_list:
        net_name = f"net_return_{int(cost_bps)}bps"
        axes[0].plot(daily["timestamp"], (1.0 + daily[net_name]).cumprod(), label=f"Net {int(cost_bps)}bps")
    axes[0].set_title("Combined Daily Strategy Curve")
    axes[0].set_ylabel("Wealth (log scale)")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # Format the strategy x-axis as calendar dates.
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].set_xlabel("Date")

    # Draw the slot Sharpe panel.
    slot_ranked = slot_summary.sort_values("minute_slot").copy()
    axes[1].bar(slot_ranked["minute_slot"].astype(int).to_numpy(), slot_ranked["annualized_sharpe"].to_numpy())
    axes[1].set_title("Per-Slot Annualized Sharpe")
    axes[1].set_xlabel("Minute Slot")
    axes[1].set_ylabel("Sharpe")
    axes[1].grid(alpha=0.25, axis="y")
    fig.savefig(output_dir / "strategy_curves.png", dpi=160)
    plt.close(fig)


def build_report_text(
    run_config: ResearchRunConfig,
    manifest_path: Path,
    decile_summary_path: Path,
    spread_summary_path: Path,
    slot_summary_path: Path,
    strategy_summary_path: Path,
) -> str:
    """Compose one markdown research report."""
    # Load the summary tables.
    decile_summary = pd.read_csv(decile_summary_path)
    spread_summary = pd.read_csv(spread_summary_path).iloc[0]
    slot_summary = pd.read_csv(slot_summary_path)
    strategy_summary = yaml.safe_load(strategy_summary_path.read_text(encoding="utf-8"))

    # Pick the headline rows.
    q1_row = decile_summary.loc[decile_summary["decile"] == 1].iloc[0]
    q10_row = decile_summary.loc[decile_summary["decile"] == 10].iloc[0]
    best_slot = slot_summary.sort_values("annualized_sharpe", ascending=False).iloc[0]
    worst_slot = slot_summary.sort_values("annualized_sharpe", ascending=True).iloc[0]

    # Compose the markdown lines.
    lines: list[str] = []
    lines.append(f"# {run_config.report_title}")
    lines.append("")
    lines.append("## 研究范围")
    lines.append("")
    lines.append(f"- 数据来源: `{manifest_path.as_posix()}`")
    lines.append("- 预测口径: 使用当前 `best checkpoint` 的 `test` 预测结果.")
    lines.append(f"- 收益口径: {run_config.return_description}")
    lines.append("- 信号层分析: 对全部时间点做 `10` 组分层收益.")
    lines.append("- 策略层分析: 用 `minute_slot = minute % 10` 拆成 `10` 个非重叠子策略, 每个子策略做 `top10-bottom10` 等权 long-short, 再按等资本合成.")
    lines.append("- 成本口径: 线性成本 `cost = turnover * bps / 10000`, 当前给出 `5/10/20 bps` 三档.")
    lines.append("")
    lines.append("## 回测规则")
    lines.append("")
    lines.append(f"- 交易规则: {run_config.trading_rule}")
    lines.append(f"- 换仓规则: {run_config.rebalance_rule}")
    lines.append(f"- 选股规则: {run_config.selection_rule}")
    lines.append(f"- 股票池控制: {run_config.pool_rule}")
    lines.append(f"- 滑点与成本: {run_config.slippage_rule}")
    lines.append("")
    lines.append("## 分层收益结论")
    lines.append("")
    lines.append("| 组合 | mean bar return (bps) | t-stat |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| Q1 | {q1_row['mean_bar_return_bps']:.3f} | {q1_row['t_stat']:.2f} |")
    lines.append(f"| Q10 | {q10_row['mean_bar_return_bps']:.3f} | {q10_row['t_stat']:.2f} |")
    lines.append(
        f"| Q10-Q1 | {spread_summary['mean_bar_return_bps']:.3f} | {spread_summary['t_stat']:.2f} |"
    )
    lines.append("")
    lines.append("- 这里的分层收益只用于信号诊断, 不展示累计收益, 因为相邻时间点的持有区间彼此重叠.")
    lines.append("- 如果 `Q10-Q1` 显著为正, 说明模型分数具备较清晰的横截面排序能力.")
    lines.append("")
    lines.append("## 基准策略结论")
    lines.append("")
    lines.append("| 指标 | Gross | Net 5bps | Net 10bps | Net 20bps |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    lines.append(
        "| daily mean return | "
        f"{strategy_summary['gross']['mean_daily_return'] * 100:.4f}% | "
        f"{strategy_summary['net_5bps']['mean_daily_return'] * 100:.4f}% | "
        f"{strategy_summary['net_10bps']['mean_daily_return'] * 100:.4f}% | "
        f"{strategy_summary['net_20bps']['mean_daily_return'] * 100:.4f}% |"
    )
    lines.append(
        "| annualized Sharpe | "
        f"{strategy_summary['gross']['annualized_sharpe']:.3f} | "
        f"{strategy_summary['net_5bps']['annualized_sharpe']:.3f} | "
        f"{strategy_summary['net_10bps']['annualized_sharpe']:.3f} | "
        f"{strategy_summary['net_20bps']['annualized_sharpe']:.3f} |"
    )
    lines.append(
        "| cumulative return | "
        f"{strategy_summary['gross']['cum_return'] * 100:.2f}% | "
        f"{strategy_summary['net_5bps']['cum_return'] * 100:.2f}% | "
        f"{strategy_summary['net_10bps']['cum_return'] * 100:.2f}% | "
        f"{strategy_summary['net_20bps']['cum_return'] * 100:.2f}% |"
    )
    lines.append(
        "| max drawdown | "
        f"{strategy_summary['gross']['max_drawdown'] * 100:.2f}% | "
        f"{strategy_summary['net_5bps']['max_drawdown'] * 100:.2f}% | "
        f"{strategy_summary['net_10bps']['max_drawdown'] * 100:.2f}% | "
        f"{strategy_summary['net_20bps']['max_drawdown'] * 100:.2f}% |"
    )
    lines.append(
        f"| mean daily turnover | {strategy_summary['turnover']['mean_daily_turnover']:.4f} | - | - | - |"
    )
    lines.append(
        "| positive day ratio | "
        f"{strategy_summary['gross']['positive_day_ratio'] * 100:.2f}% | "
        f"{strategy_summary['net_5bps']['positive_day_ratio'] * 100:.2f}% | "
        f"{strategy_summary['net_10bps']['positive_day_ratio'] * 100:.2f}% | "
        f"{strategy_summary['net_20bps']['positive_day_ratio'] * 100:.2f}% |"
    )
    lines.append("")
    lines.append("- `Gross` 使用 `10` 个非重叠子策略等资本合成, 因此更接近可交易组合, 而不是重叠持有期的信号曲线.")
    lines.append("- `Long leg` 和 `Short leg` 单独结果见 `strategy_summary.yaml`, 用来判断收益是否主要来自多头端还是空头端.")
    lines.append("")
    lines.append("## Sanity Check")
    lines.append("")
    lines.append(
        f"- 当前结果 headline: gross `daily mean return = {strategy_summary['gross']['mean_daily_return'] * 100:.4f}%`, "
        f"`positive day ratio = {strategy_summary['gross']['positive_day_ratio'] * 100:.2f}%`, "
        f"`max drawdown = {strategy_summary['gross']['max_drawdown'] * 100:.2f}%`."
    )
    lines.append(f"- 限制说明: {run_config.limitation_note}")
    lines.append("")
    lines.append("## 槽位异质性")
    lines.append("")
    lines.append(
        f"- 最优 minute slot: `{int(best_slot['minute_slot'])}`, annualized Sharpe = {best_slot['annualized_sharpe']:.3f}, "
        f"mean daily return = {best_slot['mean_daily_return'] * 100:.4f}%."
    )
    lines.append(
        f"- 最差 minute slot: `{int(worst_slot['minute_slot'])}`, annualized Sharpe = {worst_slot['annualized_sharpe']:.3f}, "
        f"mean daily return = {worst_slot['mean_daily_return'] * 100:.4f}%."
    )
    lines.append("- 如果槽位差异很大, 说明当前信号可能和日内时间结构强相关, 下一步应做 `time-of-day` 条件化分析.")
    lines.append("")
    lines.append("## 产物")
    lines.append("")
    lines.append(f"- 分层图: `{(run_config.output_dir / 'decile_cumulative.png').as_posix()}`")
    lines.append(f"- 策略图: `{(run_config.output_dir / 'strategy_curves.png').as_posix()}`")
    lines.append(f"- 分层明细: `{decile_summary_path.as_posix()}`")
    lines.append(f"- 槽位明细: `{slot_summary_path.as_posix()}`")
    lines.append(f"- 策略摘要: `{strategy_summary_path.as_posix()}`")
    lines.append("")
    lines.append("## 下一步建议")
    lines.append("")
    lines.append("- 如果 `Gross` 明显为正而 `Net 10bps` 接近为零, 下一步优先做 `buffer` 和降频调仓, 而不是继续堆模型复杂度.")
    lines.append("- 如果 `Short leg` 明显强于 `Long leg`, 可以单独研究空头端 alpha 是否更稳, 并评估 borrow 和 short constraint.")
    lines.append("- 如果分钟槽位差异很大, 应考虑只保留高质量时段, 或者让调仓频率与 horizon 更严格对齐.")
    return "\n".join(lines)


def write_report(run_config: ResearchRunConfig, manifest_path: Path) -> Path:
    """Write one markdown research report from generated summaries."""
    # Resolve the summary file paths.
    decile_summary_path = run_config.output_dir / "decile_summary.csv"
    spread_summary_path = run_config.output_dir / "decile_spread_summary.csv"
    slot_summary_path = run_config.output_dir / "slot_summary.csv"
    strategy_summary_path = run_config.output_dir / "strategy_summary.yaml"

    # Compose and write the report.
    report_text = build_report_text(
        run_config,
        manifest_path,
        decile_summary_path,
        spread_summary_path,
        slot_summary_path,
        strategy_summary_path,
    )
    report_path = run_config.output_dir / "research_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def short_date_to_full_date(short_date: int) -> int:
    """Convert one YYMMDD integer into YYYYMMDD integer."""
    # Reconstruct the full calendar date.
    return 20000000 + int(short_date)


def short_date_to_raw_path(raw_root: Path, short_date: int) -> Path:
    """Resolve the raw stock1m feather path for one trade date."""
    # Build the full date and year folder.
    full_date = short_date_to_full_date(short_date)
    year = full_date // 10000
    return raw_root / str(year) / f"{full_date}.feather"


def load_allowed_times(manifest_path: Path) -> list[int]:
    """Load the model-supported trading times from prediction output."""
    # Read the parquet source once.
    parquet_glob = load_manifest_glob(manifest_path)
    con = duckdb.connect()
    allowed_times = con.execute(
        f"""
        SELECT DISTINCT time
        FROM read_parquet('{parquet_glob}')
        ORDER BY time
        """
    ).fetchnumpy()["time"].tolist()
    con.close()
    return [int(time_value) for time_value in allowed_times]


def load_daily_trade_state(raw_root: Path, short_date: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one daily raw universe and one daily next-trade lookup."""
    # Read the minimal raw columns.
    raw_path = short_date_to_raw_path(raw_root, short_date)
    raw_df = pd.read_feather(raw_path, columns=["StockCode", "DateTime", "MinuteIndex", "Open", "Vol", "Amount"])
    raw_df = raw_df.sort_values(["StockCode", "MinuteIndex"]).reset_index(drop=True)

    # Build the base time fields.
    raw_df["time"] = (
        raw_df["DateTime"].dt.hour.astype(np.int32) * 10000
        + raw_df["DateTime"].dt.minute.astype(np.int32) * 100
        + raw_df["DateTime"].dt.second.astype(np.int32)
    )
    raw_df["current_tradable"] = (
        (raw_df["Open"].astype(np.float64) > 0.0)
        & (raw_df["Vol"].astype(np.float64) > 0.0)
        & (raw_df["Amount"].astype(np.float64) > 0.0)
    )

    # Build the next-tradable lookup by stock.
    tradable_open = raw_df["Open"].where(raw_df["current_tradable"])
    tradable_minute = raw_df["MinuteIndex"].where(raw_df["current_tradable"])
    raw_df["next_trade_open"] = tradable_open.groupby(raw_df["StockCode"], sort=False).transform(
        lambda series: series.iloc[::-1].ffill().iloc[::-1]
    )
    raw_df["next_trade_minute"] = tradable_minute.groupby(raw_df["StockCode"], sort=False).transform(
        lambda series: series.iloc[::-1].ffill().iloc[::-1]
    )

    # Materialize the current universe slice.
    current_universe = raw_df.loc[
        :,
        ["StockCode", "time", "MinuteIndex", "Open", "Vol", "Amount", "current_tradable"],
    ].rename(
        columns={
            "StockCode": "code",
            "MinuteIndex": "base_minute",
            "Open": "current_open",
            "Vol": "current_vol",
            "Amount": "current_amount",
        }
    )

    # Materialize the next-tradable execution lookup.
    next_trade_lookup = raw_df.loc[
        :,
        ["StockCode", "MinuteIndex", "next_trade_minute", "next_trade_open"],
    ].rename(columns={"StockCode": "code", "MinuteIndex": "schedule_minute"})
    next_trade_lookup["schedule_minute"] = next_trade_lookup["schedule_minute"].astype(np.int32, copy=False)
    return current_universe, next_trade_lookup


def get_cached_daily_trade_state(
    cache: OrderedDict[int, tuple[pd.DataFrame, pd.DataFrame]],
    raw_root: Path,
    short_date: int,
    lookup_cache_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch one daily trade-state pair from a small LRU cache."""
    # Reuse the cached lookups when available.
    if short_date in cache:
        current_universe, next_trade_lookup = cache.pop(short_date)
        cache[short_date] = (current_universe, next_trade_lookup)
        return current_universe, next_trade_lookup

    # Load the missing date from disk.
    current_universe, next_trade_lookup = load_daily_trade_state(raw_root, short_date)
    cache[short_date] = (current_universe, next_trade_lookup)

    # Evict the oldest cached date.
    if len(cache) > lookup_cache_size:
        cache.popitem(last=False)
    return current_universe, next_trade_lookup


def build_execution_safe_day_frame(
    pred_day: pd.DataFrame,
    current_universe: pd.DataFrame,
    next_trade_lookup: pd.DataFrame,
    allowed_times: set[int],
    entry_delay_bars: int,
    holding_bars: int,
    prediction_fill_value: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build one raw-universe-based execution-safe day slice."""
    # Restrict the raw universe to model-supported times.
    universe = current_universe.loc[current_universe["time"].isin(allowed_times)].copy()
    universe_rows = int(universe.shape[0])

    # Filter to names tradable at signal time.
    universe = universe.loc[universe["current_tradable"]].copy()
    tradable_rows = int(universe.shape[0])

    # Attach predictions without using future-label availability.
    merged = universe.merge(pred_day, on=["code", "time"], how="left")
    prediction_hit_rows = int(merged["prediction"].notna().sum())
    merged["prediction"] = merged["prediction"].fillna(prediction_fill_value)
    merged["date"] = int(pred_day["date"].iloc[0])

    # Attach the earliest tradable entry after the signal bar.
    merged["entry_schedule_minute"] = merged["base_minute"] + entry_delay_bars
    entry_lookup = next_trade_lookup.rename(
        columns={
            "schedule_minute": "entry_schedule_minute",
            "next_trade_minute": "entry_exec_minute",
            "next_trade_open": "entry_open",
        }
    )
    merged = merged.merge(entry_lookup, on=["code", "entry_schedule_minute"], how="left")

    # Attach the earliest tradable exit after the holding horizon.
    merged["exit_schedule_minute"] = merged["entry_exec_minute"] + holding_bars
    exit_lookup = next_trade_lookup.rename(
        columns={
            "schedule_minute": "exit_schedule_minute",
            "next_trade_minute": "exit_exec_minute",
            "next_trade_open": "exit_open",
        }
    )
    merged = merged.merge(exit_lookup, on=["code", "exit_schedule_minute"], how="left")

    # Compute the execution-safe simple return with cash fallback on missing fills.
    entry_filled = merged["entry_open"].notna()
    exit_filled = merged["exit_open"].notna()
    merged["simple_return"] = 0.0
    valid_trade = entry_filled & exit_filled
    merged.loc[valid_trade, "simple_return"] = (
        merged.loc[valid_trade, "exit_open"].astype(np.float64)
        / merged.loc[valid_trade, "entry_open"].astype(np.float64)
        - 1.0
    )

    # Keep only the backtest columns.
    out = merged.loc[:, ["code", "date", "time", "prediction", "simple_return"]].copy()

    # Build the day-level audit.
    audit = {
        "date": float(pred_day["date"].iloc[0]),
        "raw_universe_rows": float(universe_rows),
        "tradable_universe_rows": float(tradable_rows),
        "prediction_hit_rows": float(prediction_hit_rows),
        "prediction_coverage_ratio": float(prediction_hit_rows / tradable_rows),
        "entry_fill_ratio": float(entry_filled.mean()),
        "exit_fill_ratio": float(exit_filled.mean()),
        "zero_return_fallback_ratio": float((~valid_trade).mean()),
    }
    return out, audit


def materialize_execution_safe_open_chunks(config: BacktestConfig, chunk_paths: list[Path]) -> str:
    """Materialize raw-universe-based execution-safe parquet chunks."""
    # Prepare the output directory tree.
    ensure_output_dir(config.safe_output_dir)
    ensure_output_dir(config.safe_chunk_dir)
    for old_path in sorted(config.safe_chunk_dir.glob("*.parquet")):
        old_path.unlink()

    # Resolve the model-supported trading times once.
    allowed_times = set(load_allowed_times(config.manifest_path))

    # Process each prediction chunk once.
    lookup_cache: OrderedDict[int, tuple[pd.DataFrame, pd.DataFrame]] = OrderedDict()
    output_chunk_paths: list[Path] = []
    audit_rows: list[dict[str, float]] = []
    carry_date: int | None = None
    carry_frame: pd.DataFrame | None = None
    for chunk_idx, chunk_path in enumerate(chunk_paths, start=1):
        print(f"[execution-safe] chunk {chunk_idx}/{len(chunk_paths)} -> {chunk_path.name}", flush=True)
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

        # Build the execution-safe chunk day by day.
        day_frames: list[pd.DataFrame] = []
        for short_date, pred_day in pred_chunk.groupby("date", sort=True):
            current_universe, next_trade_lookup = get_cached_daily_trade_state(
                lookup_cache,
                config.raw_root,
                int(short_date),
                config.lookup_cache_size,
            )
            safe_day, audit = build_execution_safe_day_frame(
                pred_day.reset_index(drop=True),
                current_universe,
                next_trade_lookup,
                allowed_times,
                config.entry_delay_bars,
                config.holding_bars,
                config.prediction_fill_value,
            )
            day_frames.append(safe_day)
            audit_rows.append(audit)

        # Write the execution-safe chunk.
        if day_frames:
            safe_chunk = pd.concat(day_frames, axis=0, ignore_index=True)
            output_chunk_path = config.safe_chunk_dir / chunk_path.name
            safe_chunk.to_parquet(output_chunk_path, index=False)
            output_chunk_paths.append(output_chunk_path)

    # Process the final carried date once.
    final_universe, final_next_trade_lookup = get_cached_daily_trade_state(
        lookup_cache,
        config.raw_root,
        int(carry_date),
        config.lookup_cache_size,
    )
    final_safe_day, final_audit = build_execution_safe_day_frame(
        carry_frame.reset_index(drop=True),
        final_universe,
        final_next_trade_lookup,
        allowed_times,
        config.entry_delay_bars,
        config.holding_bars,
        config.prediction_fill_value,
    )
    final_output_chunk_path = config.safe_chunk_dir / f"part_{len(chunk_paths):06d}_tail.parquet"
    final_safe_day.to_parquet(final_output_chunk_path, index=False)
    output_chunk_paths.append(final_output_chunk_path)
    audit_rows.append(final_audit)

    # Write the execution-safe manifest and audits.
    audit_df = pd.DataFrame(audit_rows).sort_values("date").reset_index(drop=True)
    audit_df.to_csv(config.safe_output_dir / "execution_safe_open_audit.csv", index=False)
    summary = {
        "entry_delay_bars": config.entry_delay_bars,
        "holding_bars": config.holding_bars,
        "prediction_fill_value": float(config.prediction_fill_value),
        "allowed_time_count": int(len(allowed_times)),
        "date_count": int(audit_df.shape[0]),
        "raw_universe_rows_total": int(audit_df["raw_universe_rows"].sum()),
        "tradable_universe_rows_total": int(audit_df["tradable_universe_rows"].sum()),
        "prediction_hit_rows_total": int(audit_df["prediction_hit_rows"].sum()),
        "prediction_coverage_ratio_total": float(
            audit_df["prediction_hit_rows"].sum() / audit_df["tradable_universe_rows"].sum()
        ),
        "entry_fill_ratio_mean": float(audit_df["entry_fill_ratio"].mean()),
        "exit_fill_ratio_mean": float(audit_df["exit_fill_ratio"].mean()),
        "zero_return_fallback_ratio_mean": float(audit_df["zero_return_fallback_ratio"].mean()),
    }
    with (config.safe_output_dir / "execution_safe_open_audit.yaml").open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(summary, file_obj, allow_unicode=True, sort_keys=False)
    write_manifest(config.safe_manifest_path, output_chunk_paths)
    return load_manifest_glob(config.safe_manifest_path)


def write_comparison_summary(optimistic_dir: Path, safe_dir: Path, output_path: Path) -> None:
    """Write one optimistic-vs-safe comparison summary."""
    # Load the strategy and spread summaries.
    optimistic_strategy = yaml.safe_load((optimistic_dir / "strategy_summary.yaml").read_text(encoding="utf-8"))
    safe_strategy = yaml.safe_load((safe_dir / "strategy_summary.yaml").read_text(encoding="utf-8"))
    optimistic_spread = pd.read_csv(optimistic_dir / "decile_spread_summary.csv").iloc[0]
    safe_spread = pd.read_csv(safe_dir / "decile_spread_summary.csv").iloc[0]

    # Build the comparison payload.
    payload = {
        "signal": {
            "optimistic_q10_q1_bps": float(optimistic_spread["mean_bar_return_bps"]),
            "safe_q10_q1_bps": float(safe_spread["mean_bar_return_bps"]),
            "optimistic_q10_q1_tstat": float(optimistic_spread["t_stat"]),
            "safe_q10_q1_tstat": float(safe_spread["t_stat"]),
        },
        "gross": {
            "optimistic_mean_daily_return": float(optimistic_strategy["gross"]["mean_daily_return"]),
            "safe_mean_daily_return": float(safe_strategy["gross"]["mean_daily_return"]),
            "optimistic_sharpe": float(optimistic_strategy["gross"]["annualized_sharpe"]),
            "safe_sharpe": float(safe_strategy["gross"]["annualized_sharpe"]),
        },
        "net_10bps": {
            "optimistic_mean_daily_return": float(optimistic_strategy["net_10bps"]["mean_daily_return"]),
            "safe_mean_daily_return": float(safe_strategy["net_10bps"]["mean_daily_return"]),
            "optimistic_sharpe": float(optimistic_strategy["net_10bps"]["annualized_sharpe"]),
            "safe_sharpe": float(safe_strategy["net_10bps"]["annualized_sharpe"]),
        },
    }

    # Serialize the comparison file.
    with output_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, allow_unicode=True, sort_keys=False)


def run_one_research(config: BacktestConfig, run_config: ResearchRunConfig, manifest_path: Path) -> Path:
    """Run one full backtest research pipeline from one parquet source."""
    # Prepare the output directory and database.
    ensure_output_dir(run_config.output_dir)
    con = connect_duckdb(run_config.db_path)

    # Materialize and export the research tables.
    materialize_decile_table(con, run_config.parquet_glob, run_config.return_expr_sql, "decile_bar_returns")
    materialize_slot_strategy_tables(
        con,
        run_config.parquet_glob,
        config.top_frac,
        run_config.return_expr_sql,
        "slot_positions",
        "slot_bar_metrics",
    )
    export_duckdb_tables(con, run_config.output_dir, "decile_bar_returns", "slot_bar_metrics")
    con.close()

    # Summarize the signal and strategy results.
    decile_bar, decile_pivot = summarize_deciles(run_config.output_dir / "decile_bar_returns.csv", run_config.output_dir)
    _, _, combined_daily, _ = summarize_strategy(
        run_config.output_dir / "slot_bar_metrics.csv",
        config.cost_bps_list,
        config.annual_days,
        run_config.output_dir,
    )
    slot_summary = pd.read_csv(run_config.output_dir / "slot_summary.csv")

    # Draw the research figures.
    plot_decile_curves(decile_bar, decile_pivot, run_config.output_dir)
    plot_strategy_curves(combined_daily, slot_summary, run_config.output_dir, config.cost_bps_list)

    # Write the markdown report.
    report_path = write_report(run_config, manifest_path)
    return report_path


def main() -> None:
    """Run the optimistic and execution-safe backtest research pipeline."""
    # Build the fixed configuration.
    config = build_config()
    ensure_output_dir(config.output_dir)

    # Resolve the optimistic prediction source.
    optimistic_glob = load_manifest_glob(config.manifest_path)
    optimistic_run = ResearchRunConfig(
        parquet_glob=optimistic_glob,
        output_dir=config.output_dir,
        db_path=config.optimistic_db_path,
        report_title=config.report_title,
        return_expr_sql="exp(target) - 1.0",
        return_description="将 label 视为 `10 min forward log-return`, 回测时统一转成 `simple return = exp(target) - 1`.",
        limitation_note="当前口径仍然偏 optimistic, 因为执行价与 future label 可用性筛样都未完全实盘化.",
        trading_rule="每个信号 bar 在同一分钟完成排序, 直接把 `exp(target)-1` 视为该 bar 的可实现收益, 仅用于和修复后版本做对照.",
        rebalance_rule="按 `minute % 10` 切成 10 个非重叠子组合, 每个子组合在自己的触发时点调仓, 日内等资本合成.",
        selection_rule="每个 `(date,time)` 横截面按 prediction 从低到高排序, 做 bottom 10% short 和 top 10% long, 两端等权各 50%.",
        pool_rule="直接使用 prediction parquet 内已有样本作为股票池, 不额外施加当前 bar tradability 过滤.",
        slippage_rule="净值场景使用 `turnover * bps / 10000` 的线性 all-in 成本, `5/10/20 bps` 代表费用与滑点合计假设.",
    )
    optimistic_report_path = run_one_research(config, optimistic_run, config.manifest_path)

    # Build the execution-safe source first.
    chunk_paths = load_manifest_chunk_paths(config.manifest_path)
    safe_glob = materialize_execution_safe_open_chunks(config, chunk_paths)
    safe_run = ResearchRunConfig(
        parquet_glob=safe_glob,
        output_dir=config.safe_output_dir,
        db_path=config.safe_db_path,
        report_title=config.safe_report_title,
        return_expr_sql="simple_return",
        return_description="先用 raw `stock1m` 在模型支持的时刻重建当下股票池, 再用 `Open_(t+1)` 入场、`持有 10 个可交易 bar` 后的下一次可交易 `Open` 出场, 未成交样本记为现金零收益.",
        limitation_note="该口径修复了 `future-dependent universe` 和基础 `tradability filter`, 但还没有加入涨跌停、借券、冲击成本和行业/风格中性约束.",
        trading_rule="信号在 `t` 生成, 不用 `Close_t -> Close_t+10` 直接成交; 实际入场价取 `t+1` 之后首个可交易 `Open`, 出场价取持有 10 个 bar 后首个可交易 `Open`.",
        rebalance_rule="按 `minute % 10` 切成 10 个非重叠子组合, 每个子组合只在自己的分钟槽位换仓; 若个股在计划时点不可成交, 对应仓位视为现金零收益.",
        selection_rule="每个 `(date,time)` 的当前 tradable universe 上按 prediction 排序, 做 bottom 10% short 和 top 10% long, 两端等权各 50%; 缺失 prediction 的股票用 0 分填充后参与排序.",
        pool_rule="股票池由 raw `stock1m` 在模型支持时刻的全量股票重建, 仅要求当前 bar `Open>0, Vol>0, Amount>0`, 不再依赖 future label 是否可算.",
        slippage_rule="净值场景仍使用 `turnover * bps / 10000` 的线性 all-in 成本, `5/10/20 bps` 统一代表手续费与滑点合计压力测试.",
    )
    safe_report_path = run_one_research(config, safe_run, config.safe_manifest_path)

    # Write the optimistic-vs-safe comparison file.
    write_comparison_summary(
        config.output_dir,
        config.safe_output_dir,
        config.output_dir / "execution_safe_comparison.yaml",
    )
    print(optimistic_report_path.as_posix())
    print(safe_report_path.as_posix())


if __name__ == "__main__":
    main()
