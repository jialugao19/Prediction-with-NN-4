"""Run the execution-aware portfolio backtest using formal inference manifests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure the repo root is importable when this file is executed as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio_backtest.contract import PortfolioBacktestConfig, build_default_portfolio_backtest_config, ensure_output_dir, validate_required_artifacts, write_runtime_contract
from portfolio_backtest.data_source import materialize_feature_chunks
from portfolio_backtest.reporting import build_report_text, build_self_contained_html_report, plot_capacity_sweep, plot_drawdown_curve, plot_slot_sharpe, plot_strategy_curves
from portfolio_backtest.simulator import (
    connect_duckdb,
    export_duckdb_table,
    materialize_slot_strategy_table,
    summarize_baseline_open_strategy,
    summarize_realistic_vwap_strategy,
)


def run_portfolio_backtest(config: PortfolioBacktestConfig | None = None) -> Path:
    """Run the full portfolio backtest pipeline and return the report path."""
    # Resolve the canonical configuration and output directory.
    if config is None:
        config = build_default_portfolio_backtest_config()
    ensure_output_dir(config.output_dir)
    write_runtime_contract(config.output_dir)

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
        "entry_open_is_up_limit",
        "entry_open_is_down_limit",
        "exit_open_is_up_limit",
        "exit_open_is_down_limit",
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
        "entry_vwap_is_up_limit",
        "entry_vwap_is_down_limit",
        "exit_vwap_is_up_limit",
        "exit_vwap_is_down_limit",
        "current_tradable = true AND prediction_available = true AND adv_amount IS NOT NULL AND adv_amount > 0 AND sigma_intraday IS NOT NULL",
        config.spread_bps_high,
        config.spread_bps_mid,
        config.spread_bps_low,
        "vwap_slot_positions",
    )

    # Export the target-position tables to CSV for pandas simulation.
    open_position_csv = export_duckdb_table(con, config.output_dir, "open_slot_positions", "open_slot_positions.parquet")
    vwap_position_csv = export_duckdb_table(con, config.output_dir, "vwap_slot_positions", "vwap_slot_positions.parquet")
    con.close()

    # Summarize the baseline open and realistic vwap strategies.
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

    # Build the strategy summary payload and write it to YAML.
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
            "aum_list": [float(x) for x in list(config.aum_list)],
        },
    }
    strategy_summary_path = config.output_dir / "strategy_summary.yaml"
    strategy_summary_path.write_text(yaml.safe_dump(strategy_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # Compose and write the final markdown report.
    report_text = build_report_text(config, strategy_summary_path)
    report_path = config.output_dir / "research_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    # Build and write the self-contained HTML report alongside the markdown file.
    report_html = build_self_contained_html_report(config, strategy_summary_path)
    report_html_path = config.output_dir / "research_report.html"
    report_html_path.write_text(report_html, encoding="utf-8")

    # Validate the output contract before returning.
    validate_required_artifacts(config.output_dir)
    return report_path


def main() -> None:
    """Run the portfolio backtest as a script entry."""
    # Run the full pipeline and print the report path.
    report_path = run_portfolio_backtest()
    print(Path(report_path).as_posix())


if __name__ == "__main__":
    main()
