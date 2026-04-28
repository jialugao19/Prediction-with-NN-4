"""Execution-aware portfolio backtest package."""

from portfolio_backtest.contract import (
    BAR_OUTPUT_COLUMNS,
    INFERENCE_MANIFEST_COLUMNS,
    POSITION_TABLE_COLUMNS,
    PortfolioBacktestConfig,
    PortfolioBacktestInputContract,
    PortfolioBacktestOutputContract,
    PortfolioBacktestRuntimeContract,
    build_default_portfolio_backtest_config,
    build_input_contract,
    build_output_contract,
    ensure_output_dir,
    load_inference_manifest,
    validate_required_artifacts,
    write_runtime_contract,
)
from portfolio_backtest.run_portfolio_backtest import run_portfolio_backtest

__all__ = [
    "BAR_OUTPUT_COLUMNS",
    "INFERENCE_MANIFEST_COLUMNS",
    "POSITION_TABLE_COLUMNS",
    "PortfolioBacktestConfig",
    "PortfolioBacktestInputContract",
    "PortfolioBacktestOutputContract",
    "PortfolioBacktestRuntimeContract",
    "build_default_portfolio_backtest_config",
    "build_input_contract",
    "build_output_contract",
    "ensure_output_dir",
    "load_inference_manifest",
    "run_portfolio_backtest",
    "validate_required_artifacts",
    "write_runtime_contract",
]
