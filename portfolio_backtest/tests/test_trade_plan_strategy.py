"""Regression tests for trade-plan portfolio construction variants."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path("/home/maomao/prediction-NN-2")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio_backtest.simulator import connect_duckdb, materialize_slot_strategy_table


def _write_tiny_feature_parquet(path: Path) -> None:
    """Write one tiny feature parquet with enough rows for ranking tests."""
    # Build one synthetic timestamp with monotonic prediction and liquidity.
    rows: list[dict[str, object]] = []
    for idx in range(30):
        rows.append(
            {
                "date": 260101,
                "time": 93000,
                "code": int(idx + 1),
                "minute_slot": 0,
                "prediction": float(idx),
                "prediction_available": True,
                "current_tradable": True,
                "signal_amount": float(30 - idx),
                "sigma_intraday": 0.01,
                "adv_amount": 1_000_000.0,
                "is_limit_up_all_day": False,
                "is_limit_down_all_day": False,
                "fillable_vwap": True,
                "ret_vwap_exec_10": 0.001,
                "entry_vwap_is_up_limit": False,
                "entry_vwap_is_down_limit": False,
                "exit_vwap_is_up_limit": False,
                "exit_vwap_is_down_limit": False,
            }
        )

    # Persist the rows as a parquet file consumed by DuckDB.
    pd.DataFrame(rows).to_parquet(path, index=False)


def _materialize_case(tmp_path: Path, table_name: str, long_enabled: bool, short_enabled: bool, max_liq_bucket: int) -> pd.DataFrame:
    """Materialize one tiny strategy table and return exposure aggregates."""
    # Prepare the tiny feature source and DuckDB connection.
    feature_path = Path(tmp_path) / "features.parquet"
    _write_tiny_feature_parquet(feature_path)
    con = connect_duckdb(Path(tmp_path) / f"{table_name}.duckdb")

    # Run the production table builder with the requested side and liquidity knobs.
    materialize_slot_strategy_table(
        con,
        feature_path.as_posix(),
        0.10,
        bool(long_enabled),
        bool(short_enabled),
        int(max_liq_bucket),
        "ret_vwap_exec_10",
        "fillable_vwap",
        "entry_vwap_is_up_limit",
        "entry_vwap_is_down_limit",
        "exit_vwap_is_up_limit",
        "exit_vwap_is_down_limit",
        "current_tradable = true AND prediction_available = true AND adv_amount IS NOT NULL AND adv_amount > 0 AND sigma_intraday IS NOT NULL",
        5.0,
        10.0,
        20.0,
        table_name,
    )

    # Return compact exposure aggregates for assertions.
    out = con.execute(
        f"""
        SELECT
            count(*) AS n,
            sum(target_weight) AS net,
            sum(abs(target_weight)) AS gross
        FROM {table_name}
        """
    ).fetchdf()
    con.close()
    return out


def test_long_short_variant_is_market_neutral(tmp_path: Path) -> None:
    """Ensure the default long-short variant has one gross and near-zero net exposure."""
    # Build the default long-short table and assert exposure accounting.
    out = _materialize_case(tmp_path, "long_short", True, True, 3)
    row = out.iloc[0]
    assert int(row["n"]) == 6
    assert abs(float(row["net"])) < 1e-12
    assert abs(float(row["gross"]) - 1.0) < 1e-12


def test_single_leg_variants_use_full_gross(tmp_path: Path) -> None:
    """Ensure long-only and short-only variants allocate one gross to the enabled side."""
    # Build long-only and short-only tables and assert directional exposure.
    long_out = _materialize_case(tmp_path, "long_only", True, False, 3).iloc[0]
    short_out = _materialize_case(tmp_path, "short_only", False, True, 3).iloc[0]
    assert abs(float(long_out["net"]) - 1.0) < 1e-12
    assert abs(float(long_out["gross"]) - 1.0) < 1e-12
    assert abs(float(short_out["net"]) + 1.0) < 1e-12
    assert abs(float(short_out["gross"]) - 1.0) < 1e-12


def test_liquidity_filter_reranks_before_selection(tmp_path: Path) -> None:
    """Ensure liquidity filtering happens before top-bottom selection."""
    # Restrict to the highest liquidity bucket and require the remaining universe to rebalance both sides.
    out = _materialize_case(tmp_path, "liq_one", True, True, 1)
    row = out.iloc[0]
    assert int(row["n"]) == 2
    assert abs(float(row["net"])) < 1e-12
    assert abs(float(row["gross"]) - 1.0) < 1e-12
