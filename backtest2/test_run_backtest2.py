"""Unit tests for backtest2 core data-join and state-machine behavior."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    """Load the backtest2 script as a Python module."""
    # Register the module in sys.modules before executing dataclass definitions.
    module_path = Path("/home/maomao/prediction-NN-2/backtest2/run_backtest2.py")
    spec = importlib.util.spec_from_file_location("backtest2_run", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_feature_day_frame_keeps_unpredicted_universe_rows():
    """Ensure the feature join keeps raw-universe rows without predictions."""
    # Load the target module once for this test.
    mod = _load_module()

    # Build one tiny signal universe with one missing prediction.
    pred_day = pd.DataFrame(
        {
            "prediction": [0.8],
            "code": [1],
            "date": [260101],
            "time": [93000],
        }
    )
    universe = pd.DataFrame(
        {
            "code": [1, 2],
            "time": [93000, 93000],
            "minute_slot": [0, 0],
            "base_minute": [0, 0],
            "signal_open": [10.0, 20.0],
            "signal_close": [10.1, 20.1],
            "signal_vol": [100.0, 200.0],
            "signal_amount": [1000.0, 2000.0],
            "adv_amount": [5000.0, 6000.0],
            "current_tradable": [True, True],
            "sigma_intraday": [0.02, 0.03],
        }
    )

    # Build one tiny next-trade lookup with valid entry and exit bars.
    next_trade = pd.DataFrame(
        {
            "code": [1, 1, 2, 2],
            "schedule_minute": [1, 11, 1, 11],
            "next_trade_minute": [1, 11, 1, 11],
            "next_trade_open": [10.2, 10.4, 20.2, 20.4],
            "next_trade_vwap": [10.25, 10.45, 20.25, 20.45],
        }
    )

    # Run the feature join and verify the output semantics.
    out, audit = mod.build_feature_day_frame(pred_day, universe, next_trade, {93000}, 1, 10)
    assert list(out["code"]) == [1, 2]
    assert int(out["prediction_available"].sum()) == 1
    assert np.isnan(float(out.loc[out["code"] == 2, "prediction"].iloc[0]))
    assert "strategy_tradable" not in list(out.columns)
    assert bool(out.loc[out["code"] == 1, "fillable_open"].iloc[0]) is True
    assert int(audit["prediction_rows"]) == 1


def test_simulate_strategy_bars_keeps_unfilled_targets_as_cash(tmp_path: Path):
    """Ensure the state machine only executes fillable targets and keeps the rest in cash."""
    # Load the target module once for this test.
    mod = _load_module()

    # Build one two-bar position table with one unfilled short leg on bar one.
    position_df = pd.DataFrame(
        {
            "date": [260101, 260101, 260101, 260101],
            "time": [93000, 93000, 94000, 94000],
            "minute_slot": [0, 0, 0, 0],
            "slot_bar_id": [1, 1, 2, 2],
            "code": [1, 2, 1, 2],
            "prediction": [0.8, -0.8, 0.7, -0.7],
            "prediction_available": [True, True, True, True],
            "current_tradable": [True, True, True, True],
            "simple_return": [0.10, -0.05, 0.00, 0.00],
            "sigma_intraday": [0.02, 0.02, 0.02, 0.02],
            "adv_amount": [10_000.0, 10_000.0, 10_000.0, 10_000.0],
            "fillable": [True, False, True, True],
            "spread_bps": [5.0, 5.0, 5.0, 5.0],
            "side": [1, -1, 1, -1],
            "target_weight": [0.5, -0.5, 0.5, -0.5],
        }
    )
    csv_path = Path(tmp_path) / "positions.csv"
    position_df.to_csv(csv_path, index=False)

    # Run the state-machine simulation and verify execution-level accounting.
    slot_bar = mod.simulate_strategy_bars(csv_path, 0.5)
    first_bar = slot_bar.loc[slot_bar["slot_bar_id"] == 1].iloc[0]
    second_bar = slot_bar.loc[slot_bar["slot_bar_id"] == 2].iloc[0]
    assert np.isclose(float(first_bar["planned_turnover"]), 0.5)
    assert np.isclose(float(first_bar["turnover"]), 0.25)
    assert np.isclose(float(first_bar["fill_ratio"]), 0.5)
    assert np.isclose(float(first_bar["executed_gross_exposure"]), 0.5)
    assert np.isclose(float(first_bar["cash_buffer"]), 0.5)
    assert float(second_bar["turnover"]) > 0.25
