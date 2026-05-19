"""Run cost-aware entry-score experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio_backtest.contract import load_chunk_manifest_glob
from portfolio_backtest.simulator import connect_duckdb, summarize_return_series
from prediction_nn2.html_report import build_page, render_code_block, render_section, render_table


EXPERIMENT_ROOT = Path("/data-cache/nn/trade_plan_experiments/0516_cost_aware_entry")
REPO_REPORT_DIR = REPO_ROOT / "report" / "0516"
FEATURE_MANIFEST_PATH = Path("/data-cache/nn/trade_plan_experiments/0515/features/entry1_h60_slot60/feature_manifest.yaml")
AUM_FOR_NET = 10_000_000.0
ANNUAL_DAYS = 252
ENTRY_FRAC = 0.10
HOLD_FRAC = 0.20
SLOT_MOD_BARS = 60
MAX_LIQ_BUCKET = 1
IMPACT_ETA = 0.50
SPREAD_BPS_HIGH = 5.0


def build_policy_variants() -> list[dict[str, object]]:
    """Define the fixed-liq1 cost-aware entry-score matrix."""
    # Keep full-exposure sizing and compare ranking penalties for expected trading cost.
    variants: list[dict[str, object]] = [
        {
            "strategy_name": "equal_score_cost_sizing_liq1_buffer",
            "sizing_method": "equal",
            "turnover_budget": 0.0,
            "no_trade_band": 0.0,
            "entry_cost_penalty": 0.0,
        },
        {
            "strategy_name": "entry_cost000_cost_sizing_liq1_buffer",
            "sizing_method": "cost_aware",
            "turnover_budget": 0.0,
            "no_trade_band": 0.0,
            "entry_cost_penalty": 0.0,
        },
        {
            "strategy_name": "entry_cost025_cost_sizing_liq1_buffer",
            "sizing_method": "cost_aware",
            "turnover_budget": 0.0,
            "no_trade_band": 0.0,
            "entry_cost_penalty": 0.00025,
        },
        {
            "strategy_name": "entry_cost050_cost_sizing_liq1_buffer",
            "sizing_method": "cost_aware",
            "turnover_budget": 0.0,
            "no_trade_band": 0.0,
            "entry_cost_penalty": 0.00050,
        },
        {
            "strategy_name": "entry_cost100_cost_sizing_liq1_buffer",
            "sizing_method": "cost_aware",
            "turnover_budget": 0.0,
            "no_trade_band": 0.0,
            "entry_cost_penalty": 0.00100,
        },
        {
            "strategy_name": "entry_cost150_cost_sizing_liq1_buffer",
            "sizing_method": "cost_aware",
            "turnover_budget": 0.0,
            "no_trade_band": 0.0,
            "entry_cost_penalty": 0.00150,
        },
    ]

    # Attach the common construction settings to every row.
    for variant in variants:
        variant["entry_frac"] = float(ENTRY_FRAC)
        variant["hold_frac"] = float(HOLD_FRAC)
        variant["holding_bars"] = 60
        variant["slot_mod_bars"] = int(SLOT_MOD_BARS)
        variant["max_liq_bucket"] = int(MAX_LIQ_BUCKET)
    return variants


def load_slot_candidates(feature_glob: str, slot_id: int, entry_cost_penalty: float) -> pd.DataFrame:
    """Load one slot's liq1 rows ranked by cost-aware entry score."""
    # Query only names inside the hold band because other names cannot affect this state machine.
    con = connect_duckdb(Path(":memory:"))
    df = con.execute(
        f"""
        WITH filtered AS (
            SELECT
                date,
                time,
                code,
                minute_slot,
                prediction,
                signal_amount,
                sigma_intraday,
                adv_amount,
                is_limit_up_all_day,
                is_limit_down_all_day,
                CAST(fillable_vwap AS BOOLEAN) AS fillable_base,
                CAST(ret_vwap_exec_10 AS DOUBLE) AS simple_return,
                CAST(entry_vwap_is_up_limit AS BOOLEAN) AS entry_is_up_limit,
                CAST(entry_vwap_is_down_limit AS BOOLEAN) AS entry_is_down_limit,
                CAST(exit_vwap_is_up_limit AS BOOLEAN) AS exit_is_up_limit,
                CAST(exit_vwap_is_down_limit AS BOOLEAN) AS exit_is_down_limit
            FROM read_parquet('{feature_glob}')
            WHERE
                minute_slot = {int(slot_id)}
                AND current_tradable = true
                AND prediction_available = true
                AND adv_amount IS NOT NULL
                AND adv_amount > 0
                AND sigma_intraday IS NOT NULL
                AND sigma_intraday > 0
        ),
        liquidity_tagged AS (
            SELECT
                *,
                ntile(3) OVER (PARTITION BY date, time ORDER BY signal_amount DESC) AS liq_bucket
            FROM filtered
        ),
        ranked AS (
            SELECT
                *,
                percent_rank() OVER (PARTITION BY date, time ORDER BY sigma_intraday / sqrt(adv_amount)) AS cost_pct,
                count(*) OVER (PARTITION BY date, time) AS name_count
            FROM liquidity_tagged
            WHERE liq_bucket <= {int(MAX_LIQ_BUCKET)}
        ),
        side_ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY date, time
                    ORDER BY prediction - {float(entry_cost_penalty):.8f} * cost_pct DESC
                ) AS rank_long,
                row_number() OVER (
                    PARTITION BY date, time
                    ORDER BY prediction + {float(entry_cost_penalty):.8f} * cost_pct ASC
                ) AS rank_short
            FROM ranked
        ),
        tagged AS (
            SELECT
                *,
                dense_rank() OVER (ORDER BY date, time) AS slot_bar_id,
                CAST(ceil({float(ENTRY_FRAC):.8f} * name_count) AS BIGINT) AS entry_count,
                CAST(ceil({float(HOLD_FRAC):.8f} * name_count) AS BIGINT) AS hold_count
            FROM side_ranked
        )
        SELECT
            date,
            time,
            code,
            minute_slot,
            slot_bar_id,
            prediction,
            simple_return,
            sigma_intraday,
            adv_amount,
            {float(SPREAD_BPS_HIGH):.6f} AS spread_bps,
            (
                fillable_base
                AND (NOT is_limit_up_all_day)
                AND (NOT entry_is_up_limit)
                AND (NOT is_limit_down_all_day)
                AND (NOT exit_is_down_limit)
            ) AS long_fillable,
            (
                fillable_base
                AND (NOT is_limit_down_all_day)
                AND (NOT entry_is_down_limit)
                AND (NOT is_limit_up_all_day)
                AND (NOT exit_is_up_limit)
            ) AS short_fillable,
            rank_long <= entry_count AS long_entry,
            rank_long <= hold_count AS long_hold,
            rank_short <= entry_count AS short_entry,
            rank_short <= hold_count AS short_hold
        FROM tagged
        WHERE
            rank_long <= hold_count
            OR rank_short <= hold_count
        ORDER BY slot_bar_id, code
        """
    ).fetchdf()
    con.close()
    return df


def _meta_from_row(row: Any, side: int) -> dict[str, float]:
    """Build cost, return, and sizing metadata for one selected row."""
    # Convert row fields into numeric values used by the simulator.
    fillable = bool(row.long_fillable) if int(side) > 0 else bool(row.short_fillable)
    simple_return = float(row.simple_return) if np.isfinite(float(row.simple_return)) else 0.0
    return {
        "prediction": float(row.prediction),
        "simple_return": float(simple_return),
        "spread_bps": float(row.spread_bps),
        "sigma_intraday": float(row.sigma_intraday),
        "adv_amount": float(row.adv_amount),
        "fillable": float(fillable),
    }


def _side_target_weights(next_desired: dict[int, int], rows_by_code: dict[int, Any], sizing_method: str) -> dict[int, float]:
    """Convert desired long/short names into side-normalized target weights."""
    # Split the desired book into long and short lists.
    long_codes = [code for code, side in next_desired.items() if int(side) > 0]
    short_codes = [code for code, side in next_desired.items() if int(side) < 0]
    target_weights: dict[int, float] = {}

    # Allocate each side with either equal weights or cost-aware weights.
    for side_codes, side_sign in [(long_codes, 1.0), (short_codes, -1.0)]:
        if len(side_codes) == 0:
            continue
        if str(sizing_method) == "equal":
            side_scores = {int(code): 1.0 for code in side_codes}
        else:
            side_scores = {
                int(code): float(np.sqrt(float(rows_by_code[int(code)].adv_amount)) / float(rows_by_code[int(code)].sigma_intraday))
                for code in side_codes
            }
        score_sum = float(sum(side_scores.values()))
        for code in side_codes:
            target_weights[int(code)] = float(side_sign * 0.5 * float(side_scores[int(code)]) / score_sum)
    return target_weights


def _apply_no_trade_band(target_weights: dict[int, float], previous_weights: dict[int, dict[str, float]], no_trade_band: float) -> dict[int, float]:
    """Suppress small target-weight changes by keeping prior weights."""
    # Keep prior weights when a retained name only needs a small rebalance.
    adjusted = dict(target_weights)
    for code in sorted(set(target_weights.keys()) & set(previous_weights.keys())):
        target_weight = float(target_weights.get(int(code), 0.0))
        previous_weight = float(previous_weights.get(int(code), {}).get("weight", 0.0))
        if abs(target_weight - previous_weight) < float(no_trade_band):
            adjusted[int(code)] = float(previous_weight)
    return adjusted


def _apply_turnover_budget(target_weights: dict[int, float], previous_weights: dict[int, dict[str, float]], turnover_budget: float) -> dict[int, float]:
    """Scale trades toward target weights when planned turnover exceeds the budget."""
    # Return the original target when this policy has no turnover budget.
    if float(turnover_budget) <= 0.0:
        return dict(target_weights)

    # Compute planned daily-turnover convention for this slot bar.
    union_codes = sorted(set(target_weights.keys()) | set(previous_weights.keys()))
    planned_turnover = 0.5 * sum(
        abs(float(target_weights.get(int(code), 0.0)) - float(previous_weights.get(int(code), {}).get("weight", 0.0)))
        for code in union_codes
    )
    if float(planned_turnover) <= float(turnover_budget):
        return dict(target_weights)

    # Interpolate from previous weights to target weights to respect the budget.
    scale = float(turnover_budget) / float(planned_turnover)
    adjusted: dict[int, float] = {}
    for code in union_codes:
        previous_weight = float(previous_weights.get(int(code), {}).get("weight", 0.0))
        target_weight = float(target_weights.get(int(code), 0.0))
        adjusted[int(code)] = float(previous_weight + scale * (target_weight - previous_weight))
    return adjusted


def _current_payload(
    code: int,
    side: int,
    rows_by_code: dict[int, Any],
    previous_weights: dict[int, dict[str, float]],
    target_weight: float,
) -> dict[str, float]:
    """Resolve one current holding's metadata from current rows or prior state."""
    # Prefer current-row metadata when the name is actively desired.
    row = rows_by_code.get(int(code))
    if row is not None and int(side) != 0:
        meta = _meta_from_row(row, int(side))
    else:
        meta = dict(previous_weights.get(int(code), {}))

    # Store the executable target weight with its metadata.
    meta["weight"] = float(target_weight)
    return meta


def simulate_policy_slot(slot_df: pd.DataFrame, variant: dict[str, object]) -> pd.DataFrame:
    """Simulate one slot for a sizing and trading-gate policy."""
    # Initialize desired holdings and previous executed weights for this slot.
    desired_state: dict[int, int] = {}
    previous_weights: dict[int, dict[str, float]] = {}
    bar_rows: list[dict[str, float]] = []
    minute_slot = int(slot_df["minute_slot"].iloc[0]) if int(slot_df.shape[0]) > 0 else -1

    # Walk bars in chronological order and update the buffer state machine.
    for key, bar_df in slot_df.groupby(["slot_bar_id", "date", "time"], sort=True):
        slot_bar_id, date, time_value = key
        rows_by_code = {int(row.code): row for row in bar_df.itertuples(index=False)}

        # Keep prior holdings only while they remain inside the hold band.
        next_desired: dict[int, int] = {}
        for code, side in desired_state.items():
            row = rows_by_code.get(int(code))
            if row is None:
                continue
            if int(side) > 0 and bool(row.long_hold):
                next_desired[int(code)] = 1
            if int(side) < 0 and bool(row.short_hold):
                next_desired[int(code)] = -1

        # Add fresh entries from the tighter entry band.
        for code, row in rows_by_code.items():
            if int(code) in next_desired:
                continue
            if bool(row.long_entry):
                next_desired[int(code)] = 1
                continue
            if bool(row.short_entry):
                next_desired[int(code)] = -1

        # Build raw target weights and apply trading gates.
        target_weights = _side_target_weights(next_desired, rows_by_code, str(variant["sizing_method"]))
        if float(variant["no_trade_band"]) > 0.0:
            target_weights = _apply_no_trade_band(target_weights, previous_weights, float(variant["no_trade_band"]))
        target_weights = _apply_turnover_budget(target_weights, previous_weights, float(variant["turnover_budget"]))

        # Apply fillability and compute return/cost accounting.
        current_weights: dict[int, dict[str, float]] = {}
        union_codes = sorted(set(previous_weights.keys()) | set(target_weights.keys()))
        turnover_abs = 0.0
        spread_cost = 0.0
        impact_coeff = 0.0
        gross_return = 0.0
        long_exposure = 0.0
        short_exposure = 0.0
        filled_count = 0.0
        for code in union_codes:
            side = int(next_desired.get(int(code), 0))
            target_weight = float(target_weights.get(int(code), 0.0))
            meta = _current_payload(int(code), int(side), rows_by_code, previous_weights, float(target_weight))
            executed_weight = float(target_weight) if float(meta.get("fillable", 0.0)) > 0.0 else 0.0
            previous_weight = float(previous_weights.get(int(code), {}).get("weight", 0.0))
            abs_delta = abs(executed_weight - previous_weight)
            turnover_abs += abs_delta
            if abs_delta > 0.0:
                spread_cost += 0.5 * float(meta["spread_bps"]) / 10000.0 * abs_delta
                impact_coeff += float(IMPACT_ETA) * float(meta["sigma_intraday"]) * abs_delta * np.sqrt(abs_delta / float(meta["adv_amount"]))
            gross_return += executed_weight * float(meta.get("simple_return", 0.0))
            if executed_weight > 0.0:
                long_exposure += executed_weight
                filled_count += 1.0
            if executed_weight < 0.0:
                short_exposure += -executed_weight
                filled_count += 1.0
            if executed_weight != 0.0:
                meta["weight"] = float(executed_weight)
                meta["fillable"] = 1.0
                current_weights[int(code)] = dict(meta)

        # Store one bar-level row and roll the state forward.
        desired_count = float(len(next_desired))
        bar_rows.append(
            {
                "minute_slot": float(minute_slot),
                "slot_bar_id": float(slot_bar_id),
                "date": float(date),
                "time": float(time_value),
                "gross_return": float(gross_return),
                "turnover": float(0.5 * turnover_abs),
                "spread_cost": float(spread_cost),
                "impact_coeff": float(impact_coeff),
                "fill_ratio": float(filled_count / desired_count) if desired_count > 0.0 else float("nan"),
                "long_exposure": float(long_exposure),
                "short_exposure": float(short_exposure),
                "desired_name_count": float(desired_count),
            }
        )
        desired_state = next_desired
        previous_weights = current_weights
    return pd.DataFrame(bar_rows)


def summarize_bar_frame(bar_df: pd.DataFrame, variant: dict[str, object]) -> dict[str, Any]:
    """Aggregate one policy's slot bars into daily performance metrics."""
    # Aggregate slot bars into slot-level daily returns first.
    slot_daily = (
        bar_df.groupby(["minute_slot", "date"], sort=True)
        .agg(
            gross_return=("gross_return", lambda values: float((1.0 + values.astype(float)).prod() - 1.0)),
            turnover=("turnover", "sum"),
            spread_cost=("spread_cost", "sum"),
            impact_coeff=("impact_coeff", "sum"),
            fill_ratio=("fill_ratio", "mean"),
            long_exposure=("long_exposure", "mean"),
            short_exposure=("short_exposure", "mean"),
            desired_name_count=("desired_name_count", "mean"),
        )
        .reset_index()
    )

    # Combine equal-capital slot returns into one daily strategy series.
    combined_daily = (
        slot_daily.groupby("date", sort=True)
        .agg(
            gross_return=("gross_return", "mean"),
            turnover=("turnover", "mean"),
            spread_cost=("spread_cost", "mean"),
            impact_coeff=("impact_coeff", "mean"),
            fill_ratio=("fill_ratio", "mean"),
            long_exposure=("long_exposure", "mean"),
            short_exposure=("short_exposure", "mean"),
            desired_name_count=("desired_name_count", "mean"),
        )
        .reset_index()
    )
    combined_daily["net_return_10m"] = (
        combined_daily["gross_return"].astype(float)
        - combined_daily["spread_cost"].astype(float)
        - combined_daily["impact_coeff"].astype(float) * (float(AUM_FOR_NET) ** 0.5)
    )

    # Persist diagnostics for this policy.
    output_dir = EXPERIMENT_ROOT / "variants" / str(variant["strategy_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    bar_df.to_csv(output_dir / "cost_aware_slot_bar.csv", index=False)
    combined_daily.to_csv(output_dir / "cost_aware_combined_daily.csv", index=False)

    # Build the nested summary payload.
    mean_coeff = float(combined_daily["impact_coeff"].astype(float).mean())
    payload = {
        "variant": dict(variant),
        "gross": summarize_return_series(combined_daily["gross_return"], ANNUAL_DAYS),
        "net_10m": summarize_return_series(combined_daily["net_return_10m"], ANNUAL_DAYS),
        "turnover": {
            "mean_daily_turnover": float(combined_daily["turnover"].astype(float).mean()),
            "p50_daily_turnover": float(combined_daily["turnover"].astype(float).quantile(0.50)),
            "p95_daily_turnover": float(combined_daily["turnover"].astype(float).quantile(0.95)),
        },
        "execution": {
            "mean_fill_ratio": float(combined_daily["fill_ratio"].astype(float).mean()),
            "mean_long_exposure": float(combined_daily["long_exposure"].astype(float).mean()),
            "mean_short_exposure": float(combined_daily["short_exposure"].astype(float).mean()),
            "mean_desired_name_count": float(combined_daily["desired_name_count"].astype(float).mean()),
        },
        "capacity": {
            "capacity_at_10bps": float(((10.0 / 10000.0) / mean_coeff) ** 2) if mean_coeff > 0.0 else float("nan"),
            "capacity_at_20bps": float(((20.0 / 10000.0) / mean_coeff) ** 2) if mean_coeff > 0.0 else float("nan"),
        },
        "method": "stateful_cost_aware_entry_python",
    }
    (output_dir / "cost_aware_strategy_summary.yaml").write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return payload


def flatten_summary(payload: dict[str, Any]) -> dict[str, object]:
    """Flatten one policy summary into a comparison row."""
    # Extract nested metrics into one CSV-friendly row.
    variant = dict(payload["variant"])
    return {
        "strategy_name": str(variant["strategy_name"]),
        "sizing_method": str(variant["sizing_method"]),
        "turnover_budget": float(variant["turnover_budget"]),
        "no_trade_band": float(variant["no_trade_band"]),
        "entry_cost_penalty": float(variant["entry_cost_penalty"]),
        "max_liq_bucket": int(variant["max_liq_bucket"]),
        "gross_daily_return": float(payload["gross"]["mean_daily_return"]),
        "gross_sharpe": float(payload["gross"]["annualized_sharpe"]),
        "daily_turnover": float(payload["turnover"]["mean_daily_turnover"]),
        "net_10m_daily_return": float(payload["net_10m"]["mean_daily_return"]),
        "net_10m_sharpe": float(payload["net_10m"]["annualized_sharpe"]),
        "net_10m_max_drawdown": float(payload["net_10m"]["max_drawdown"]),
        "capacity_10bps": float(payload["capacity"]["capacity_at_10bps"]),
        "capacity_20bps": float(payload["capacity"]["capacity_at_20bps"]),
        "mean_long_exposure": float(payload["execution"]["mean_long_exposure"]),
        "mean_short_exposure": float(payload["execution"]["mean_short_exposure"]),
        "mean_desired_name_count": float(payload["execution"]["mean_desired_name_count"]),
        "method": str(payload["method"]),
    }


def plot_policy_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot the main trade-offs across cost-aware policies."""
    # Draw gross, turnover, net, and capacity panels.
    df = summary_df.copy().reset_index(drop=True)
    x = list(range(int(df.shape[0])))
    labels = df["strategy_name"].astype(str).tolist()
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    axes[0].bar(x, df["gross_daily_return"].astype(float) * 1e4, color="#2E86AB")
    axes[0].set_title("Gross Return")
    axes[0].set_ylabel("Daily bps")
    axes[0].grid(alpha=0.25, axis="y")
    axes[1].bar(x, df["daily_turnover"].astype(float), color="#3D405B")
    axes[1].set_title("Turnover")
    axes[1].set_ylabel("Daily x")
    axes[1].grid(alpha=0.25, axis="y")
    axes[2].bar(x, df["net_10m_daily_return"].astype(float) * 1e4, color="#E07A5F")
    axes[2].set_title("Net Return at 10M AUM")
    axes[2].set_ylabel("Daily bps")
    axes[2].grid(alpha=0.25, axis="y")
    axes[3].bar(x, df["capacity_10bps"].astype(float) / 1_000_000.0, color="#81B29A")
    axes[3].set_title("Capacity at 10bps")
    axes[3].set_ylabel("M CNY")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=25, ha="right")
    axes[3].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_markdown_report(summary_df: pd.DataFrame) -> str:
    """Build the cost-aware entry-score experiment markdown report."""
    # Sort by net result and identify the best candidate.
    ordered = summary_df.sort_values(["net_10m_daily_return", "capacity_10bps"], ascending=[False, False]).reset_index(drop=True)
    best = ordered.iloc[0].to_dict()
    lines: list[str] = []
    lines.append("# Cost-Aware Entry Score 实验报告")
    lines.append("")
    lines.append("## 研究目标")
    lines.append("")
    lines.append("上一轮 cost-aware sizing 能改善 net 和 capacity, 但强 turnover budget 主要通过降仓转正。本实验固定 h60 / buffer / liq1 和接近满仓的 cost-aware sizing, 只在 entry/hold 排名中惩罚高成本名字, 检查是否能在不主动降仓的前提下改善 net result。")
    lines.append("")
    lines.append("## 实验定义")
    lines.append("")
    lines.append("- base candidate: `h60 + entry top/bottom 10% + hold top/bottom 20% + liq_bucket <= 1`.")
    lines.append("- cost-aware entry score: long ranking 使用 `prediction - penalty * cost_pct`, short ranking 使用 `prediction + penalty * cost_pct`。")
    lines.append("- cost_pct: 每个 date/time 内按 `sigma_intraday / sqrt(adv_amount)` 计算的 percentile rank。")
    lines.append("- cost-aware sizing: 单边 gross 仍为 `0.5`, 按 `sqrt(adv_amount) / sigma_intraday` 在每一侧分配权重。")
    lines.append("- 成本: spread cost + impact proxy at 10M AUM.")
    lines.append("- 方法: `stateful_cost_aware_entry_python`, 每个 slot 顺序维护 desired holdings 与 executed weights。")
    lines.append("")
    lines.append("## 结果摘要")
    lines.append("")
    lines.append(f"- 最佳方案: `{best['strategy_name']}`.")
    lines.append(f"- gross daily return: `{float(best['gross_daily_return']) * 1e4:.2f}` bps.")
    lines.append(f"- daily turnover: `{float(best['daily_turnover']):.4f}`.")
    lines.append(f"- net daily return at 10M AUM: `{float(best['net_10m_daily_return']) * 1e4:.2f}` bps.")
    lines.append(f"- capacity at 10bps: `{float(best['capacity_10bps']) / 1_000_000:.2f}`M.")
    lines.append("")
    lines.append("## 结果解读")
    lines.append("")
    lines.append("本实验不使用主动降仓的 turnover cap, 因此 raw net 的改善更能代表 portfolio construction 本身是否有效。")
    lines.append("")
    lines.append("如果 penalty 提高后 net 改善但 exposure 基本保持, 说明高成本名字确实在侵蚀信号；如果 gross 下降快于成本下降, 则说明成本惩罚过强, 把有效 alpha 也剔除了。")
    lines.append("")
    lines.append("这一步的重点不是让 raw net 勉强转正, 而是在接近满仓约束下找到 gross alpha 与执行成本的最优折中。")
    lines.append("")
    lines.append("## 完整对比表")
    lines.append("")
    cols = [
        "strategy_name",
        "sizing_method",
        "entry_cost_penalty",
        "turnover_budget",
        "no_trade_band",
        "gross_daily_return",
        "gross_sharpe",
        "daily_turnover",
        "net_10m_daily_return",
        "net_10m_sharpe",
        "capacity_10bps",
        "mean_long_exposure",
        "mean_short_exposure",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in ordered.loc[:, cols].itertuples(index=False):
        values: list[str] = []
        for value in list(row):
            if isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## 输出位置")
    lines.append("")
    lines.append(f"- experiment root: `{EXPERIMENT_ROOT.as_posix()}`.")
    lines.append("- summary CSV: `cost_aware_entry_experiment_summary.csv`.")
    lines.append("- report HTML: `cost_aware_entry_experiment_report.html`.")
    return "\n".join(lines)


def write_html_report(markdown_text: str, summary_df: pd.DataFrame, output_path: Path) -> None:
    """Write one compact HTML report."""
    # Render the summary table, comparison chart, and markdown appendix.
    headers = list(summary_df.columns)
    rows: list[list[str]] = []
    for row in summary_df.itertuples(index=False):
        row_values: list[str] = []
        for value in list(row):
            if isinstance(value, float):
                row_values.append(f"{value:.8g}")
            else:
                row_values.append(str(value))
        rows.append(row_values)
    plot_path = EXPERIMENT_ROOT / "cost_aware_entry_experiment_comparison.png"
    sections = [
        render_section("Summary Table", render_table(headers, rows)),
        render_section("Comparison Figure", f'<img src="{plot_path.as_posix()}" style="max-width:100%;height:auto;" />'),
        render_section("Markdown Appendix", render_code_block(markdown_text)),
    ]
    html = build_page("Cost-Aware Entry Score Experiments", "Cost-aware ranking penalty experiment report.", sections)
    Path(output_path).write_text(html, encoding="utf-8")


def run_cost_aware_experiments() -> Path:
    """Run all cost-aware entry-score policies and write unified outputs."""
    # Prepare directories, feature source, and policy containers.
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    REPO_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    feature_glob = load_chunk_manifest_glob(FEATURE_MANIFEST_PATH)
    variants = build_policy_variants()
    # Run each ranking policy independently because each penalty changes the selected hold band.
    rows: list[dict[str, object]] = []
    for variant in variants:
        strategy_name = str(variant["strategy_name"])
        slot_parts: list[pd.DataFrame] = []
        for slot_id in range(int(SLOT_MOD_BARS)):
            print(f"[cost-aware-entry] {strategy_name} slot {slot_id}/{int(SLOT_MOD_BARS)}", flush=True)
            slot_df = load_slot_candidates(feature_glob, int(slot_id), float(variant["entry_cost_penalty"]))
            slot_parts.append(simulate_policy_slot(slot_df, variant))
        bar_df = pd.concat(slot_parts, axis=0, ignore_index=True)
        payload = summarize_bar_frame(bar_df, variant)
        rows.append(flatten_summary(payload))

    # Persist unified outputs and synchronized repo reports.
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(EXPERIMENT_ROOT / "cost_aware_entry_experiment_summary.csv", index=False)
    (EXPERIMENT_ROOT / "cost_aware_entry_experiment_summary.yaml").write_text(yaml.safe_dump({"rows": rows}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    plot_policy_comparison(summary_df, EXPERIMENT_ROOT / "cost_aware_entry_experiment_comparison.png")
    markdown_text = build_markdown_report(summary_df)
    report_path = EXPERIMENT_ROOT / "cost_aware_entry_experiment_report.md"
    report_path.write_text(markdown_text, encoding="utf-8")
    repo_report_path = REPO_REPORT_DIR / "cost_aware_entry_experiment_report_0516.md"
    repo_report_path.write_text(markdown_text, encoding="utf-8")
    write_html_report(markdown_text, summary_df, EXPERIMENT_ROOT / "cost_aware_entry_experiment_report.html")
    write_html_report(markdown_text, summary_df, REPO_REPORT_DIR / "cost_aware_entry_experiment_report_0516.html")
    return report_path


def main() -> None:
    """Run the cost-aware experiment script."""
    # Execute the fixed policy matrix and print the report path.
    report_path = run_cost_aware_experiments()
    print(Path(report_path).as_posix())


if __name__ == "__main__":
    main()
