"""Run focused buffer and hysteresis experiments on the best holding-period candidate."""

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


EXPERIMENT_ROOT = Path("/data-cache/nn/trade_plan_experiments/0515_buffer_liquidity")
REPO_REPORT_DIR = REPO_ROOT / "report" / "0515"
FEATURE_MANIFEST_PATH = Path("/data-cache/nn/trade_plan_experiments/0515/features/entry1_h60_slot60/feature_manifest.yaml")
AUM_FOR_NET = 10_000_000.0
ANNUAL_DAYS = 252
ENTRY_FRAC = 0.10
HOLD_FRACS = [0.125, 0.15, 0.20]
SLOT_MOD_BARS = 60
IMPACT_ETA = 0.50
SPREAD_BPS_HIGH = 5.0
SPREAD_BPS_MID = 10.0
SPREAD_BPS_LOW = 20.0


def build_buffer_variants() -> list[dict[str, object]]:
    """Define the focused buffer experiment variants."""
    # Compare liquidity filters on the best h60 entry10/hold20 buffer candidate.
    variants: list[dict[str, object]] = []
    for max_liq_bucket in [3, 2, 1]:
        variants.append(
            {
                "strategy_name": f"buffer_h60_entry10_hold200_ls_liq{int(max_liq_bucket)}",
                "experiment_group": "buffer_liquidity",
                "entry_frac": float(ENTRY_FRAC),
                "hold_frac": 0.20,
                "holding_bars": 60,
                "slot_mod_bars": int(SLOT_MOD_BARS),
                "max_liq_bucket": int(max_liq_bucket),
            }
        )
    return variants


def load_slot_candidates(feature_glob: str, slot_id: int, entry_frac: float, hold_frac: float, max_liq_bucket: int) -> pd.DataFrame:
    """Load one slot's entry/hold candidate rows for stateful buffer simulation."""
    # Query only rows that can affect the state machine to keep memory bounded.
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
                row_number() OVER (PARTITION BY date, time ORDER BY prediction) AS rank_asc,
                count(*) OVER (PARTITION BY date, time) AS name_count
            FROM liquidity_tagged
            WHERE liq_bucket <= {int(max_liq_bucket)}
        ),
        tagged AS (
            SELECT
                *,
                dense_rank() OVER (ORDER BY date, time) AS slot_bar_id,
                CAST(ceil({float(entry_frac):.8f} * name_count) AS BIGINT) AS entry_count,
                CAST(ceil({float(hold_frac):.8f} * name_count) AS BIGINT) AS hold_count
            FROM ranked
        )
        SELECT
            date,
            time,
            code,
            minute_slot,
            slot_bar_id,
            simple_return,
            sigma_intraday,
            adv_amount,
            CASE
                WHEN liq_bucket = 1 THEN {float(SPREAD_BPS_HIGH):.6f}
                WHEN liq_bucket = 2 THEN {float(SPREAD_BPS_MID):.6f}
                ELSE {float(SPREAD_BPS_LOW):.6f}
            END AS spread_bps,
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
            rank_asc > name_count - entry_count AS long_entry,
            rank_asc > name_count - hold_count AS long_hold,
            rank_asc <= entry_count AS short_entry,
            rank_asc <= hold_count AS short_hold
        FROM tagged
        WHERE
            rank_asc > name_count - hold_count
            OR rank_asc <= hold_count
        ORDER BY slot_bar_id, code
        """
    ).fetchdf()
    con.close()
    return df


def _meta_from_row(row: Any, side: int) -> dict[str, float]:
    """Build cost and return metadata for one selected row."""
    # Store the fields needed for return and cost accounting.
    fillable = bool(row.long_fillable) if int(side) > 0 else bool(row.short_fillable)
    return {
        "simple_return": float(row.simple_return) if np.isfinite(float(row.simple_return)) else 0.0,
        "spread_bps": float(row.spread_bps),
        "sigma_intraday": float(row.sigma_intraday),
        "adv_amount": float(row.adv_amount),
        "fillable": float(fillable),
    }


def simulate_buffer_slot(slot_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate one minute-slot state machine with entry and hold thresholds."""
    # Initialize desired holdings and previous executed weights for this slot.
    desired_state: dict[int, int] = {}
    previous_weights: dict[int, dict[str, float]] = {}
    bar_rows: list[dict[str, float]] = []
    minute_slot = int(slot_df["minute_slot"].iloc[0]) if int(slot_df.shape[0]) > 0 else -1

    # Walk bars in chronological order and update desired holdings.
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

        # Convert desired holdings into side-normalized target weights.
        long_codes = [code for code, side in next_desired.items() if int(side) > 0]
        short_codes = [code for code, side in next_desired.items() if int(side) < 0]
        target_weights: dict[int, float] = {}
        for code in list(long_codes):
            target_weights[int(code)] = 0.5 / float(len(long_codes)) if len(long_codes) > 0 else 0.0
        for code in list(short_codes):
            target_weights[int(code)] = -0.5 / float(len(short_codes)) if len(short_codes) > 0 else 0.0

        # Apply fillability and compute turnover/cost against previous executed weights.
        current_weights: dict[int, dict[str, float]] = {}
        union_codes = sorted(set(previous_weights.keys()) | set(target_weights.keys()))
        turnover_abs = 0.0
        spread_cost = 0.0
        impact_coeff = 0.0
        gross_return = 0.0
        long_exposure = 0.0
        short_exposure = 0.0
        filled_count = 0.0
        for code in list(union_codes):
            side = int(next_desired.get(int(code), 0))
            row = rows_by_code.get(int(code))
            meta = _meta_from_row(row, side) if row is not None and side != 0 else previous_weights.get(int(code), {})
            target_weight = float(target_weights.get(int(code), 0.0))
            executed_weight = target_weight if float(meta.get("fillable", 0.0)) > 0.0 else 0.0
            previous_weight = float(previous_weights.get(int(code), {}).get("weight", 0.0))
            abs_delta = abs(executed_weight - previous_weight)
            turnover_abs += abs_delta
            if abs_delta > 0.0 and len(meta) > 0:
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
                current_weights[int(code)] = {
                    "weight": float(executed_weight),
                    "simple_return": float(meta.get("simple_return", 0.0)),
                    "spread_bps": float(meta["spread_bps"]),
                    "sigma_intraday": float(meta["sigma_intraday"]),
                    "adv_amount": float(meta["adv_amount"]),
                    "fillable": 1.0,
                }

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


def summarize_buffer_variant(feature_glob: str, variant: dict[str, object]) -> dict[str, Any]:
    """Run one buffer variant and persist its diagnostics."""
    # Prepare variant output paths.
    output_dir = EXPERIMENT_ROOT / "variants" / str(variant["strategy_name"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulate each slot separately to bound memory.
    slot_parts: list[pd.DataFrame] = []
    for slot_id in range(int(SLOT_MOD_BARS)):
        print(f"[buffer] {variant['strategy_name']} slot {slot_id}/{int(SLOT_MOD_BARS)}", flush=True)
        slot_df = load_slot_candidates(
            feature_glob,
            int(slot_id),
            float(variant["entry_frac"]),
            float(variant["hold_frac"]),
            int(variant["max_liq_bucket"]),
        )
        slot_parts.append(simulate_buffer_slot(slot_df))

    # Stack slot bars and aggregate to daily performance.
    bar_df = pd.concat(slot_parts, axis=0, ignore_index=True)
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

    # Persist detailed outputs and compact summary.
    bar_df.to_csv(output_dir / "buffer_vwap_slot_bar.csv", index=False)
    combined_daily.to_csv(output_dir / "buffer_vwap_combined_daily.csv", index=False)
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
        "method": "stateful_buffer_python",
    }
    (output_dir / "buffer_strategy_summary.yaml").write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return payload


def flatten_summary(payload: dict[str, Any]) -> dict[str, object]:
    """Flatten one buffer summary into a comparison row."""
    # Extract nested metrics into one CSV-friendly row.
    variant = dict(payload["variant"])
    return {
        "strategy_name": str(variant["strategy_name"]),
        "entry_frac": float(variant["entry_frac"]),
        "hold_frac": float(variant["hold_frac"]),
        "max_liq_bucket": int(variant["max_liq_bucket"]),
        "gross_daily_return": float(payload["gross"]["mean_daily_return"]),
        "gross_sharpe": float(payload["gross"]["annualized_sharpe"]),
        "daily_turnover": float(payload["turnover"]["mean_daily_turnover"]),
        "net_10m_daily_return": float(payload["net_10m"]["mean_daily_return"]),
        "net_10m_sharpe": float(payload["net_10m"]["annualized_sharpe"]),
        "net_10m_max_drawdown": float(payload["net_10m"]["max_drawdown"]),
        "capacity_10bps": float(payload["capacity"]["capacity_at_10bps"]),
        "capacity_20bps": float(payload["capacity"]["capacity_at_20bps"]),
        "mean_desired_name_count": float(payload["execution"]["mean_desired_name_count"]),
        "method": str(payload["method"]),
    }


def plot_buffer_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot the main buffer trade-offs."""
    # Draw gross, turnover, net, and capacity panels.
    df = summary_df.copy().reset_index(drop=True)
    x = list(range(int(df.shape[0])))
    labels = df["strategy_name"].astype(str).tolist()
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
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
    """Build the buffer experiment markdown report."""
    # Sort by net result and identify the best candidate.
    ordered = summary_df.sort_values(["net_10m_daily_return", "capacity_10bps"], ascending=[False, False]).reset_index(drop=True)
    best = ordered.iloc[0].to_dict()
    lines: list[str] = []
    lines.append("# Buffer + Liquidity 实验报告")
    lines.append("")
    lines.append("## 研究目标")
    lines.append("")
    lines.append("上一轮 buffer 实验显示 `h60 + entry10 + hold20 + liq3` 最优, 但 10M AUM 下 net 仍为负。本实验固定 h60 / entry top-bottom 10% / hold top-bottom 20%, 继续叠加 liquidity filter, 检查 capacity 和 net 是否继续改善。")
    lines.append("")
    lines.append("## 实验定义")
    lines.append("")
    lines.append("- entry threshold: top/bottom `10%`.")
    lines.append("- hold threshold: top/bottom `20%`.")
    lines.append("- liquidity filters: `liq_bucket <= 3`, `liq_bucket <= 2`, `liq_bucket <= 1`.")
    lines.append("- holding bars: `60`.")
    lines.append("- slot definition: `base_minute % 60`.")
    lines.append("- long/short gross: 两端各 `0.5`.")
    lines.append("- 成本: spread cost + impact proxy at 10M AUM.")
    lines.append("- 方法: `stateful_buffer_python`, 按 slot 顺序维护 desired holdings.")
    lines.append("")
    lines.append("## 结果摘要")
    lines.append("")
    lines.append(f"- 最佳 buffer + liquidity 方案: `{best['strategy_name']}`.")
    lines.append(f"- gross daily return: `{float(best['gross_daily_return']) * 1e4:.2f}` bps.")
    lines.append(f"- daily turnover: `{float(best['daily_turnover']):.4f}`.")
    lines.append(f"- net daily return at 10M AUM: `{float(best['net_10m_daily_return']) * 1e4:.2f}` bps.")
    lines.append(f"- capacity at 10bps: `{float(best['capacity_10bps']) / 1_000_000:.2f}`M.")
    lines.append("")
    lines.append("## 完整对比表")
    lines.append("")
    cols = [
        "strategy_name",
        "entry_frac",
        "hold_frac",
        "max_liq_bucket",
        "gross_daily_return",
        "gross_sharpe",
        "daily_turnover",
        "net_10m_daily_return",
        "net_10m_sharpe",
        "capacity_10bps",
        "mean_desired_name_count",
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
    lines.append("- summary CSV: `buffer_experiment_summary.csv`.")
    lines.append("- report HTML: `buffer_experiment_report.html`.")
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
    plot_path = EXPERIMENT_ROOT / "buffer_experiment_comparison.png"
    sections = [
        render_section("Summary Table", render_table(headers, rows)),
        render_section("Comparison Figure", f'<img src="{plot_path.as_posix()}" style="max-width:100%;height:auto;" />'),
        render_section("Markdown Appendix", render_code_block(markdown_text)),
    ]
    html = build_page("Buffer Experiments", "Stateful buffer and hysteresis experiment report.", sections)
    Path(output_path).write_text(html, encoding="utf-8")


def run_buffer_experiments() -> Path:
    """Run all focused buffer experiments and write unified outputs."""
    # Prepare directories and feature source.
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    REPO_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    feature_glob = load_chunk_manifest_glob(FEATURE_MANIFEST_PATH)
    variants = build_buffer_variants()

    # Run each buffer variant and collect summaries.
    rows: list[dict[str, object]] = []
    for variant in list(variants):
        payload = summarize_buffer_variant(feature_glob, variant)
        rows.append(flatten_summary(payload))

    # Persist unified outputs.
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(EXPERIMENT_ROOT / "buffer_experiment_summary.csv", index=False)
    (EXPERIMENT_ROOT / "buffer_experiment_summary.yaml").write_text(yaml.safe_dump({"rows": rows}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    plot_buffer_comparison(summary_df, EXPERIMENT_ROOT / "buffer_experiment_comparison.png")
    markdown_text = build_markdown_report(summary_df)
    report_path = EXPERIMENT_ROOT / "buffer_experiment_report.md"
    report_path.write_text(markdown_text, encoding="utf-8")
    repo_report_path = REPO_REPORT_DIR / "buffer_experiment_report_0515.md"
    repo_report_path.write_text(markdown_text, encoding="utf-8")
    write_html_report(markdown_text, summary_df, EXPERIMENT_ROOT / "buffer_experiment_report.html")
    write_html_report(markdown_text, summary_df, REPO_REPORT_DIR / "buffer_experiment_report_0515.html")
    return report_path


def main() -> None:
    """Run the focused buffer experiment script."""
    # Execute the fixed buffer matrix and print the report path.
    report_path = run_buffer_experiments()
    print(Path(report_path).as_posix())


if __name__ == "__main__":
    main()
