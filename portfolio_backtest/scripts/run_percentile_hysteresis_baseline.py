"""Run a 10min long-only percentile hysteresis baseline."""

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
from prediction_nn2.html_report import build_page, render_code_block, render_embedded_figure, render_section, render_table


EXPERIMENT_ROOT = Path("/data-cache/nn/trade_plan_experiments/0516_percentile_hysteresis_baseline")
REPO_REPORT_DIR = REPO_ROOT / "report" / "0516"
FEATURE_MANIFEST_PATH = Path("/data-cache/nn/trade_plan_experiments/0515/features/entry1_h60_slot60/feature_manifest.yaml")
ANNUAL_DAYS = 252
MIN_TIMESTAMP_NAME_COUNT = 1000
FEE_BPS = 1.0
SPREAD_BPS_HIGH = 5.0
SPREAD_BPS_MID = 10.0
SPREAD_BPS_LOW = 20.0


def build_variants() -> list[dict[str, object]]:
    """Define the 10min percentile hysteresis baseline matrix."""
    # Keep the matrix small and interpretable for the first clean baseline.
    return [
        {"strategy_name": "q90_q80", "open_quantile": 0.90, "close_quantile": 0.80},
        {"strategy_name": "q90_q70", "open_quantile": 0.90, "close_quantile": 0.70},
        {"strategy_name": "q95_q85", "open_quantile": 0.95, "close_quantile": 0.85},
        {"strategy_name": "q95_q80", "open_quantile": 0.95, "close_quantile": 0.80},
        {"strategy_name": "q85_q75", "open_quantile": 0.85, "close_quantile": 0.75},
    ]


def load_available_dates(con: Any, feature_glob: str) -> list[int]:
    """Load dates with available prediction and 10min return data."""
    # Only require fields needed to rank signals and compute gross return.
    df = con.execute(
        f"""
        SELECT DISTINCT CAST(date AS INTEGER) AS date
        FROM read_parquet('{feature_glob}')
        WHERE prediction IS NOT NULL
          AND ret_vwap_exec_10 IS NOT NULL
          AND signal_amount IS NOT NULL
          AND minute_slot % 10 = 0
        ORDER BY date
        """
    ).fetchdf()
    return [int(value) for value in df["date"].tolist()]


def load_day_candidates(con: Any, feature_glob: str, date: int, close_quantile: float) -> pd.DataFrame:
    """Load one day's rows that can affect the hysteresis state."""
    # Rank the full timestamp universe, then keep only rows inside the close band.
    df = con.execute(
        f"""
        WITH base AS (
            SELECT
                CAST(date AS INTEGER) AS date,
                CAST(time AS INTEGER) AS time,
                CAST(code AS INTEGER) AS code,
                CAST(minute_slot AS INTEGER) AS minute_slot,
                CAST(prediction AS DOUBLE) AS prediction,
                CAST(ret_vwap_exec_10 AS DOUBLE) AS simple_return,
                CAST(signal_amount AS DOUBLE) AS signal_amount
            FROM read_parquet('{feature_glob}')
            WHERE date = {int(date)}
              AND prediction IS NOT NULL
              AND ret_vwap_exec_10 IS NOT NULL
              AND signal_amount IS NOT NULL
              AND minute_slot % 10 = 0
        ),
        complete_times AS (
            SELECT
                date,
                time
            FROM base
            GROUP BY date, time
            HAVING count(*) >= {int(MIN_TIMESTAMP_NAME_COUNT)}
        ),
        ranked AS (
            SELECT
                base.*,
                cume_dist() OVER (PARTITION BY base.date, base.time ORDER BY base.prediction) AS signal_pct,
                ntile(3) OVER (PARTITION BY base.date, base.time ORDER BY base.signal_amount DESC) AS liq_bucket
            FROM base
            INNER JOIN complete_times
                ON base.date = complete_times.date
               AND base.time = complete_times.time
        )
        SELECT
            date,
            time,
            code,
            minute_slot,
            prediction,
            simple_return,
            CASE
                WHEN liq_bucket = 1 THEN {float(SPREAD_BPS_HIGH):.6f}
                WHEN liq_bucket = 2 THEN {float(SPREAD_BPS_MID):.6f}
                ELSE {float(SPREAD_BPS_LOW):.6f}
            END AS spread_bps,
            signal_pct
        FROM ranked
        WHERE signal_pct >= {float(close_quantile):.8f}
        ORDER BY date, time, code
        """
    ).fetchdf()
    return df


def simulate_day(day_df: pd.DataFrame, variant: dict[str, object]) -> pd.DataFrame:
    """Simulate one day of long-only percentile hysteresis."""
    # Initialize daily holdings; no overnight state is carried into the baseline.
    active_state: set[int] = set()
    previous_state: dict[int, dict[str, float]] = {}
    bar_rows: list[dict[str, float]] = []
    open_quantile = float(variant["open_quantile"])
    close_quantile = float(variant["close_quantile"])

    # Walk timestamps and update the active long book.
    for key, bar_df in day_df.groupby(["date", "time"], sort=True):
        date, time_value = key
        rows_by_code = {int(row.code): row for row in bar_df.itertuples(index=False)}

        # Retain existing longs while the signal stays above the close threshold.
        next_active: set[int] = set()
        for code in active_state:
            row = rows_by_code.get(int(code))
            if row is not None and float(row.signal_pct) >= close_quantile:
                next_active.add(int(code))

        # Open new longs whose signal is above the open threshold.
        for code, row in rows_by_code.items():
            if int(code) in next_active:
                continue
            if float(row.signal_pct) >= open_quantile:
                next_active.add(int(code))

        # Convert the active book into equal-weight long exposure.
        target_weights: dict[int, float] = {}
        if len(next_active) > 0:
            weight = 1.0 / float(len(next_active))
            for code in next_active:
                target_weights[int(code)] = float(weight)

        # Compute gross return and turnover against the previous executed book.
        gross_return = 0.0
        active_name_count = float(len(next_active))
        for code, target_weight in target_weights.items():
            row = rows_by_code[int(code)]
            gross_return += float(target_weight) * float(row.simple_return)

        # Charge spread and fee costs on absolute weight changes.
        turnover_abs = 0.0
        spread_cost = 0.0
        fee_cost = 0.0
        current_state: dict[int, dict[str, float]] = {}
        for code in sorted(set(target_weights.keys()) | set(previous_state.keys())):
            target_weight = float(target_weights.get(int(code), 0.0))
            previous_weight = float(previous_state.get(int(code), {}).get("weight", 0.0))
            abs_delta = abs(target_weight - previous_weight)
            turnover_abs += abs_delta
            if int(code) in rows_by_code:
                row = rows_by_code[int(code)]
                meta = {
                    "weight": float(target_weight),
                    "spread_bps": float(row.spread_bps),
                }
            else:
                meta = dict(previous_state[int(code)])
                meta["weight"] = float(target_weight)
            if abs_delta > 0.0:
                spread_cost += 0.5 * float(meta["spread_bps"]) / 10000.0 * abs_delta
                fee_cost += float(FEE_BPS) / 10000.0 * abs_delta
            if target_weight != 0.0:
                current_state[int(code)] = dict(meta)
        turnover = 0.5 * float(turnover_abs)

        # Store one 10min bar row and advance state.
        bar_rows.append(
            {
                "date": float(date),
                "time": float(time_value),
                "gross_return": float(gross_return),
                "turnover": float(turnover),
                "spread_cost": float(spread_cost),
                "fee_cost": float(fee_cost),
                "gross_exposure": float(sum(abs(value) for value in target_weights.values())),
                "active_name_count": float(active_name_count),
            }
        )
        active_state = next_active
        previous_state = current_state
    return pd.DataFrame(bar_rows)


def summarize_variant(con: Any, feature_glob: str, dates: list[int], variant: dict[str, object]) -> dict[str, Any]:
    """Run one baseline variant and persist diagnostics."""
    # Prepare the output directory for this variant.
    output_dir = EXPERIMENT_ROOT / "variants" / str(variant["strategy_name"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulate each date independently to avoid overnight assumptions and bound memory.
    day_parts: list[pd.DataFrame] = []
    for date in list(dates):
        print(f"[percentile-baseline] {variant['strategy_name']} date {date}", flush=True)
        day_df = load_day_candidates(con, feature_glob, int(date), float(variant["close_quantile"]))
        if int(day_df.shape[0]) == 0:
            continue
        day_parts.append(simulate_day(day_df, variant))

    # Aggregate 10min bars into daily gross performance and diagnostics.
    bar_df = pd.concat(day_parts, axis=0, ignore_index=True)
    daily = (
        bar_df.groupby("date", sort=True)
        .agg(
            gross_return=("gross_return", lambda values: float((1.0 + values.astype(float)).prod() - 1.0)),
            turnover=("turnover", "sum"),
            spread_cost=("spread_cost", "sum"),
            fee_cost=("fee_cost", "sum"),
            gross_exposure=("gross_exposure", "mean"),
            active_name_count=("active_name_count", "mean"),
        )
        .reset_index()
    )
    daily["net_return"] = (
        daily["gross_return"].astype(float)
        - daily["spread_cost"].astype(float)
        - daily["fee_cost"].astype(float)
    )

    # Persist detailed outputs and build the summary payload.
    bar_df.to_csv(output_dir / "percentile_hysteresis_10min_bar.csv", index=False)
    daily.to_csv(output_dir / "percentile_hysteresis_daily.csv", index=False)
    payload = {
        "variant": dict(variant),
        "gross": summarize_return_series(daily["gross_return"], ANNUAL_DAYS),
        "net": summarize_return_series(daily["net_return"], ANNUAL_DAYS),
        "turnover": {
            "mean_daily_turnover": float(daily["turnover"].astype(float).mean()),
            "p50_daily_turnover": float(daily["turnover"].astype(float).quantile(0.50)),
            "p95_daily_turnover": float(daily["turnover"].astype(float).quantile(0.95)),
        },
        "cost": {
            "mean_daily_spread_cost": float(daily["spread_cost"].astype(float).mean()),
            "mean_daily_fee_cost": float(daily["fee_cost"].astype(float).mean()),
            "fee_bps": float(FEE_BPS),
        },
        "execution": {
            "mean_gross_exposure": float(daily["gross_exposure"].astype(float).mean()),
            "mean_active_name_count": float(daily["active_name_count"].astype(float).mean()),
        },
        "method": "long_only_10min_percentile_hysteresis",
    }
    (output_dir / "percentile_hysteresis_summary.yaml").write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return payload


def flatten_summary(payload: dict[str, Any]) -> dict[str, object]:
    """Flatten one nested summary payload into a CSV row."""
    # Extract the most important comparison fields.
    variant = dict(payload["variant"])
    return {
        "strategy_name": str(variant["strategy_name"]),
        "open_quantile": float(variant["open_quantile"]),
        "close_quantile": float(variant["close_quantile"]),
        "gross_daily_return": float(payload["gross"]["mean_daily_return"]),
        "gross_sharpe": float(payload["gross"]["annualized_sharpe"]),
        "gross_max_drawdown": float(payload["gross"]["max_drawdown"]),
        "net_daily_return": float(payload["net"]["mean_daily_return"]),
        "net_sharpe": float(payload["net"]["annualized_sharpe"]),
        "daily_turnover": float(payload["turnover"]["mean_daily_turnover"]),
        "p95_daily_turnover": float(payload["turnover"]["p95_daily_turnover"]),
        "daily_spread_cost": float(payload["cost"]["mean_daily_spread_cost"]),
        "daily_fee_cost": float(payload["cost"]["mean_daily_fee_cost"]),
        "daily_total_cost": float(payload["cost"]["mean_daily_spread_cost"]) + float(payload["cost"]["mean_daily_fee_cost"]),
        "mean_gross_exposure": float(payload["execution"]["mean_gross_exposure"]),
        "mean_active_name_count": float(payload["execution"]["mean_active_name_count"]),
        "method": str(payload["method"]),
    }


def plot_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot return, turnover, cost, and active-name diagnostics."""
    # Draw a compact five-panel comparison figure.
    df = summary_df.copy().reset_index(drop=True)
    x = list(range(int(df.shape[0])))
    labels = df["strategy_name"].astype(str).tolist()
    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True)
    axes[0].bar(x, df["gross_daily_return"].astype(float) * 1e4, color="#2E86AB")
    axes[0].set_title("Gross Return")
    axes[0].set_ylabel("Daily bps")
    axes[0].grid(alpha=0.25, axis="y")
    axes[1].bar(x, df["net_daily_return"].astype(float) * 1e4, color="#E07A5F")
    axes[1].set_title("Net Return")
    axes[1].set_ylabel("Daily bps")
    axes[1].grid(alpha=0.25, axis="y")
    axes[2].bar(x, df["daily_turnover"].astype(float), color="#3D405B")
    axes[2].set_title("Daily Turnover")
    axes[2].set_ylabel("Daily x")
    axes[2].grid(alpha=0.25, axis="y")
    axes[3].bar(x, df["daily_spread_cost"].astype(float) * 1e4, color="#81B29A", label="spread")
    axes[3].bar(x, df["daily_fee_cost"].astype(float) * 1e4, bottom=df["daily_spread_cost"].astype(float) * 1e4, color="#F2CC8F", label="fee")
    axes[3].set_title("Daily Cost")
    axes[3].set_ylabel("Daily bps")
    axes[3].legend()
    axes[3].grid(alpha=0.25, axis="y")
    axes[4].bar(x, df["mean_active_name_count"].astype(float), color="#E07A5F")
    axes[4].set_title("Mean Active Names")
    axes[4].set_ylabel("Names")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(labels, rotation=25, ha="right")
    axes[4].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_backtest_curves(variants: list[dict[str, object]]) -> dict[str, Path]:
    """Plot gross/net wealth and drawdown curves from daily outputs."""
    # Load daily outputs produced for each variant.
    series: dict[str, pd.DataFrame] = {}
    for variant in list(variants):
        strategy_name = str(variant["strategy_name"])
        daily_path = EXPERIMENT_ROOT / "variants" / strategy_name / "percentile_hysteresis_daily.csv"
        df = pd.read_csv(daily_path).sort_values("date")
        df["date_ts"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%y%m%d")
        df["gross_wealth"] = (1.0 + df["gross_return"].astype(float)).cumprod()
        df["net_wealth"] = (1.0 + df["net_return"].astype(float)).cumprod()
        df["gross_drawdown"] = df["gross_wealth"] / df["gross_wealth"].cummax() - 1.0
        df["net_drawdown"] = df["net_wealth"] / df["net_wealth"].cummax() - 1.0
        series[strategy_name] = df

    # Draw gross wealth.
    gross_wealth_path = EXPERIMENT_ROOT / "percentile_hysteresis_baseline_gross_wealth_curve.png"
    plt.figure(figsize=(13, 7))
    for strategy_name, df in series.items():
        plt.plot(df["date_ts"], df["gross_wealth"], label=strategy_name, linewidth=1.8)
    plt.title("10min Percentile Hysteresis Baseline - Gross Wealth")
    plt.ylabel("Cumulative gross wealth")
    plt.xlabel("Date")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(gross_wealth_path, dpi=160)
    plt.close()

    # Draw net wealth.
    net_wealth_path = EXPERIMENT_ROOT / "percentile_hysteresis_baseline_net_wealth_curve.png"
    plt.figure(figsize=(13, 7))
    for strategy_name, df in series.items():
        plt.plot(df["date_ts"], df["net_wealth"], label=strategy_name, linewidth=1.8)
    plt.title("10min Percentile Hysteresis Baseline - Net Wealth")
    plt.ylabel("Cumulative net wealth")
    plt.xlabel("Date")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(net_wealth_path, dpi=160)
    plt.close()

    # Draw gross drawdown.
    gross_drawdown_path = EXPERIMENT_ROOT / "percentile_hysteresis_baseline_gross_drawdown_curve.png"
    plt.figure(figsize=(13, 7))
    for strategy_name, df in series.items():
        plt.plot(df["date_ts"], df["gross_drawdown"] * 100.0, label=strategy_name, linewidth=1.8)
    plt.title("10min Percentile Hysteresis Baseline - Gross Drawdown")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(gross_drawdown_path, dpi=160)
    plt.close()

    # Draw net drawdown.
    net_drawdown_path = EXPERIMENT_ROOT / "percentile_hysteresis_baseline_net_drawdown_curve.png"
    plt.figure(figsize=(13, 7))
    for strategy_name, df in series.items():
        plt.plot(df["date_ts"], df["net_drawdown"] * 100.0, label=strategy_name, linewidth=1.8)
    plt.title("10min Percentile Hysteresis Baseline - Net Drawdown")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(net_drawdown_path, dpi=160)
    plt.close()

    return {
        "gross_wealth": gross_wealth_path,
        "net_wealth": net_wealth_path,
        "gross_drawdown": gross_drawdown_path,
        "net_drawdown": net_drawdown_path,
    }


def build_markdown_report(summary_df: pd.DataFrame) -> str:
    """Build the baseline markdown research report."""
    # Sort variants by net return after adding spread and fee costs.
    ordered = summary_df.sort_values(["net_daily_return", "gross_daily_return"], ascending=[False, False]).reset_index(drop=True)
    best = ordered.iloc[0].to_dict()
    lines: list[str] = []
    lines.append("# 10min Long-Only Percentile Hysteresis Baseline 研究报告")
    lines.append("")
    lines.append("## 研究目标")
    lines.append("")
    lines.append("本实验保留最干净的 signal-to-position baseline: 不加 liquidity / tradable / cost-aware 过滤, 不做 short, 不做 h60 slot, 只用 10min prediction 横截面分位数决定 long-only 开仓和平仓, 并在此基础上加入 spread + fee 成本模型。")
    lines.append("")
    lines.append("## 交易规则")
    lines.append("")
    lines.append("- universe: 仅要求 `prediction`、`ret_vwap_exec_10` 与成本计算字段可用。")
    lines.append(f"- data completeness: 每个 10min 决策截面至少 `{int(MIN_TIMESTAMP_NAME_COUNT)}` 只股票可用, 避免尾盘 forward return 不完整导致组合集中到极少数名字。")
    lines.append("- signal: 每个 `date/time` 内对 `prediction` 计算 `signal_pct`, 越高越看多。")
    lines.append("- open: 空仓股票当 `signal_pct >= x` 时开 long。")
    lines.append("- close: 已持仓股票当 `signal_pct < y` 时平仓。")
    lines.append("- hysteresis: 要求 `x > y`, 用 close band 降低频繁进出。")
    lines.append("- holding: 每 10min 更新一次, 仅使用 `minute_slot % 10 = 0` 的决策点。")
    lines.append("- overnight: 每日状态重置, 不携带隔夜持仓。")
    lines.append("- sizing: active long names 等权, gross exposure 目标为 `1.0`。")
    lines.append("- return: 使用 `ret_vwap_exec_10` 计算 gross return。")
    lines.append(f"- cost: 使用 spread cost + fee cost, `net = gross - spread - fee`, fee 按 traded notional `{float(FEE_BPS):.2f}` bps 计算。")
    lines.append("- spread: 不筛 liquidity, 只按全截面 `signal_amount` 三等分估计 spread bps。")
    lines.append("")
    lines.append("## 结果摘要")
    lines.append("")
    lines.append(f"- net 最佳方案: `{best['strategy_name']}`.")
    lines.append(f"- gross daily return: `{float(best['gross_daily_return']) * 1e4:.2f}` bps.")
    lines.append(f"- net daily return: `{float(best['net_daily_return']) * 1e4:.2f}` bps.")
    lines.append(f"- gross Sharpe: `{float(best['gross_sharpe']):.2f}`.")
    lines.append(f"- daily turnover: `{float(best['daily_turnover']):.4f}`.")
    lines.append(f"- daily spread cost: `{float(best['daily_spread_cost']) * 1e4:.2f}` bps.")
    lines.append(f"- daily fee cost: `{float(best['daily_fee_cost']) * 1e4:.2f}` bps.")
    lines.append(f"- mean active names: `{float(best['mean_active_name_count']):.1f}`.")
    lines.append("")
    lines.append("## 完整对比表")
    lines.append("")
    cols = [
        "strategy_name",
        "open_quantile",
        "close_quantile",
        "gross_daily_return",
        "gross_sharpe",
        "gross_max_drawdown",
        "net_daily_return",
        "net_sharpe",
        "daily_turnover",
        "p95_daily_turnover",
        "daily_spread_cost",
        "daily_fee_cost",
        "daily_total_cost",
        "mean_gross_exposure",
        "mean_active_name_count",
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
    lines.append("## 回测曲线")
    lines.append("")
    lines.append("![Gross wealth](percentile_hysteresis_baseline_gross_wealth_curve.png)")
    lines.append("")
    lines.append("![Net wealth](percentile_hysteresis_baseline_net_wealth_curve.png)")
    lines.append("")
    lines.append("![Gross drawdown](percentile_hysteresis_baseline_gross_drawdown_curve.png)")
    lines.append("")
    lines.append("![Net drawdown](percentile_hysteresis_baseline_net_drawdown_curve.png)")
    lines.append("")
    lines.append("## 复盘")
    lines.append("")
    lines.append("这份报告回答两个基础问题: prediction 分位数 hysteresis 在不引入复杂组合构建时是否有 gross alpha, 以及在 spread + fee 成本模型下是否能覆盖交易成本。")
    lines.append("")
    lines.append("后续应在同一规则上逐步加回之前实验过的模块: tradable / fillable, liquidity, cost-aware sizing, cost-aware entry score。")
    lines.append("")
    lines.append("## 输出位置")
    lines.append("")
    lines.append(f"- experiment root: `{EXPERIMENT_ROOT.as_posix()}`.")
    lines.append("- summary CSV: `percentile_hysteresis_baseline_summary.csv`.")
    lines.append("- report HTML: `percentile_hysteresis_baseline_report.html`.")
    return "\n".join(lines)


def write_html_report(markdown_text: str, summary_df: pd.DataFrame, curve_paths: dict[str, Path], output_path: Path) -> None:
    """Write one compact HTML report."""
    # Render tables and figures first; keep Markdown Appendix as the last section.
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
    plot_path = EXPERIMENT_ROOT / "percentile_hysteresis_baseline_comparison.png"
    sections = [
        render_section("Summary Table", render_table(headers, rows)),
        render_section(
            "Comparison Figure",
            render_embedded_figure("Comparison Figure", plot_path, "Gross, net, turnover, spread, fee, and active-name diagnostics."),
        ),
        render_section(
            "Backtest Curves",
            "\n".join(
                [
                    render_embedded_figure("Gross Wealth", Path(curve_paths["gross_wealth"]), "Cumulative gross wealth across threshold variants."),
                    render_embedded_figure("Net Wealth", Path(curve_paths["net_wealth"]), "Cumulative net wealth after spread and fee costs."),
                    render_embedded_figure("Gross Drawdown", Path(curve_paths["gross_drawdown"]), "Gross drawdown across threshold variants."),
                    render_embedded_figure("Net Drawdown", Path(curve_paths["net_drawdown"]), "Net drawdown after spread and fee costs."),
                ]
            ),
        ),
        render_section("Markdown Appendix", render_code_block(markdown_text)),
    ]
    html = build_page("10min Percentile Hysteresis Baseline", "Long-only signal percentile hysteresis baseline report.", sections)
    Path(output_path).write_text(html, encoding="utf-8")


def run_baseline() -> Path:
    """Run the full baseline matrix and write unified outputs."""
    # Prepare directories and the feature data source.
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    REPO_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    feature_glob = load_chunk_manifest_glob(FEATURE_MANIFEST_PATH)
    con = connect_duckdb(Path(":memory:"))
    dates = load_available_dates(con, feature_glob)
    variants = build_variants()

    # Run variants and collect summary rows.
    rows: list[dict[str, object]] = []
    for variant in variants:
        payload = summarize_variant(con, feature_glob, dates, variant)
        rows.append(flatten_summary(payload))
    con.close()

    # Persist unified outputs.
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(EXPERIMENT_ROOT / "percentile_hysteresis_baseline_summary.csv", index=False)
    (EXPERIMENT_ROOT / "percentile_hysteresis_baseline_summary.yaml").write_text(yaml.safe_dump({"rows": rows}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    plot_comparison(summary_df, EXPERIMENT_ROOT / "percentile_hysteresis_baseline_comparison.png")
    curve_paths = plot_backtest_curves(variants)
    markdown_text = build_markdown_report(summary_df)
    report_path = EXPERIMENT_ROOT / "percentile_hysteresis_baseline_report.md"
    report_path.write_text(markdown_text, encoding="utf-8")
    repo_report_path = REPO_REPORT_DIR / "percentile_hysteresis_baseline_report_0516.md"
    repo_report_path.write_text(markdown_text, encoding="utf-8")
    for curve_path in list(curve_paths.values()):
        (REPO_REPORT_DIR / Path(curve_path).name).write_bytes(Path(curve_path).read_bytes())
    (REPO_REPORT_DIR / "percentile_hysteresis_baseline_comparison.png").write_bytes((EXPERIMENT_ROOT / "percentile_hysteresis_baseline_comparison.png").read_bytes())
    write_html_report(markdown_text, summary_df, curve_paths, EXPERIMENT_ROOT / "percentile_hysteresis_baseline_report.html")
    write_html_report(markdown_text, summary_df, curve_paths, REPO_REPORT_DIR / "percentile_hysteresis_baseline_report_0516.html")
    return report_path


def main() -> None:
    """Run the baseline script."""
    # Execute the fixed matrix and print the report path.
    report_path = run_baseline()
    print(Path(report_path).as_posix())


if __name__ == "__main__":
    main()
