"""Diagnose prediction power, liquidity interaction, and alpha per turnover."""

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
from portfolio_backtest.simulator import connect_duckdb
from prediction_nn2.html_report import build_page, render_code_block, render_embedded_figure, render_section, render_table


EXPERIMENT_ROOT = Path("/data-cache/nn/trade_plan_experiments/0516_model_signal_diagnostics")
REPO_REPORT_DIR = REPO_ROOT / "report" / "0516"
FEATURE_MANIFEST_PATH = Path("/data-cache/nn/trade_plan_experiments/0515/features/entry1_h60_slot60/feature_manifest.yaml")
BASELINE_ROOT = Path("/data-cache/nn/trade_plan_experiments/0516_percentile_hysteresis_baseline")
MIN_TIMESTAMP_NAME_COUNT = 1000
FEE_BPS = 1.0
SPREAD_BPS_HIGH = 5.0
SPREAD_BPS_MID = 10.0
SPREAD_BPS_LOW = 20.0


def prepare_signal_table(con: Any, feature_glob: str) -> None:
    """Create the reusable ranked signal table inside DuckDB."""
    # Rank prediction and liquidity inside each complete 10min cross-section.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE signal_base AS
        WITH base AS (
            SELECT
                CAST(date AS INTEGER) AS date,
                CAST(time AS INTEGER) AS time,
                CAST(minute_slot AS INTEGER) AS minute_slot,
                CAST(code AS INTEGER) AS code,
                CAST(prediction AS DOUBLE) AS prediction,
                CAST(ret_vwap_exec_10 AS DOUBLE) AS simple_return,
                CAST(signal_amount AS DOUBLE) AS signal_amount
            FROM read_parquet('{feature_glob}')
            WHERE prediction IS NOT NULL
              AND ret_vwap_exec_10 IS NOT NULL
              AND signal_amount IS NOT NULL
              AND minute_slot % 10 = 0
        ),
        complete_times AS (
            SELECT date, time
            FROM base
            GROUP BY date, time
            HAVING count(*) >= {int(MIN_TIMESTAMP_NAME_COUNT)}
        ),
        ranked AS (
            SELECT
                base.*,
                ntile(10) OVER (PARTITION BY base.date, base.time ORDER BY base.prediction) AS signal_decile,
                ntile(3) OVER (PARTITION BY base.date, base.time ORDER BY base.signal_amount DESC) AS liq_bucket
            FROM base
            INNER JOIN complete_times
                ON base.date = complete_times.date
               AND base.time = complete_times.time
        )
        SELECT
            date,
            time,
            minute_slot,
            code,
            prediction,
            simple_return,
            signal_amount,
            signal_decile,
            liq_bucket,
            CASE
                WHEN liq_bucket = 1 THEN {float(SPREAD_BPS_HIGH):.6f}
                WHEN liq_bucket = 2 THEN {float(SPREAD_BPS_MID):.6f}
                ELSE {float(SPREAD_BPS_LOW):.6f}
            END AS spread_bps
        FROM ranked
        """
    )


def compute_signal_decile_summary(con: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize return monotonicity by prediction decile."""
    # Aggregate by date first so stability metrics are daily statistics.
    daily = con.execute(
        """
        SELECT
            date,
            signal_decile,
            count(*) AS row_count,
            avg(prediction) AS mean_prediction,
            avg(simple_return) AS mean_return,
            avg(spread_bps) AS mean_spread_bps
        FROM signal_base
        GROUP BY date, signal_decile
        ORDER BY date, signal_decile
        """
    ).fetchdf()

    # Collapse daily rows into one summary row per decile.
    summary = (
        daily.groupby("signal_decile", sort=True)
        .agg(
            mean_daily_return=("mean_return", "mean"),
            std_daily_return=("mean_return", "std"),
            hit_rate=("mean_return", lambda values: float((values.astype(float) > 0.0).mean())),
            mean_prediction=("mean_prediction", "mean"),
            mean_spread_bps=("mean_spread_bps", "mean"),
            mean_row_count=("row_count", "mean"),
        )
        .reset_index()
    )
    summary["daily_return_bps"] = summary["mean_daily_return"].astype(float) * 1e4
    summary["t_stat"] = (
        summary["mean_daily_return"].astype(float)
        / summary["std_daily_return"].astype(float)
        * np.sqrt(float(daily["date"].nunique()))
    )
    return daily, summary


def compute_monotonicity(daily_decile: pd.DataFrame) -> pd.DataFrame:
    """Measure daily top-minus-bottom spread and rank monotonicity."""
    # Pivot daily decile returns and compute top-bottom gross spread.
    pivot = daily_decile.pivot(index="date", columns="signal_decile", values="mean_return").sort_index()
    rows: list[dict[str, float]] = []
    for date, row in pivot.iterrows():
        series = row.dropna().astype(float)
        corr = float(pd.Series(series.index.astype(float)).corr(pd.Series(series.values), method="spearman"))
        rows.append(
            {
                "date": float(date),
                "top_decile_return": float(row[10]),
                "bottom_decile_return": float(row[1]),
                "top_minus_bottom": float(row[10] - row[1]),
                "spearman_decile_return": float(corr),
            }
        )
    monotonicity = pd.DataFrame(rows)

    # Add cumulative spread for visual inspection.
    monotonicity["date_ts"] = pd.to_datetime(monotonicity["date"].astype(int).astype(str), format="%y%m%d")
    monotonicity["cum_top_minus_bottom"] = (1.0 + monotonicity["top_minus_bottom"].astype(float)).cumprod() - 1.0
    return monotonicity


def compute_liquidity_signal_summary(con: Any) -> pd.DataFrame:
    """Summarize signal quality across prediction decile and liquidity bucket."""
    # Estimate a one-shot net proxy using half-spread plus fee for entry notional.
    summary = con.execute(
        f"""
        SELECT
            liq_bucket,
            signal_decile,
            count(*) AS row_count,
            avg(simple_return) AS mean_return,
            avg(spread_bps) AS mean_spread_bps,
            avg(simple_return - 0.5 * spread_bps / 10000.0 - {float(FEE_BPS):.6f} / 10000.0) AS mean_entry_net_proxy
        FROM signal_base
        GROUP BY liq_bucket, signal_decile
        ORDER BY liq_bucket, signal_decile
        """
    ).fetchdf()

    # Convert return fields to bps for report readability.
    summary["gross_return_bps"] = summary["mean_return"].astype(float) * 1e4
    summary["entry_net_proxy_bps"] = summary["mean_entry_net_proxy"].astype(float) * 1e4
    return summary


def compute_alpha_per_turnover() -> pd.DataFrame:
    """Summarize realized baseline gross alpha per unit turnover."""
    # Read the baseline daily files produced by the existing backtest.
    rows: list[dict[str, float | str]] = []
    for daily_path in sorted((BASELINE_ROOT / "variants").glob("*/percentile_hysteresis_daily.csv")):
        strategy_name = daily_path.parent.name
        daily = pd.read_csv(daily_path)

        # Compute gross and net return per unit of turnover.
        turnover = daily["turnover"].astype(float)
        gross_return = daily["gross_return"].astype(float)
        net_return = daily["net_return"].astype(float)
        total_cost = daily["spread_cost"].astype(float) + daily["fee_cost"].astype(float)
        rows.append(
            {
                "strategy_name": strategy_name,
                "mean_gross_return": float(gross_return.mean()),
                "mean_net_return": float(net_return.mean()),
                "mean_turnover": float(turnover.mean()),
                "mean_total_cost": float(total_cost.mean()),
                "gross_bps_per_turnover": float(gross_return.sum() / turnover.sum() * 1e4),
                "net_bps_per_turnover": float(net_return.sum() / turnover.sum() * 1e4),
                "cost_bps_per_turnover": float(total_cost.sum() / turnover.sum() * 1e4),
                "positive_gross_day_rate": float((gross_return > 0.0).mean()),
                "positive_net_day_rate": float((net_return > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("gross_bps_per_turnover", ascending=False).reset_index(drop=True)


def plot_signal_decile(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot average future return by signal decile."""
    # Draw the monotonicity check as a simple decile bar chart.
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(summary["signal_decile"].astype(int), summary["daily_return_bps"].astype(float), color="#2E86AB")
    axis.axhline(0.0, color="#1f2933", linewidth=1.0)
    axis.set_title("Forward Return by Prediction Decile")
    axis.set_xlabel("Prediction decile, 10 = highest prediction")
    axis.set_ylabel("Mean 10min return, bps")
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_monotonicity(monotonicity: pd.DataFrame, output_path: Path) -> None:
    """Plot cumulative top-minus-bottom return."""
    # Draw the cumulative daily spread between top and bottom deciles.
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.plot(monotonicity["date_ts"], monotonicity["cum_top_minus_bottom"].astype(float) * 100.0, color="#3D405B", linewidth=1.8)
    axis.axhline(0.0, color="#1f2933", linewidth=1.0)
    axis.set_title("Cumulative Top-Minus-Bottom Decile Spread")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative spread, %")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_liquidity_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot gross return by signal decile and liquidity bucket."""
    # Convert the grouped result into a heatmap matrix.
    matrix = summary.pivot(index="liq_bucket", columns="signal_decile", values="gross_return_bps").sort_index()
    fig, axis = plt.subplots(figsize=(12, 5))
    image = axis.imshow(matrix.values, aspect="auto", cmap="RdYlGn")
    axis.set_title("Gross Return by Liquidity Bucket and Prediction Decile")
    axis.set_xlabel("Prediction decile, 10 = highest prediction")
    axis.set_ylabel("Liquidity bucket, 1 = highest signal_amount")
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels([str(int(value)) for value in matrix.columns])
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels([str(int(value)) for value in matrix.index])
    fig.colorbar(image, ax=axis, label="Mean 10min return, bps")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_alpha_per_turnover(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot baseline alpha and cost per unit turnover."""
    # Compare realized gross alpha, cost, and net alpha per turnover unit.
    df = summary.copy().reset_index(drop=True)
    x = np.arange(int(df.shape[0]))
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(x - 0.25, df["gross_bps_per_turnover"].astype(float), width=0.25, label="gross", color="#2E86AB")
    axis.bar(x, df["cost_bps_per_turnover"].astype(float), width=0.25, label="cost", color="#E07A5F")
    axis.bar(x + 0.25, df["net_bps_per_turnover"].astype(float), width=0.25, label="net", color="#81B29A")
    axis.axhline(0.0, color="#1f2933", linewidth=1.0)
    axis.set_title("Realized Baseline Alpha per Turnover")
    axis.set_xlabel("Strategy")
    axis.set_ylabel("bps per turnover")
    axis.set_xticks(x)
    axis.set_xticklabels(df["strategy_name"].astype(str).tolist(), rotation=25, ha="right")
    axis.legend()
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def frame_to_markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Render selected DataFrame columns as markdown table lines."""
    # Build header and separator lines.
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]

    # Format each row with compact numeric precision.
    for row in df.loc[:, columns].itertuples(index=False):
        values: list[str] = []
        for value in list(row):
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_markdown_report(
    decile_summary: pd.DataFrame,
    monotonicity: pd.DataFrame,
    liquidity_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
) -> str:
    """Build the model signal diagnostics markdown report."""
    # Extract headline numbers used in the written diagnosis.
    top_row = decile_summary.loc[decile_summary["signal_decile"].astype(int) == 10].iloc[0]
    bottom_row = decile_summary.loc[decile_summary["signal_decile"].astype(int) == 1].iloc[0]
    top_minus_bottom_bps = (float(top_row["mean_daily_return"]) - float(bottom_row["mean_daily_return"])) * 1e4
    positive_spread_rate = float((monotonicity["top_minus_bottom"].astype(float) > 0.0).mean())
    mean_spearman = float(monotonicity["spearman_decile_return"].astype(float).mean())
    best_turnover = turnover_summary.iloc[0].to_dict()
    top_liq = liquidity_summary[liquidity_summary["signal_decile"].astype(int) == 10].copy()
    top_liq = top_liq.sort_values("gross_return_bps", ascending=False).reset_index(drop=True)

    # Write the report in Chinese with English terms for common quant concepts.
    lines: list[str] = []
    lines.append("# Model Signal Diagnostics 研究报告")
    lines.append("")
    lines.append("## 研究目标")
    lines.append("")
    lines.append("本诊断从模型端回答一个问题: 当前 prediction 是完全没有 prediction power, 还是有 gross alpha 但 alpha density / tradability 不足。诊断不改变 baseline 回测规则, 只分析 feature manifest 中的 `prediction`, `ret_vwap_exec_10`, `signal_amount` 以及 baseline 已实现的 spread + fee 成本假设。")
    lines.append("")
    lines.append("## 诊断设计")
    lines.append("")
    lines.append(f"- 样本: `minute_slot % 10 = 0`, 且 `prediction`、`ret_vwap_exec_10`、`signal_amount` 非空。")
    lines.append(f"- data completeness: 每个 10min 截面至少 `{int(MIN_TIMESTAMP_NAME_COUNT)}` 只股票。")
    lines.append("- signal monotonicity: 每个 10min 截面对 `prediction` 做 decile, 观察 decile forward return 是否单调。")
    lines.append("- liquidity interaction: 每个截面对 `signal_amount` 做 3 桶, 其中 `liq_bucket = 1` 代表最高流动性, 并观察 alpha 是否集中在高成本桶。")
    lines.append("- alpha per turnover: 使用已跑完的 percentile hysteresis baseline daily output, 计算 realized gross / cost / net bps per turnover。")
    lines.append("")
    lines.append("## 核心结论")
    lines.append("")
    lines.append(f"- top decile 平均 10min return 为 `{float(top_row['daily_return_bps']):.3f}` bps, bottom decile 为 `{float(bottom_row['daily_return_bps']):.3f}` bps, top-minus-bottom 为 `{top_minus_bottom_bps:.3f}` bps。")
    lines.append(f"- top-minus-bottom 在 `{positive_spread_rate:.1%}` 的交易日为正, 平均 decile-return Spearman correlation 为 `{mean_spearman:.3f}`。")
    lines.append(f"- top decile 中 gross return 最好的 liquidity bucket 是 `{int(top_liq.iloc[0]['liq_bucket'])}`, gross return `{float(top_liq.iloc[0]['gross_return_bps']):.3f}` bps, entry net proxy `{float(top_liq.iloc[0]['entry_net_proxy_bps']):.3f}` bps。")
    lines.append(f"- realized baseline 中 gross bps per turnover 最好的方案是 `{best_turnover['strategy_name']}`, gross `{float(best_turnover['gross_bps_per_turnover']):.3f}` bps/turnover, cost `{float(best_turnover['cost_bps_per_turnover']):.3f}` bps/turnover, net `{float(best_turnover['net_bps_per_turnover']):.3f}` bps/turnover。")
    lines.append("")
    lines.append("整体判断: prediction 不是完全没有 prediction power, 但当前 top-tail alpha 很薄, 且每单位 turnover 的 gross alpha 明显低于 spread + fee 成本。模型改进应优先提高 top quantile ranking 的 alpha density, 并显式建模 liquidity / spread interaction。")
    lines.append("")
    lines.append("## Signal Decile Summary")
    lines.append("")
    lines.extend(
        frame_to_markdown_table(
            decile_summary,
            ["signal_decile", "daily_return_bps", "t_stat", "hit_rate", "mean_prediction", "mean_spread_bps", "mean_row_count"],
        )
    )
    lines.append("")
    lines.append("## Liquidity x Signal Summary")
    lines.append("")
    lines.extend(
        frame_to_markdown_table(
            liquidity_summary,
            ["liq_bucket", "signal_decile", "gross_return_bps", "entry_net_proxy_bps", "mean_spread_bps", "row_count"],
        )
    )
    lines.append("")
    lines.append("## Alpha Per Turnover")
    lines.append("")
    lines.extend(
        frame_to_markdown_table(
            turnover_summary,
            [
                "strategy_name",
                "gross_bps_per_turnover",
                "cost_bps_per_turnover",
                "net_bps_per_turnover",
                "mean_turnover",
                "positive_gross_day_rate",
                "positive_net_day_rate",
            ],
        )
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("![Signal decile return](model_signal_decile_return.png)")
    lines.append("")
    lines.append("![Top minus bottom spread](model_signal_top_minus_bottom.png)")
    lines.append("")
    lines.append("![Liquidity heatmap](model_signal_liquidity_heatmap.png)")
    lines.append("")
    lines.append("![Alpha per turnover](model_signal_alpha_per_turnover.png)")
    lines.append("")
    lines.append("## 后续方向")
    lines.append("")
    lines.append("- 如果 top decile 的 gross alpha 主要来自低流动性桶, 下一步应做 cost-aware label 或 liquidity-conditional model。")
    lines.append("- 如果 decile monotonicity 稳定但 alpha per turnover 过低, 下一步应做 top-tail ranking loss 和 longer-horizon / persistence 目标。")
    lines.append("- 后续模型评估必须把普通 IC 拆成 top quantile return、top-minus-bottom、liquidity bucket alpha、gross bps per turnover、net bps per turnover。")
    lines.append("")
    lines.append("## 输出位置")
    lines.append("")
    lines.append(f"- experiment root: `{EXPERIMENT_ROOT.as_posix()}`.")
    lines.append("- markdown report: `model_signal_diagnostics_report.md`.")
    lines.append("- html report: `model_signal_diagnostics_report.html`.")
    return "\n".join(lines)


def write_html_report(markdown_text: str, tables: dict[str, pd.DataFrame], figure_paths: dict[str, Path], output_path: Path) -> None:
    """Write a self-contained HTML diagnostics report."""
    # Render compact tables before figures, with the markdown appendix last.
    sections: list[str] = []
    for title, df in tables.items():
        headers = list(df.columns)
        rows: list[list[str]] = []
        for row in df.itertuples(index=False):
            values: list[str] = []
            for value in list(row):
                if isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value))
            rows.append(values)
        sections.append(render_section(str(title), render_table(headers, rows)))

    # Embed all figures so the HTML can be moved as a single file.
    figure_html = "\n".join(
        [
            render_embedded_figure("Signal Decile Return", figure_paths["signal_decile"], "Mean 10min forward return by prediction decile."),
            render_embedded_figure("Top Minus Bottom", figure_paths["monotonicity"], "Cumulative daily spread between top and bottom prediction deciles."),
            render_embedded_figure("Liquidity Heatmap", figure_paths["liquidity"], "Gross return by liquidity bucket and prediction decile."),
            render_embedded_figure("Alpha Per Turnover", figure_paths["turnover"], "Realized baseline gross, cost, and net bps per turnover."),
        ]
    )
    sections.append(render_section("Figures", figure_html))
    sections.append(render_section("Markdown Appendix", render_code_block(markdown_text)))
    html = build_page("Model Signal Diagnostics", "Prediction power, liquidity interaction, and alpha per turnover.", sections)
    Path(output_path).write_text(html, encoding="utf-8")


def run_diagnostics() -> Path:
    """Run all signal diagnostics and write unified outputs."""
    # Prepare output directories and feature data source.
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    REPO_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    feature_glob = load_chunk_manifest_glob(FEATURE_MANIFEST_PATH)
    con = connect_duckdb(Path(":memory:"))
    prepare_signal_table(con, feature_glob)

    # Compute the three diagnostic families.
    daily_decile, decile_summary = compute_signal_decile_summary(con)
    monotonicity = compute_monotonicity(daily_decile)
    liquidity_summary = compute_liquidity_signal_summary(con)
    turnover_summary = compute_alpha_per_turnover()
    con.close()

    # Persist tabular outputs.
    daily_decile.to_csv(EXPERIMENT_ROOT / "model_signal_decile_daily.csv", index=False)
    decile_summary.to_csv(EXPERIMENT_ROOT / "model_signal_decile_summary.csv", index=False)
    monotonicity.drop(columns=["date_ts"]).to_csv(EXPERIMENT_ROOT / "model_signal_monotonicity_daily.csv", index=False)
    liquidity_summary.to_csv(EXPERIMENT_ROOT / "model_signal_liquidity_summary.csv", index=False)
    turnover_summary.to_csv(EXPERIMENT_ROOT / "model_signal_alpha_per_turnover.csv", index=False)
    yaml_payload = {
        "decile_summary": decile_summary.to_dict(orient="records"),
        "monotonicity": {
            "mean_top_minus_bottom": float(monotonicity["top_minus_bottom"].astype(float).mean()),
            "positive_spread_rate": float((monotonicity["top_minus_bottom"].astype(float) > 0.0).mean()),
            "mean_spearman_decile_return": float(monotonicity["spearman_decile_return"].astype(float).mean()),
        },
        "alpha_per_turnover": turnover_summary.to_dict(orient="records"),
    }
    (EXPERIMENT_ROOT / "model_signal_diagnostics_summary.yaml").write_text(yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Draw figures and write reports.
    figure_paths = {
        "signal_decile": EXPERIMENT_ROOT / "model_signal_decile_return.png",
        "monotonicity": EXPERIMENT_ROOT / "model_signal_top_minus_bottom.png",
        "liquidity": EXPERIMENT_ROOT / "model_signal_liquidity_heatmap.png",
        "turnover": EXPERIMENT_ROOT / "model_signal_alpha_per_turnover.png",
    }
    plot_signal_decile(decile_summary, figure_paths["signal_decile"])
    plot_monotonicity(monotonicity, figure_paths["monotonicity"])
    plot_liquidity_heatmap(liquidity_summary, figure_paths["liquidity"])
    plot_alpha_per_turnover(turnover_summary, figure_paths["turnover"])
    markdown_text = build_markdown_report(decile_summary, monotonicity, liquidity_summary, turnover_summary)
    report_path = EXPERIMENT_ROOT / "model_signal_diagnostics_report.md"
    report_path.write_text(markdown_text, encoding="utf-8")
    (REPO_REPORT_DIR / "model_signal_diagnostics_report_0516.md").write_text(markdown_text, encoding="utf-8")
    for figure_path in figure_paths.values():
        (REPO_REPORT_DIR / Path(figure_path).name).write_bytes(Path(figure_path).read_bytes())
    write_html_report(
        markdown_text,
        {
            "Signal Decile Summary": decile_summary,
            "Liquidity x Signal Summary": liquidity_summary,
            "Alpha Per Turnover": turnover_summary,
        },
        figure_paths,
        EXPERIMENT_ROOT / "model_signal_diagnostics_report.html",
    )
    write_html_report(
        markdown_text,
        {
            "Signal Decile Summary": decile_summary,
            "Liquidity x Signal Summary": liquidity_summary,
            "Alpha Per Turnover": turnover_summary,
        },
        figure_paths,
        REPO_REPORT_DIR / "model_signal_diagnostics_report_0516.html",
    )
    return report_path


def main() -> None:
    """Run the diagnostics script."""
    # Execute all diagnostics and print the markdown report path.
    report_path = run_diagnostics()
    print(Path(report_path).as_posix())


if __name__ == "__main__":
    main()
