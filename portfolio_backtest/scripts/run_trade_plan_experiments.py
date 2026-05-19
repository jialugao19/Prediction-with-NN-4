"""Run trade-plan improvement experiments and write one unified research report."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio_backtest.contract import PortfolioBacktestConfig, build_default_portfolio_backtest_config, load_chunk_manifest_glob
from portfolio_backtest.data_source import materialize_feature_chunks
from portfolio_backtest.simulator import connect_duckdb, materialize_slot_strategy_table, summarize_return_series
from prediction_nn2.html_report import build_page, render_code_block, render_section, render_table


EXPERIMENT_ROOT = Path("/data-cache/nn/trade_plan_experiments/0515")
REPO_REPORT_DIR = REPO_ROOT / "report" / "0515"
AUM_FOR_NET = 10_000_000.0


def build_trade_plan_variants() -> list[dict[str, object]]:
    """Define the fixed trade-plan experiment matrix."""
    # Compare rebalance and holding periods first because turnover is the primary failure mode.
    variants: list[dict[str, object]] = [
        {
            "strategy_name": "baseline_h10_top10_ls_liq3",
            "experiment_group": "baseline",
            "holding_bars": 10,
            "slot_mod_bars": 10,
            "top_frac": 0.10,
            "long_enabled": True,
            "short_enabled": True,
            "max_liq_bucket": 3,
        },
        {
            "strategy_name": "holding_h20_top10_ls_liq3",
            "experiment_group": "holding",
            "holding_bars": 20,
            "slot_mod_bars": 20,
            "top_frac": 0.10,
            "long_enabled": True,
            "short_enabled": True,
            "max_liq_bucket": 3,
        },
        {
            "strategy_name": "holding_h30_top10_ls_liq3",
            "experiment_group": "holding",
            "holding_bars": 30,
            "slot_mod_bars": 30,
            "top_frac": 0.10,
            "long_enabled": True,
            "short_enabled": True,
            "max_liq_bucket": 3,
        },
        {
            "strategy_name": "holding_h60_top10_ls_liq3",
            "experiment_group": "holding",
            "holding_bars": 60,
            "slot_mod_bars": 60,
            "top_frac": 0.10,
            "long_enabled": True,
            "short_enabled": True,
            "max_liq_bucket": 3,
        },
    ]

    # Compare group widths on the baseline execution cadence.
    variants.extend(
        [
            {
                "strategy_name": "width_h10_top05_ls_liq3",
                "experiment_group": "width",
                "holding_bars": 10,
                "slot_mod_bars": 10,
                "top_frac": 0.05,
                "long_enabled": True,
                "short_enabled": True,
                "max_liq_bucket": 3,
            },
            {
                "strategy_name": "width_h10_top20_ls_liq3",
                "experiment_group": "width",
                "holding_bars": 10,
                "slot_mod_bars": 10,
                "top_frac": 0.20,
                "long_enabled": True,
                "short_enabled": True,
                "max_liq_bucket": 3,
            },
            {
                "strategy_name": "width_h10_top30_ls_liq3",
                "experiment_group": "width",
                "holding_bars": 10,
                "slot_mod_bars": 10,
                "top_frac": 0.30,
                "long_enabled": True,
                "short_enabled": True,
                "max_liq_bucket": 3,
            },
        ]
    )

    # Split long and short legs to identify whether the alpha is symmetric.
    variants.extend(
        [
            {
                "strategy_name": "leg_h10_top10_long_only_liq3",
                "experiment_group": "leg",
                "holding_bars": 10,
                "slot_mod_bars": 10,
                "top_frac": 0.10,
                "long_enabled": True,
                "short_enabled": False,
                "max_liq_bucket": 3,
            },
            {
                "strategy_name": "leg_h10_top10_short_only_liq3",
                "experiment_group": "leg",
                "holding_bars": 10,
                "slot_mod_bars": 10,
                "top_frac": 0.10,
                "long_enabled": False,
                "short_enabled": True,
                "max_liq_bucket": 3,
            },
        ]
    )

    # Filter low-liquidity names to test capacity and cost sensitivity.
    variants.extend(
        [
            {
                "strategy_name": "liq_h10_top10_ls_liq2",
                "experiment_group": "liquidity",
                "holding_bars": 10,
                "slot_mod_bars": 10,
                "top_frac": 0.10,
                "long_enabled": True,
                "short_enabled": True,
                "max_liq_bucket": 2,
            },
            {
                "strategy_name": "liq_h10_top10_ls_liq1",
                "experiment_group": "liquidity",
                "holding_bars": 10,
                "slot_mod_bars": 10,
                "top_frac": 0.10,
                "long_enabled": True,
                "short_enabled": True,
                "max_liq_bucket": 1,
            },
        ]
    )
    return variants


def build_variant_config(base_config: PortfolioBacktestConfig, variant: dict[str, object], experiment_root: Path) -> PortfolioBacktestConfig:
    """Build one portfolio backtest config for a named experiment variant."""
    # Resolve the variant output layout.
    strategy_name = str(variant["strategy_name"])
    output_dir = Path(experiment_root) / "variants" / strategy_name

    # Return a dataclass copy with only trade-plan fields changed.
    return replace(
        base_config,
        output_dir=output_dir,
        feature_db_path=output_dir / "portfolio_backtest.duckdb",
        feature_chunk_dir=output_dir / "feature_chunks",
        feature_manifest_path=output_dir / "feature_manifest.yaml",
        top_frac=float(variant["top_frac"]),
        slot_mod_bars=int(variant["slot_mod_bars"]),
        long_enabled=bool(variant["long_enabled"]),
        short_enabled=bool(variant["short_enabled"]),
        max_liq_bucket=int(variant["max_liq_bucket"]),
        holding_bars=int(variant["holding_bars"]),
        report_title=f"Trade Plan Experiment: {strategy_name}",
    )


def build_feature_config(base_config: PortfolioBacktestConfig, variant: dict[str, object], experiment_root: Path) -> PortfolioBacktestConfig:
    """Build one shared feature config for a holding/slot pair."""
    # Place feature artifacts outside strategy variants so multiple variants can reuse them.
    feature_key = f"entry{int(base_config.entry_delay_bars)}_h{int(variant['holding_bars'])}_slot{int(variant['slot_mod_bars'])}"
    feature_root = Path(experiment_root) / "features" / feature_key
    return replace(
        base_config,
        output_dir=feature_root,
        feature_db_path=feature_root / "feature_only.duckdb",
        feature_chunk_dir=feature_root / "feature_chunks",
        feature_manifest_path=feature_root / "feature_manifest.yaml",
        slot_mod_bars=int(variant["slot_mod_bars"]),
        holding_bars=int(variant["holding_bars"]),
    )


def resolve_feature_glob(base_config: PortfolioBacktestConfig, variant: dict[str, object], experiment_root: Path) -> str:
    """Resolve or build the shared feature parquet glob for one holding/slot pair."""
    # Reuse the interrupted baseline feature output when it already exists.
    if int(variant["holding_bars"]) == 10 and int(variant["slot_mod_bars"]) == 10:
        legacy_manifest = Path(experiment_root) / "variants" / "baseline_h10_top10_ls_liq3" / "feature_manifest.yaml"
        if legacy_manifest.exists():
            return load_chunk_manifest_glob(legacy_manifest)

    # Build or reuse the shared feature output for this holding/slot pair.
    feature_config = build_feature_config(base_config, variant, experiment_root)
    if feature_config.feature_manifest_path.exists():
        return load_chunk_manifest_glob(feature_config.feature_manifest_path)
    return materialize_feature_chunks(feature_config)


def summarize_fast_variant(config: PortfolioBacktestConfig, feature_glob: str) -> dict[str, Any]:
    """Run a fast VWAP-only trade-plan summary for one variant."""
    # Materialize the VWAP position table with production portfolio-construction SQL.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    con = connect_duckdb(config.feature_db_path)
    materialize_slot_strategy_table(
        con,
        feature_glob,
        config.top_frac,
        config.long_enabled,
        config.short_enabled,
        config.max_liq_bucket,
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

    # Compute bar-level return and turnover proxies in DuckDB to avoid Python row loops.
    bar_df = con.execute(
        f"""
        WITH base AS (
            SELECT
                minute_slot,
                slot_bar_id,
                date,
                time,
                code,
                CASE WHEN fillable THEN target_weight ELSE 0.0 END AS weight,
                simple_return,
                spread_bps,
                sigma_intraday,
                adv_amount,
                lag(CASE WHEN fillable THEN target_weight ELSE 0.0 END) OVER (
                    PARTITION BY minute_slot, code ORDER BY slot_bar_id
                ) AS prev_weight_raw,
                lag(slot_bar_id) OVER (PARTITION BY minute_slot, code ORDER BY slot_bar_id) AS prev_slot_bar_id,
                lead(slot_bar_id) OVER (PARTITION BY minute_slot, code ORDER BY slot_bar_id) AS next_slot_bar_id
            FROM vwap_slot_positions
        ),
        delta AS (
            SELECT
                minute_slot,
                slot_bar_id,
                date,
                time,
                weight,
                simple_return,
                spread_bps,
                sigma_intraday,
                adv_amount,
                abs(weight - CASE WHEN prev_slot_bar_id = slot_bar_id - 1 THEN prev_weight_raw ELSE 0.0 END) AS rebalance_abs_delta,
                CASE WHEN next_slot_bar_id IS NULL OR next_slot_bar_id > slot_bar_id + 1 THEN abs(weight) ELSE 0.0 END AS exit_abs_delta
            FROM base
        )
        SELECT
            minute_slot,
            slot_bar_id,
            date,
            time,
            sum(weight * simple_return) AS gross_return,
            0.5 * (sum(rebalance_abs_delta) + sum(exit_abs_delta)) AS turnover,
            sum(0.5 * spread_bps / 10000.0 * (rebalance_abs_delta + exit_abs_delta)) AS spread_cost,
            sum({float(config.impact_eta):.8f} * sigma_intraday * (rebalance_abs_delta + exit_abs_delta)
                * sqrt((rebalance_abs_delta + exit_abs_delta) / adv_amount)) AS impact_coeff,
            avg(CASE WHEN weight != 0 THEN 1.0 ELSE 0.0 END) AS fill_ratio,
            sum(CASE WHEN weight > 0 THEN weight ELSE 0.0 END) AS long_exposure,
            sum(CASE WHEN weight < 0 THEN -weight ELSE 0.0 END) AS short_exposure
        FROM delta
        GROUP BY minute_slot, slot_bar_id, date, time
        ORDER BY minute_slot, slot_bar_id
        """
    ).fetchdf()
    con.close()

    # Aggregate bar rows into equal-capital slot-combined daily returns.
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
        )
        .reset_index()
    )
    combined_daily["net_return_10m"] = (
        combined_daily["gross_return"].astype(float)
        - combined_daily["spread_cost"].astype(float)
        - combined_daily["impact_coeff"].astype(float) * (float(AUM_FOR_NET) ** 0.5)
    )

    # Persist variant-level diagnostics.
    bar_df.to_csv(config.output_dir / "fast_vwap_slot_bar.csv", index=False)
    combined_daily.to_csv(config.output_dir / "fast_vwap_combined_daily.csv", index=False)
    mean_coeff = float(combined_daily["impact_coeff"].astype(float).mean())
    capacity_10bps = float(((10.0 / 10000.0) / mean_coeff) ** 2) if mean_coeff > 0.0 else float("nan")
    capacity_20bps = float(((20.0 / 10000.0) / mean_coeff) ** 2) if mean_coeff > 0.0 else float("nan")
    payload = {
        "gross": summarize_return_series(combined_daily["gross_return"], config.annual_days),
        "net_10m": summarize_return_series(combined_daily["net_return_10m"], config.annual_days),
        "turnover": {
            "mean_daily_turnover": float(combined_daily["turnover"].astype(float).mean()),
            "p50_daily_turnover": float(combined_daily["turnover"].astype(float).quantile(0.50)),
            "p95_daily_turnover": float(combined_daily["turnover"].astype(float).quantile(0.95)),
        },
        "execution": {
            "mean_fill_ratio": float(combined_daily["fill_ratio"].astype(float).mean()),
            "mean_long_exposure": float(combined_daily["long_exposure"].astype(float).mean()),
            "mean_short_exposure": float(combined_daily["short_exposure"].astype(float).mean()),
        },
        "capacity": {
            "capacity_at_10bps": capacity_10bps,
            "capacity_at_20bps": capacity_20bps,
        },
        "method": "fast_vwap_sql_proxy",
    }
    (config.output_dir / "fast_strategy_summary.yaml").write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return payload


def extract_variant_summary(config: PortfolioBacktestConfig, variant: dict[str, object], strategy: dict[str, Any]) -> dict[str, object]:
    """Flatten one variant's fast strategy summary into a comparison row."""
    # Extract the summary blocks.
    gross = dict(strategy["gross"])
    turnover = dict(strategy["turnover"])
    execution = dict(strategy["execution"])
    capacity = dict(strategy["capacity"])
    net_10m = dict(strategy["net_10m"])

    # Return a flat row that can be sorted and written to CSV/YAML.
    return {
        "strategy_name": str(variant["strategy_name"]),
        "experiment_group": str(variant["experiment_group"]),
        "output_dir": Path(config.output_dir).as_posix(),
        "holding_bars": int(config.holding_bars),
        "slot_mod_bars": int(config.slot_mod_bars),
        "top_frac": float(config.top_frac),
        "long_enabled": bool(config.long_enabled),
        "short_enabled": bool(config.short_enabled),
        "max_liq_bucket": int(config.max_liq_bucket),
        "gross_daily_return": float(gross["mean_daily_return"]),
        "gross_sharpe": float(gross["annualized_sharpe"]),
        "daily_turnover": float(turnover["mean_daily_turnover"]),
        "net_10m_daily_return": float(net_10m["mean_daily_return"]),
        "net_10m_sharpe": float(net_10m["annualized_sharpe"]),
        "net_10m_max_drawdown": float(net_10m["max_drawdown"]),
        "positive_day_ratio": float(net_10m["positive_day_ratio"]),
        "mean_fill_ratio": float(execution["mean_fill_ratio"]),
        "mean_long_exposure": float(execution["mean_long_exposure"]),
        "mean_short_exposure": float(execution["mean_short_exposure"]),
        "capacity_10bps": float(capacity["capacity_at_10bps"]),
        "capacity_20bps": float(capacity["capacity_at_20bps"]),
        "method": str(strategy["method"]),
    }


def plot_experiment_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot gross, turnover, net, and capacity across experiment variants."""
    # Prepare compact x labels and plotting values.
    df = summary_df.copy().reset_index(drop=True)
    x = list(range(int(df.shape[0])))
    labels = df["strategy_name"].astype(str).tolist()

    # Render four panels for the core trade-off.
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    axes[0].bar(x, df["gross_daily_return"].astype(float) * 1e4, color="#2E86AB")
    axes[0].set_ylabel("Gross daily bps")
    axes[0].set_title("Gross Return")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(x, df["daily_turnover"].astype(float), color="#3D405B")
    axes[1].set_ylabel("Daily turnover")
    axes[1].set_title("Turnover")
    axes[1].grid(alpha=0.25, axis="y")

    axes[2].bar(x, df["net_10m_daily_return"].astype(float) * 1e4, color="#E07A5F")
    axes[2].set_ylabel("Net 10M daily bps")
    axes[2].set_title("Net Return at 10M AUM")
    axes[2].grid(alpha=0.25, axis="y")

    axes[3].bar(x, df["capacity_10bps"].astype(float) / 1_000_000.0, color="#81B29A")
    axes[3].set_ylabel("Capacity 10bps, M")
    axes[3].set_title("Capacity")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=35, ha="right")
    axes[3].grid(alpha=0.25, axis="y")

    # Save the plot for both markdown and HTML reports.
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_markdown_report(summary_df: pd.DataFrame, experiment_root: Path, variants: list[dict[str, object]]) -> str:
    """Build the unified markdown research report for all trade-plan experiments."""
    # Sort the comparison table by net return so the reader sees the strongest candidates first.
    ordered = summary_df.sort_values(["net_10m_daily_return", "capacity_10bps"], ascending=[False, False]).reset_index(drop=True)
    best = ordered.iloc[0].to_dict()

    # Compose the report with experiment design first and results second.
    lines: list[str] = []
    lines.append("# 交易计划改善实验统一研究报告")
    lines.append("")
    lines.append("## 研究目标")
    lines.append("")
    lines.append("本实验固定 0428 clean benchmark 的模型和 prediction, 只改变 portfolio construction 与执行前的交易规则。目标是判断现有 10min 横截面信号能否通过降低 turnover、改善流动性过滤和拆分 long/short leg 转化为更好的 net result。")
    lines.append("")
    lines.append("## 固定输入")
    lines.append("")
    lines.append("- 模型与 checkpoint: `GruMlpRegressor / iter_140000`.")
    lines.append("- inference manifest: `/data-cache/nn/0428/date_ranges/run/inference_test/iter_140000/inference_manifest.yaml`.")
    lines.append("- test period: `2024-03-01 -> 2024-12-31`.")
    lines.append("- execution: VWAP, point-in-time universe, ST 剔除, 涨跌停方向约束.")
    lines.append("- cost: spread cost + impact cost.")
    lines.append("- 计算方式: 本报告使用 `fast_vwap_sql_proxy`, 即用 production SQL 生成目标仓位, 再用 DuckDB 计算 bar-level gross、turnover、spread 和 impact proxy。它用于批量筛选交易计划, 正式候选策略仍需再跑完整 `run_portfolio_backtest`。")
    lines.append("")
    lines.append("## 实验方案")
    lines.append("")
    lines.append("本轮实现四类实验:")
    lines.append("")
    lines.append("1. `holding`: 比较 `10/20/30/60` bars 的非重叠持有期, 检查 gross alpha decay 与 turnover 下降速度。")
    lines.append("2. `width`: 比较 top/bottom `5%/10%/20%/30%`, 检查更宽组合是否能用 capacity 换取更好的 net。")
    lines.append("3. `leg`: 拆分 long-only 与 short-only, 判断 alpha 是否主要来自单边。")
    lines.append("4. `liquidity`: 只保留高流动性 bucket, 判断低流动性股票是否是成本和 capacity 的核心约束。")
    lines.append("")
    lines.append("本轮没有实现 buffer / hysteresis, 因为当前目标仓位由 SQL 直接按单截面排序生成, buffer 需要把上一期持仓状态引入选股逻辑。该部分应在确认 holding 和 liquidity 主线后单独实现。")
    lines.append("")
    lines.append("## 实验矩阵")
    lines.append("")
    lines.append("| strategy_name | group | holding | top_frac | long | short | max_liq_bucket |")
    lines.append("| --- | --- | ---: | ---: | --- | --- | ---: |")
    for variant in list(variants):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(variant["strategy_name"]),
                    str(variant["experiment_group"]),
                    str(int(variant["holding_bars"])),
                    f"{float(variant['top_frac']):.2f}",
                    str(bool(variant["long_enabled"])),
                    str(bool(variant["short_enabled"])),
                    str(int(variant["max_liq_bucket"])),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## 结果摘要")
    lines.append("")
    lines.append(f"- 当前按 `net_10m_daily_return` 排序的最佳方案是 `{best['strategy_name']}`.")
    lines.append(f"- 该方案 gross daily return 为 `{float(best['gross_daily_return']) * 1e4:.2f}` bps.")
    lines.append(f"- daily turnover 为 `{float(best['daily_turnover']):.4f}`.")
    lines.append(f"- net daily return at 10M AUM 为 `{float(best['net_10m_daily_return']) * 1e4:.2f}` bps.")
    lines.append(f"- capacity at 10bps budget 为 `{float(best['capacity_10bps']) / 1_000_000:.2f}`M.")
    lines.append("")
    lines.append("## 完整对比表")
    lines.append("")
    display_cols = [
        "strategy_name",
        "experiment_group",
        "holding_bars",
        "top_frac",
        "long_enabled",
        "short_enabled",
        "max_liq_bucket",
        "gross_daily_return",
        "gross_sharpe",
        "daily_turnover",
        "net_10m_daily_return",
        "net_10m_sharpe",
        "net_10m_max_drawdown",
        "capacity_10bps",
    ]
    lines.append("| " + " | ".join(display_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
    for row in ordered.loc[:, display_cols].itertuples(index=False):
        values = []
        for value in list(row):
            if isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## 输出位置")
    lines.append("")
    lines.append(f"- experiment root: `{Path(experiment_root).as_posix()}`.")
    lines.append("- 每个 strategy variant 的 fast summary 位于 `variants/<strategy_name>/fast_strategy_summary.yaml`.")
    lines.append("- 统一汇总表: `trade_plan_experiment_summary.csv`.")
    lines.append("- 统一汇总 YAML: `trade_plan_experiment_summary.yaml`.")
    lines.append("- 对比图: `trade_plan_experiment_comparison.png`.")
    lines.append("")
    lines.append("## 下一步")
    lines.append("")
    lines.append("如果 holding 或 liquidity 实验能显著改善 net, 下一步应实现 buffer / hysteresis。实现方式不应继续扩大单截面 SQL, 而应在 position simulation 前增加一个 stateful target builder, 明确 entry threshold 和 hold threshold。")
    return "\n".join(lines)


def write_html_report(markdown_text: str, summary_df: pd.DataFrame, experiment_root: Path, output_path: Path) -> None:
    """Write a compact self-contained HTML report for the experiment summary."""
    # Convert the comparison dataframe into a small HTML table.
    headers = list(summary_df.columns)
    rows: list[list[str]] = []
    for row in summary_df.itertuples(index=False):
        values: list[str] = []
        for value in list(row):
            if isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        rows.append(values)

    # Render sections with the comparison plot and markdown appendix.
    plot_path = Path(experiment_root) / "trade_plan_experiment_comparison.png"
    sections = [
        render_section("Summary Table", render_table(headers, rows)),
        render_section("Comparison Figure", f'<img src="{plot_path.as_posix()}" style="max-width:100%;height:auto;" />'),
        render_section("Markdown Appendix", render_code_block(markdown_text)),
    ]
    html = build_page("Trade Plan Experiments", "Unified report for portfolio_backtest trade-plan experiments.", sections)
    Path(output_path).write_text(html, encoding="utf-8")


def run_trade_plan_experiments() -> Path:
    """Run all configured trade-plan experiments and write unified reports."""
    # Prepare output directories and the canonical base config.
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    REPO_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = build_default_portfolio_backtest_config()
    variants = build_trade_plan_variants()

    # Execute each variant through the fast VWAP-only experiment path.
    rows: list[dict[str, object]] = []
    for variant in list(variants):
        config = build_variant_config(base_config, variant, EXPERIMENT_ROOT)
        print(f"[trade-plan] run {variant['strategy_name']} -> {config.output_dir}", flush=True)
        feature_glob = resolve_feature_glob(base_config, variant, EXPERIMENT_ROOT)
        strategy = summarize_fast_variant(config, feature_glob)
        rows.append(extract_variant_summary(config, variant, strategy))

    # Persist machine-readable summaries.
    summary_df = pd.DataFrame(rows)
    summary_csv = EXPERIMENT_ROOT / "trade_plan_experiment_summary.csv"
    summary_yaml = EXPERIMENT_ROOT / "trade_plan_experiment_summary.yaml"
    summary_df.to_csv(summary_csv, index=False)
    summary_yaml.write_text(yaml.safe_dump({"rows": rows}, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Build unified human-readable reports in data-cache and repo report folders.
    plot_experiment_comparison(summary_df, EXPERIMENT_ROOT / "trade_plan_experiment_comparison.png")
    markdown_text = build_markdown_report(summary_df, EXPERIMENT_ROOT, variants)
    report_path = EXPERIMENT_ROOT / "trade_plan_experiment_report.md"
    report_path.write_text(markdown_text, encoding="utf-8")
    repo_report_path = REPO_REPORT_DIR / "trade_plan_experiment_report_0515.md"
    repo_report_path.write_text(markdown_text, encoding="utf-8")
    write_html_report(markdown_text, summary_df, EXPERIMENT_ROOT, EXPERIMENT_ROOT / "trade_plan_experiment_report.html")
    write_html_report(markdown_text, summary_df, EXPERIMENT_ROOT, REPO_REPORT_DIR / "trade_plan_experiment_report_0515.html")
    return report_path


def main() -> None:
    """Run the trade-plan experiment script."""
    # Execute the fixed experiment matrix and print the unified report path.
    report_path = run_trade_plan_experiments()
    print(Path(report_path).as_posix())


if __name__ == "__main__":
    main()
