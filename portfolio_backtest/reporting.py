"""Render portfolio backtest plots and markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from portfolio_backtest.contract import PortfolioBacktestConfig
from prediction_nn2.html_report import build_page, render_code_block, render_embedded_figure, render_section, render_table, render_value_rows


def drawdown_curve_from_returns(returns: pd.Series) -> pd.Series:
    """Build one drawdown curve series from periodic returns."""
    # Convert returns to a wealth path and derive running drawdown.
    values = returns.astype(float).to_numpy()
    wealth = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return pd.Series(drawdown)


def plot_drawdown_curve(daily: pd.DataFrame, output_path: Path, aum_list: list[float]) -> None:
    """Plot drawdown curves for gross and selected net AUMs."""
    # Build the timestamp axis from yymmdd date integers.
    timestamp = pd.to_datetime(daily["date"].astype(int).astype(str), format="%y%m%d")

    # Render the gross and net drawdown curves.
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(timestamp, drawdown_curve_from_returns(daily["gross_return"]), label="Gross", color="#2E86AB")
    for aum in list(aum_list):
        col = f"net_return_aum_{int(aum/1_000_000):d}m"
        ax.plot(timestamp, drawdown_curve_from_returns(daily[col]), label=f"Net {int(aum/1_000_000)}M")
    ax.set_title("Drawdown Curve")
    ax.set_ylabel("Drawdown (fraction)")
    ax.grid(alpha=0.25)
    ax.legend()

    # Format the x-axis as calendar dates.
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_slot_sharpe(slot_summary: pd.DataFrame, output_path: Path, title: str) -> None:
    """Plot a bar chart of per-slot annualized Sharpe."""
    # Prepare the bar inputs from the per-slot summary table.
    x = slot_summary["minute_slot"].astype(int).to_numpy()
    y = slot_summary["annualized_sharpe"].astype(float).to_numpy()

    # Render the figure.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, y, color="#3D405B")
    ax.set_title(title)
    ax.set_xlabel("Minute Slot (minute % 10)")
    ax.set_ylabel("Annualized Sharpe")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_strategy_curves(daily: pd.DataFrame, output_path: Path, aum_list: list[float]) -> None:
    """Plot cumulative gross and net curves together with turnover and costs."""
    # Build the timestamp axis from yymmdd date integers.
    timestamp = pd.to_datetime(daily["date"].astype(int).astype(str), format="%y%m%d")

    # Render the figure panels.
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(timestamp, (1.0 + daily["gross_return"].astype(float)).cumprod(), label="Gross")
    for aum in list(aum_list):
        col = f"net_return_aum_{int(aum/1_000_000):d}m"
        axes[0].plot(timestamp, (1.0 + daily[col].astype(float)).cumprod(), label=f"Net {int(aum/1_000_000)}M")
    axes[0].set_yscale("log")
    axes[0].set_title("Combined Daily Strategy Curve (Log)")
    axes[0].set_ylabel("Wealth (log scale)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(timestamp, daily["turnover"].astype(float), color="#2E86AB")
    axes[1].set_title("Mean Daily Turnover (Equal-Capital Slots)")
    axes[1].set_ylabel("Turnover (daily fraction)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(timestamp, daily["spread_cost"].astype(float) * 1e4, label="Spread (bps)", color="#E07A5F")
    axes[2].plot(timestamp, daily["impact_coeff"].astype(float) * 1e4, label="Impact coeff (scaled)", color="#3D405B")
    axes[2].set_title("Daily Cost Components (Diagnostic)")
    axes[2].set_ylabel("Spread cost / impact coeff x1e4")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    # Format the x-axis as calendar dates.
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(formatter)
    axes[2].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_capacity_sweep(strategy_summary: dict[str, Any], output_path: Path, aum_list: list[float]) -> None:
    """Plot net return and Sharpe vs AUM for the realistic strategy."""
    # Build the sweep vectors from the summary payload.
    x = np.array([float(aum) / 1_000_000.0 for aum in list(aum_list)], dtype=float)
    net_mean = np.array([float(strategy_summary[f"net_{int(aum/1_000_000):d}m"]["mean_daily_return"]) for aum in list(aum_list)])
    net_sharpe = np.array([float(strategy_summary[f"net_{int(aum/1_000_000):d}m"]["annualized_sharpe"]) for aum in list(aum_list)])

    # Render the two-panel figure.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x, net_mean * 1e4, marker="o", color="#2E86AB")
    axes[0].set_title("Net Mean Daily Return vs AUM")
    axes[0].set_ylabel("Mean daily return (bps)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, net_sharpe, marker="o", color="#E07A5F")
    axes[1].set_title("Net Sharpe vs AUM")
    axes[1].set_xlabel("AUM (million CNY)")
    axes[1].set_ylabel("Sharpe")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_report_text(config: PortfolioBacktestConfig, strategy_summary_path: Path) -> str:
    """Compose one markdown report focused on portfolio backtest outputs."""
    # Load the strategy summary payload for textual reporting.
    strategy_summary = yaml.safe_load(strategy_summary_path.read_text(encoding="utf-8"))
    baseline = strategy_summary["baseline_open"]
    realistic = strategy_summary["realistic_vwap"]

    # Compose the markdown lines.
    lines: list[str] = []
    lines.append(f"# {config.report_title}")
    lines.append("")
    lines.append("生成时间: " + pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    lines.append("")

    lines.append("## 研究范围")
    lines.append("")
    lines.append("- 本报告只覆盖 execution-aware 组合回测, 不输出 prediction vs target 的信号质量结论。")
    lines.append("- 信号质量(IC/RankIC/ICIR/rolling group IC/rank turnover)应由独立的信号评估报告负责。")
    lines.append("- 回测使用分钟级数据重建 point-in-time universe, 再与 inference prediction 做时间对齐。")
    lines.append("")

    lines.append("## 交易与限制方案(完整描述)")
    lines.append("")
    lines.append("### 时间与持有期")
    lines.append("")
    lines.append(f"- 每个交易日将分钟 bar 按 `base_minute % {int(config.slot_mod_bars)}` 分为不重叠子序列, 以避免重叠持有带来的伪高频复用。")
    lines.append(f"- Entry: 信号产生后延迟 `entry_delay_bars = {int(config.entry_delay_bars)}` 个可交易 bar 入场。")
    lines.append(f"- Holding: 持有 `holding_bars = {int(config.holding_bars)}` 个 bar, 到期后在首个可交易 bar 出场。")
    lines.append("")

    lines.append("### 选股与目标权重")
    lines.append("")
    lines.append(f"- 每个时间截面(同一 date+time)对所有可用 prediction 的股票做排序。")
    if bool(config.long_enabled) and bool(config.short_enabled):
        lines.append(f"- 取 top {float(config.top_frac) * 100:.0f}% 做多, bottom {float(config.top_frac) * 100:.0f}% 做空。")
        lines.append("- 两端等权且总资金各占 50%, 组合目标为 market-neutral(忽略融资与借券约束)。")
    elif bool(config.long_enabled):
        lines.append(f"- 只取 top {float(config.top_frac) * 100:.0f}% 做多, gross exposure 目标为 100%。")
    else:
        lines.append(f"- 只取 bottom {float(config.top_frac) * 100:.0f}% 做空, gross exposure 目标为 100%。")
    lines.append(f"- 流动性过滤: 保留 `liq_bucket <= {int(config.max_liq_bucket)}` 的候选标的。")
    lines.append("")

    lines.append("### Universe 与可交易性")
    lines.append("")
    lines.append("- Universe 以分钟级当时可见信息(point-in-time)构建: 价格/成交量/成交额为正、ST 剔除。")
    lines.append("- 一字涨跌停 bar 视为不可交易, 不用于入场/出场撮合。")
    lines.append("- 本回测不建模 borrow/shortable 约束, 也不做市值分桶等额外限制。")
    lines.append("")

    lines.append("### 涨跌停执行约束(简化且保守)")
    lines.append("")
    lines.append("- 先按日线判断是否全天封死涨停/跌停(全日每个分钟 bar 都在涨停价/跌停价)。")
    lines.append("- 全天封死涨停: 当天所有买入委托失败(无法买入)。")
    lines.append("- 全天封死跌停: 当天所有卖出委托失败(无法卖出)。")
    lines.append("- 盘中开板(非一字): 在执行 bar 上若触及涨停价则禁止买入; 若触及跌停价则禁止卖出。")
    lines.append("- 该约束按方向生效: 开多/平空属于买入, 开空/平多属于卖出。")
    lines.append("- 暂不建模封单量与部分成交, 触板方向直接视为不可成交。")
    lines.append("")

    lines.append("### 成交、持仓与失败处理")
    lines.append("")
    lines.append("- 先生成目标权重, 再在执行层判定是否可成交(fillable)。")
    lines.append("- 某标的若入场或出场在该方向上不可成交, 则该标的当次执行权重记为 0, 等价于回到现金, 不反向污染排序选股。")
    lines.append("- 持仓在 bar 间滚动更新为自融资权重(按组合当根收益归一化), 用于后续换手与成本计算。")
    lines.append("")

    lines.append("## 成本模型")
    lines.append("")
    lines.append("- spread 成本: 依据流动性分桶(成交额)给定 bps, 对换手收取半点差。")
    lines.append(f"- ADV: 使用过去 {int(config.adv_lookback_days)} 个交易日的日成交额均值作为 ex-ante 成交额代理。")
    lines.append(f"- intraday sigma: 使用过去 {int(config.sigma_lookback_bars)} 个分钟 bar 的 1m return rolling std 作为波动代理。")
    lines.append("- impact 成本: 对换手收取 `impact_coeff * sqrt(AUM)` 形式的冲击, 其中 impact_coeff 由 sigma 与 ADV 决定。")
    lines.append("")

    lines.append("## 结果摘要")
    lines.append("")
    lines.append("### Baseline(Open, no cost)")
    lines.append("")
    lines.append(f"- Mean fill ratio: {baseline['execution']['mean_fill_ratio'] * 100:.2f}%.")
    lines.append(f"- Mean executed gross exposure: {baseline['execution']['mean_executed_gross_exposure']:.4f}.")
    lines.append(f"- Mean cash buffer: {baseline['execution']['mean_cash_buffer']:.4f}.")
    lines.append("")

    lines.append("### Realistic(VWAP, with costs)")
    lines.append("")
    lines.append(f"- Mean fill ratio: {realistic['execution']['mean_fill_ratio'] * 100:.2f}%.")
    lines.append(f"- Mean executed gross exposure: {realistic['execution']['mean_executed_gross_exposure']:.4f}.")
    lines.append(f"- Mean cash buffer: {realistic['execution']['mean_cash_buffer']:.4f}.")
    lines.append("")

    lines.append("### Capacity (impact budget)")
    lines.append("")
    for budget_bps in list(config.impact_budget_bps_list):
        key = f"capacity_at_{int(budget_bps)}bps"
        lines.append(f"- {int(budget_bps)}bps: AUM ~= {realistic['capacity'][key] / 1_000_000:.2f}M.")
    return "\n".join(lines)


def build_self_contained_html_report(config: PortfolioBacktestConfig, strategy_summary_path: Path) -> str:
    """Compose one self-contained HTML report for the portfolio backtest outputs."""
    # Load the strategy summary and markdown appendix once.
    strategy_summary = yaml.safe_load(strategy_summary_path.read_text(encoding="utf-8"))
    report_md_path = Path(config.output_dir) / "research_report.md"
    report_md = report_md_path.read_text(encoding="utf-8")
    baseline = dict(strategy_summary["baseline_open"])
    realistic = dict(strategy_summary["realistic_vwap"])
    cost_model = dict(strategy_summary["cost_model"])

    # Build the baseline overview rows.
    baseline_rows = [
        ("mean_daily_return", f"{float(baseline['gross']['mean_daily_return']) * 100:.4f}%"),
        ("annualized_return", f"{float(baseline['gross']['annualized_return']) * 100:.2f}%"),
        ("annualized_sharpe", f"{float(baseline['gross']['annualized_sharpe']):.3f}"),
        ("max_drawdown", f"{float(baseline['gross']['max_drawdown']) * 100:.2f}%"),
        ("mean_fill_ratio", f"{float(baseline['execution']['mean_fill_ratio']) * 100:.2f}%"),
        ("mean_cash_buffer", f"{float(baseline['execution']['mean_cash_buffer']):.4f}"),
    ]

    # Build the realistic net-by-aum table.
    realistic_headers = ["AUM", "mean_daily_return", "annualized_return", "annualized_sharpe", "max_drawdown"]
    realistic_rows: list[list[str]] = []
    for aum in list(config.aum_list):
        key = f"net_{int(aum / 1_000_000):d}m"
        block = dict(realistic[key])
        realistic_rows.append(
            [
                f"{int(aum / 1_000_000)}M",
                f"{float(block['mean_daily_return']) * 100:.4f}%",
                f"{float(block['annualized_return']) * 100:.2f}%",
                f"{float(block['annualized_sharpe']):.3f}",
                f"{float(block['max_drawdown']) * 100:.2f}%",
            ]
        )

    # Build the capacity table.
    capacity_headers = ["impact_budget", "aum"]
    capacity_rows: list[list[str]] = []
    for budget_bps in list(config.impact_budget_bps_list):
        key = f"capacity_at_{int(budget_bps)}bps"
        capacity_rows.append([f"{int(budget_bps)}bps", f"{float(realistic['capacity'][key]) / 1_000_000:.2f}M"])

    # Build the self-contained sections.
    sections = [
        render_section(
            "Overview",
            render_value_rows(
                [
                    ("report_title", str(config.report_title)),
                    ("inference_manifest", Path(config.inference_manifest_path).as_posix()),
                    ("output_dir", Path(config.output_dir).as_posix()),
                    ("entry_delay_bars", str(int(config.entry_delay_bars))),
                    ("holding_bars", str(int(config.holding_bars))),
                    ("slot_mod_bars", str(int(config.slot_mod_bars))),
                    ("top_frac", f"{float(config.top_frac):.2f}"),
                    ("long_enabled", str(bool(config.long_enabled))),
                    ("short_enabled", str(bool(config.short_enabled))),
                    ("max_liq_bucket", str(int(config.max_liq_bucket))),
                ]
            ),
        ),
        render_section(
            "Cost Model",
            render_value_rows(
                [
                    ("spread_bps_high", f"{float(cost_model['spread_bps_high']):.2f}"),
                    ("spread_bps_mid", f"{float(cost_model['spread_bps_mid']):.2f}"),
                    ("spread_bps_low", f"{float(cost_model['spread_bps_low']):.2f}"),
                    ("impact_eta", f"{float(cost_model['impact_eta']):.2f}"),
                    ("adv_lookback_days", str(int(cost_model['adv_lookback_days']))),
                    ("sigma_lookback_bars", str(int(cost_model['sigma_lookback_bars']))),
                ]
            )
            + render_code_block(yaml.safe_dump(cost_model, sort_keys=False, allow_unicode=True)),
        ),
        render_section("Baseline Open Summary", render_value_rows(baseline_rows)),
        render_section("Realistic VWAP Net Summary", render_table(realistic_headers, realistic_rows)),
        render_section("Capacity", render_table(capacity_headers, capacity_rows)),
        render_section(
            "Figures",
            render_embedded_figure("Baseline Open Strategy", Path(config.output_dir) / "baseline_open_strategy.png", "Baseline open gross strategy curve and diagnostics.")
            + render_embedded_figure("Realistic VWAP Strategy", Path(config.output_dir) / "strategy_curves.png", "Gross and net wealth curves under multiple AUM assumptions.")
            + render_embedded_figure("Drawdown Curve", Path(config.output_dir) / "drawdown_curve.png", "Gross and net drawdown curves.")
            + render_embedded_figure("Slot Sharpe", Path(config.output_dir) / "slot_sharpe.png", "Per-slot annualized Sharpe for the realistic VWAP strategy.")
            + render_embedded_figure("Capacity Sweep", Path(config.output_dir) / "capacity_sweep.png", "Capacity diagnostics under multiple AUM assumptions."),
        ),
        render_section("Markdown Appendix", render_code_block(report_md)),
    ]
    return build_page("Portfolio Backtest Report", "Self-contained HTML report generated from portfolio_backtest outputs.", sections)
