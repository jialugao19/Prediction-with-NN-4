"""Render benchmark reports from evaluation artifacts."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

from prediction_nn2.html_report import build_page, render_block_title, render_code_block, render_embedded_figure, render_html_table, render_section, render_subsection, render_table, render_value_rows
from portfolio_backtest.benchmark.evaluation_builder import BENCHMARK_ID, BENCHMARK_ROOT, BEST_CHECKPOINT_ITER, REPORT_FIGURE_DIR, REPO_BENCHMARK_REPORT_DIR, REPO_ROOT, SOURCE_RUN_ROOT, cached_artifacts_are_current, copy_file, multi_file_stat_cache_payload, read_yaml, write_artifact_cache, write_yaml
from portfolio_backtest.benchmark.report_fields import FIELD_DEFINITIONS, display_metric_name, format_report_cell, format_report_value


matplotlib.use("Agg")
import matplotlib.pyplot as plt

def render_csv_table(path: Path, n_rows: int) -> str:
    """Render a CSV preview table."""
    # Read a compact preview.
    df = pd.read_csv(path).head(int(n_rows))

    # Format cells as strings.
    rows = [[format_report_cell(str(col), value) for col, value in zip(list(df.columns), list(row), strict=True)] for row in df.astype(object).to_numpy().tolist()]
    return '<div class="table-wrap">' + render_table([str(col) for col in df.columns], rows) + "</div>"


def render_rows_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render an in-memory row table."""
    # Format every cell through the same report value formatter used by CSV tables.
    formatted_rows = [[format_report_cell(str(col), value) for col, value in zip(list(headers), list(row), strict=True)] for row in list(rows)]
    return '<div class="table-wrap">' + render_table(list(headers), formatted_rows) + "</div>"


def report_badge(label: str, status: str) -> str:
    """Render one status badge."""
    # Choose a small semantic class for visual scanning.
    class_name = {
        "good": "badge-good",
        "watch": "badge-watch",
        "bad": "badge-bad",
        "neutral": "badge-neutral",
    }[str(status)]
    return f'<span class="badge {class_name}">{html.escape(str(label))}</span>'


def render_summary_cards(cards: list[tuple[str, str, str]]) -> str:
    """Render compact summary cards for the report first screen."""
    # Render each card with escaped label, value and note.
    items: list[str] = []
    for key, value, note in list(cards):
        items.append(
            f"""
            <div class="summary-card">
              <div class="summary-card-key">{html.escape(str(key))}</div>
              <div class="summary-card-value">{html.escape(str(value))}</div>
              <div class="summary-card-note">{html.escape(str(note))}</div>
            </div>
            """
        )
    return '<div class="summary-grid">' + "\n".join(items) + "</div>"


def render_takeaways(items: list[str]) -> str:
    """Render concise takeaway bullets."""
    # Escape each takeaway while keeping the list visually distinct.
    rows = [f"<li>{html.escape(str(item))}</li>" for item in list(items)]
    return '<ul class="takeaways">' + "\n".join(rows) + "</ul>"


def render_metric_review_table(rows: list[tuple[str, Any, str, str]]) -> str:
    """Render a status-aware metric review table."""
    # Convert metric rows into trusted HTML cells.
    table_rows: list[list[str]] = []
    for metric, value, interpretation, status in list(rows):
        table_rows.append(
            [
                html.escape(str(metric)),
                html.escape(format_report_value(str(metric), value)),
                html.escape(str(interpretation)),
                report_badge(str(status), metric_status_class(str(status))),
            ]
        )
    return '<div class="table-wrap">' + render_html_table(["metric", "value", "interpretation", "status"], table_rows) + "</div>"


def metric_status_class(status: str) -> str:
    """Map report status text to badge class names."""
    # Keep status wording separate from CSS class names.
    if str(status) in {"passed", "selected", "positive", "stable", "baseline"}:
        return "good"
    if str(status) in {"watch", "mixed", "reference"}:
        return "watch"
    if str(status) in {"negative", "failed", "bad"}:
        return "bad"
    return "neutral"


def render_details_table(title: str, body: str) -> str:
    """Render an expandable detail block."""
    # Keep long appendix tables available without dominating the first read.
    return f'<details class="details"><summary>{html.escape(str(title))}</summary>{body}</details>'


def render_details_code(title: str, text: str) -> str:
    """Render an expandable raw text block."""
    # Reuse the report details styling for raw YAML payloads.
    return f'<details class="details"><summary>{html.escape(str(title))}</summary>{render_code_block(str(text))}</details>'


def render_field_notes(raw_fields: list[str], section_note: str) -> str:
    """Render visible Chinese explanations for selected fields."""
    # Start with the section-level reading guide.
    notes = [f'<div class="field-note">{html.escape(str(section_note))}</div>']

    # Render one detailed note per known raw field.
    for raw_field in list(raw_fields):
        field = FIELD_DEFINITIONS.get(str(raw_field))
        if field is None:
            notes.append(
                "<div class=\"field-note\">"
                f"<strong>{html.escape(str(raw_field))}</strong>: "
                "当前 raw field 暂未登记详细解释, 需要在后续版本补充字段定义。"
                "</div>"
            )
            continue
        notes.append(
            "<div class=\"field-note\">"
            f"<strong>{html.escape(str(field['display_name']))}</strong>: "
            f"{html.escape(str(field['explanation']))} "
            f"单位: {html.escape(str(field['unit']))}; "
            f"读数方向: {html.escape(str(field['direction']))}."
            "</div>"
        )
    return '<div class="field-notes"><div class="field-notes-title">字段解释</div>' + "\n".join(notes) + "</div>"


def field_name_appendix() -> str:
    """Render the raw-field appendix table."""
    # Build the final mapping from Chinese display names back to raw field names.
    rows = []
    for raw_name, field in FIELD_DEFINITIONS.items():
        rows.append(
            [
                str(field["display_name"]),
                str(raw_name),
                str(field["unit"]),
                str(field["direction"]),
                str(field["explanation"]),
            ]
        )
    return '<div class="table-wrap">' + render_table(["display_name", "raw_field_name", "unit", "direction", "explanation"], rows) + "</div>"


def ic_summary_table(train_ic: dict[str, Any], test_ic: dict[str, Any]) -> str:
    """Render train and test IC summaries as columns."""
    # Build one row per split from the Pearson IC YAML payloads.
    pooled = source_pooled_ic_values()
    rows = []
    for split, payload in [("train", train_ic), ("test", test_ic)]:
        rows.append(
            [
                split,
                format_report_value("count", payload["count"]),
                format_report_value("mean_ic", payload["mean"]),
                format_report_value("std_ic", payload["std"]),
                format_report_value("t_stat", payload["t_stat"]),
                format_report_value("positive_ratio", payload["positive_ratio"]),
                format_report_value("icir", payload["icir"]),
                format_report_value("pooled_ic", pooled[f"{split}_pooled_ic"]),
                format_report_value("pooled_rank_ic", float("nan")),
            ]
        )
    return '<div class="table-wrap">' + render_table(["split", "count", "mean_ic", "std_ic", "t_stat", "positive_ratio", "icir", "pooled_ic", "pooled_rank_ic"], rows) + "</div>"


def source_pooled_ic_values() -> dict[str, float]:
    """Read pooled Pearson IC values from the frozen source report."""
    # Extract the source training report text that already records train/test pooled IC.
    text = (SOURCE_RUN_ROOT / "train_report.html").read_text(encoding="utf-8")

    # Pull exact key/value pairs from the report HTML.
    values: dict[str, float] = {}
    for key in ["pooled_ic_train", "pooled_ic_test"]:
        pattern = rf'<div class="kv-key">{re.escape(key)}</div>\s*<div class="kv-value">([^<]+)</div>'
        match = re.search(pattern, text)
        values[key] = float(match.group(1).replace(",", ""))
    return {"train_pooled_ic": values["pooled_ic_train"], "test_pooled_ic": values["pooled_ic_test"]}


def prediction_target_scale_rows() -> list[tuple[str, float]]:
    """Build prediction/target scale check rows from normalization metrics."""
    # Convert the normalization metric rows into a lookup table.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv")
    metrics = {str(row["metric_name"]): float(row["value"]) for _, row in df[df["metric_name"].astype(str).str.startswith("val/dist/")].iterrows()}

    # Map TensorBoard scalar names to the report raw field names from outline.md.
    return [
        ("prediction_mean", metrics["val/dist/mean/pred"]),
        ("prediction_std", metrics["val/dist/std/pred"]),
        ("target_mean", metrics["val/dist/mean/target"]),
        ("target_std", metrics["val/dist/std/target"]),
        ("pred_std_over_target_std", metrics["val/dist/pred_std_over_target_std"]),
        ("prediction_p01", metrics["val/dist/p01/pred"]),
        ("prediction_p50", metrics["val/dist/p50/pred"]),
        ("prediction_p99", metrics["val/dist/p99/pred"]),
        ("target_p01", metrics["val/dist/p01/target"]),
        ("target_p50", metrics["val/dist/p50/target"]),
        ("target_p99", metrics["val/dist/p99/target"]),
    ]


def prediction_target_scale_table() -> str:
    """Render the prediction/target scale check table."""
    # Build a two-column raw field/value table.
    rows = [[field, value] for field, value in prediction_target_scale_rows()]
    return render_rows_table(["raw field name", "value"], rows)


def checkpoint_summary_table() -> str:
    """Render the compact checkpoint selector table."""
    # Read only the columns needed to explain the selection decision.
    df = pd.read_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv")
    columns = ["iter", "selected", "val/objective/mse", "val/quality/global_ic", "val/quality/rank_ic", "val/dist/pred_std_over_target_std", "metric_rank", "decision"]
    compact = df.loc[:, columns].sort_values(["selected", "metric_rank"], ascending=[False, True])

    # Format selected values for scan-friendly display.
    rows = []
    for _, row in compact.iterrows():
        rows.append([format_report_value(col, row[col]) for col in columns])
    return '<div class="table-wrap">' + render_table(columns, rows) + "</div>"


def experiment_comparison_placeholder(summary: dict[str, Any], test_ic: dict[str, Any]) -> str:
    """Render the current baseline row plus future experiment columns."""
    # Use the baseline as the first comparable experiment row.
    rows = [
        [
            BENCHMARK_ID,
            "2026-05-19",
            "GruMlpRegressor",
            "14",
            "log_close[t+10] - log_close[t+1]",
            str(BEST_CHECKPOINT_ITER),
            format_report_value("test_ic", test_ic["mean"]),
            "signal proxy",
            "baseline reference",
        ],
        ["future_experiment", "", "", "", "", "", "", "", ""],
    ]
    headers = ["experiment", "date", "model", "features", "label", "best_iter", "test_ic", "cost_model", "verdict"]
    return '<div class="table-wrap">' + render_table(headers, rows) + "</div>"


def comparison_delta_table(comparison: dict[str, Any]) -> str:
    """Render current-vs-parent comparison with explicit delta columns."""
    # Skip metrics that are already emphasized elsewhere in the report.
    excluded_metrics = {"q95_q80_net_bps_per_turnover", "top_minus_bottom_bps", "high_liq_top_decile_net_proxy_bps"}

    # Convert the nested comparison payload into a compact baseline comparison.
    primary = dict(comparison["primary_metric"])
    rows = []
    if str(primary["name"]) not in excluded_metrics:
        rows.append([
            str(primary["name"]),
            format_report_value(str(primary["name"]), primary["baseline"]),
            format_report_value(str(primary["name"]), primary["experiment"]),
            format_report_value(str(primary["name"]), primary["absolute_change"]),
            "same baseline",
        ])
    for key, value in dict(comparison["secondary_metrics"]).items():
        if str(key) in excluded_metrics:
            continue
        rows.append([str(key), format_report_value(str(key), value), format_report_value(str(key), value), "0.0000", "reference"])
    return '<div class="table-wrap">' + render_table(["metric", "baseline", "current", "delta", "note"], rows) + "</div>"


def configure_report_plot_style() -> None:
    """Configure matplotlib for compact report figures."""
    # Prefer common CJK fonts when available so Chinese labels render correctly.
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def signed_colors(values: list[float]) -> list[str]:
    """Return red/green colors by value sign."""
    # Use one stable palette across all report figures.
    return ["#c53030" if float(value) >= 0 else "#2f855a" for value in list(values)]


def ascii_plot_label(raw_name: str) -> str:
    """Return one ASCII-safe plot label."""
    # Avoid CJK glyph warnings on machines without Chinese fonts.
    labels = {
        "top_decile_return_bps": "top decile return",
        "bottom_decile_return_bps": "bottom decile return",
        "top_minus_bottom_bps": "top-bottom spread",
        "high_liq_top_decile_net_proxy_bps": "high-liq top net proxy",
        "q95_q80_gross_daily_return_bps": "q95/q80 gross daily",
        "q95_q80_net_daily_return_bps": "q95/q80 net daily",
        "q95_q80_net_bps_per_turnover": "q95/q80 net bps/turnover",
    }
    return labels.get(str(raw_name), str(raw_name))


def save_headline_metrics_figure(summary: dict[str, Any], out_path: Path) -> Path:
    """Plot the main headline metrics as a signed bar chart."""
    # Select the metrics that explain the baseline in one glance.
    metrics = dict(summary["headline_metrics"])
    raw_fields = [
        "top_decile_return_bps",
        "bottom_decile_return_bps",
        "top_minus_bottom_bps",
        "high_liq_top_decile_net_proxy_bps",
        "q95_q80_gross_daily_return_bps",
        "q95_q80_net_daily_return_bps",
        "q95_q80_net_bps_per_turnover",
    ]
    values = [float(metrics[field]) for field in raw_fields]
    labels = [ascii_plot_label(field) for field in raw_fields]

    # Draw a horizontal bar chart with a zero line for cost-aware interpretation.
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.barh(labels, values, color=signed_colors(values))
    ax.axvline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Headline metric sign check")
    ax.set_xlabel("bps or bps/turnover")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_ic_summary_figure(train_ic: dict[str, Any], test_ic: dict[str, Any], out_path: Path) -> Path:
    """Plot train/test IC mean and ICIR."""
    # Prepare side-by-side IC and stability values.
    splits = ["train", "test"]
    mean_values = [float(train_ic["mean"]), float(test_ic["mean"])]
    icir_values = [float(train_ic["icir"]), float(test_ic["icir"])]

    # Draw two small panels so level and stability are not mixed on one axis.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(splits, mean_values, color=["#2b6cb0", "#2f855a"])
    axes[0].set_title("Daily IC mean")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(splits, icir_values, color=["#2b6cb0", "#2f855a"])
    axes[1].set_title("Daily ICIR")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_checkpoint_selection_figure(out_path: Path) -> Path:
    """Plot checkpoint selection metrics by iteration."""
    # Load checkpoint selector output.
    df = pd.read_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv").sort_values("iter")
    selected = df[df["selected"].astype(bool)].iloc[0]

    # Plot MSE and IC metrics on separate axes because scales differ.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(df["iter"], df["val/objective/mse"], marker="o", color="#2b6cb0")
    axes[0].axvline(float(selected["iter"]), color="#c53030", linestyle="--", linewidth=1.0)
    axes[0].set_title("Validation MSE by checkpoint")
    axes[0].set_xlabel("iter")
    axes[0].grid(alpha=0.25)
    axes[1].plot(df["iter"], df["val/quality/global_ic"], marker="o", label="global IC", color="#2f855a")
    axes[1].plot(df["iter"], df["val/quality/rank_ic"], marker="o", label="rank IC", color="#805ad5")
    axes[1].axvline(float(selected["iter"]), color="#c53030", linestyle="--", linewidth=1.0)
    axes[1].set_title("Validation IC by checkpoint")
    axes[1].set_xlabel("iter")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_signal_quality_figure(out_path: Path) -> Path:
    """Plot signal bucket return and hit rate."""
    # Load signal bucket metrics.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv")

    # Plot return bars and hit-rate line to show monotonicity and direction together.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(df["signal_bucket"].astype(str), df["mean_return_bps"], color=signed_colors(df["mean_return_bps"].astype(float).tolist()))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Signal bucket return and hit rate")
    ax.set_xlabel("signal bucket")
    ax.set_ylabel("mean return bps")
    ax.grid(axis="y", alpha=0.25)
    twin = ax.twinx()
    twin.plot(df["signal_bucket"].astype(str), df["hit_rate"], marker="o", color="#2b6cb0")
    twin.set_ylabel("hit rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_trading_cost_figure(out_path: Path) -> Path:
    """Plot gross, net and cost drag for the q95/q80 rule."""
    # Select the current headline trading rule.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv")
    row = df.query("strategy_name == 'q95_q80'").iloc[0]
    values = [
        float(row["gross_daily_return_bps"]),
        -float(row["spread_cost_bps"]),
        -float(row["fee_cost_bps"]),
        float(row["net_daily_return_bps"]),
    ]
    labels = ["gross daily return", "spread cost", "fee cost", "net daily return"]

    # Draw the cost bridge as signed bars.
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values, color=signed_colors(values))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("q95/q80 trading cost bridge")
    ax.set_ylabel("bps")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_time_bucket_figure(out_path: Path) -> Path:
    """Plot time bucket net bps per turnover."""
    # Load time bucket metrics ordered by intraday bucket.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv")
    values = df["net_bps_per_turnover"].astype(float).tolist()

    # Draw positive and negative buckets in different colors.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(df["time_bucket"].astype(str), values, color=signed_colors(values))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Time bucket net bps per turnover")
    ax.set_xlabel("time bucket")
    ax.set_ylabel("net bps/turnover")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_liquidity_top_bucket_figure(out_path: Path) -> Path:
    """Plot top signal bucket net proxy by liquidity bucket."""
    # Use signal bucket 10 as the strongest long-side bucket.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv")
    top = df[df["signal_bucket"].astype(int) == 10].sort_values("liq_bucket")
    values = top["entry_net_proxy_bps"].astype(float).tolist()

    # Draw liquidity sensitivity of the top signal bucket.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(top["liq_bucket"].astype(str), values, color=signed_colors(values))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Top signal bucket by liquidity")
    ax.set_xlabel("liquidity bucket")
    ax.set_ylabel("entry net proxy bps")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_prediction_target_distribution_figure(out_path: Path) -> Path:
    """Plot prediction and target distribution checkpoints from normalization metrics."""
    # Extract validation distribution quantiles from the normalization diagnostics.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv")
    mapping = dict(zip(df["metric_name"].astype(str), df["value"], strict=False))
    quantiles = ["p01", "p50", "p99"]
    pred_values = [float(mapping[f"val/dist/{quantile}/pred"]) for quantile in quantiles]
    target_values = [float(mapping[f"val/dist/{quantile}/target"]) for quantile in quantiles]

    # Plot prediction and target quantiles on the same axis.
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(quantiles))
    width = 0.36
    ax.bar(positions - width / 2, pred_values, width=width, label="prediction", color="#2b6cb0")
    ax.bar(positions + width / 2, target_values, width=width, label="target", color="#805ad5")
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(quantiles)
    ax.set_title("Prediction vs target distribution")
    ax.set_ylabel("normalized value")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_prediction_target_scale_by_checkpoint_figure(out_path: Path) -> Path:
    """Plot prediction-to-target scale ratio by checkpoint."""
    # Read checkpoint selector diagnostics.
    df = pd.read_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv")
    selected = df[df["selected"].astype(bool)].iloc[0]

    # Plot the scale ratio across checkpoints.
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["iter"], df["val/dist/pred_std_over_target_std"], marker="o", color="#2b6cb0")
    ax.axvline(float(selected["iter"]), color="#c53030", linestyle="--", linewidth=1.0)
    ax.axhline(1.0, color="#334155", linestyle=":", linewidth=1.0)
    ax.set_title("Prediction/target scale by checkpoint")
    ax.set_xlabel("iter")
    ax.set_ylabel("pred std / target std")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_bootstrap_ci_figure(out_path: Path) -> Path:
    """Plot bootstrap confidence intervals."""
    # Load bootstrap interval metrics.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv")
    labels = [ascii_plot_label(str(metric)) for metric in df["metric"].astype(str)]
    means = df["mean"].astype(float).to_numpy()
    lower = means - df["ci_025"].astype(float).to_numpy()
    upper = df["ci_975"].astype(float).to_numpy() - means

    # Draw horizontal error bars so long metric names remain readable.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.errorbar(means, labels, xerr=[lower, upper], fmt="o", color="#2b6cb0", ecolor="#718096", capsize=4)
    ax.axvline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Bootstrap confidence intervals")
    ax.set_xlabel("metric value")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_month_stability_figure(out_path: Path) -> Path:
    """Plot monthly signal spread and trading efficiency."""
    # Load month stability metrics in chronological order.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv")
    labels = df["month"].astype(str).tolist()

    # Draw signal spread bars and net efficiency line together.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    values = df["top_minus_bottom_bps"].astype(float).tolist()
    ax.bar(labels, values, color=signed_colors(values))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Monthly signal spread and net efficiency")
    ax.set_xlabel("month")
    ax.set_ylabel("top-minus-bottom bps")
    ax.tick_params(axis="x", rotation=45)
    twin = ax.twinx()
    twin.plot(labels, df["q95_q80_net_bps_per_turnover"].astype(float), marker="o", color="#2b6cb0")
    twin.set_ylabel("q95/q80 net bps/turnover")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_regime_stability_figure(out_path: Path) -> Path:
    """Plot gross and net daily bps by regime."""
    # Load regime stability metrics.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv")
    labels = df["regime"].astype(str).tolist()
    positions = np.arange(len(labels))
    width = 0.36

    # Compare gross and net daily bps by regime.
    fig, ax = plt.subplots(figsize=(8.5, 4))
    gross = df["gross_daily_bps"].astype(float).to_numpy()
    net = df["net_daily_bps"].astype(float).to_numpy()
    ax.bar(positions - width / 2, gross, width=width, label="gross", color=signed_colors(gross.tolist()))
    ax.bar(positions + width / 2, net, width=width, label="net", color=signed_colors(net.tolist()))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Regime gross and net daily bps")
    ax.set_ylabel("bps")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_volatility_bucket_stability_figure(out_path: Path) -> Path:
    """Plot IC by volatility bucket."""
    # Load volatility bucket stability metrics.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv")
    labels = df["volatility_bucket"].astype(str).tolist()

    # Compare Pearson IC and rank IC across volatility buckets.
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(labels, df["mean_ic"].astype(float), marker="o", label="mean_ic", color="#2b6cb0")
    ax.plot(labels, df["mean_rank_ic"].astype(float), marker="o", label="mean_rank_ic", color="#805ad5")
    ax.set_title("IC by volatility bucket")
    ax.set_xlabel("volatility bucket")
    ax.set_ylabel("IC")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def figure_is_current(out_path: Path, input_paths: list[Path]) -> bool:
    """Return whether one figure is newer than all source tables."""
    # Require the figure and every input table before comparing mtimes.
    if not Path(out_path).exists():
        return False
    for input_path in list(input_paths):
        if not Path(input_path).exists():
            return False

    # Reuse the figure when no source table is newer.
    output_mtime = Path(out_path).stat().st_mtime_ns
    return all(output_mtime >= Path(input_path).stat().st_mtime_ns for input_path in list(input_paths))


def build_or_reuse_figure(name: str, out_path: Path, input_paths: list[Path], builders: dict[str, Any]) -> Path:
    """Build one report figure only when its source tables changed."""
    # Skip matplotlib work when the figure already reflects its small source artifacts.
    if figure_is_current(Path(out_path), list(input_paths)):
        return Path(out_path)

    # Dispatch to the explicit figure builder.
    return Path(builders[str(name)]())


def build_report_figures(summary: dict[str, Any], train_ic: dict[str, Any], test_ic: dict[str, Any]) -> dict[str, Path]:
    """Build all report-specific figures."""
    # Ensure the target figure directory exists.
    configure_report_plot_style()
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Register figure builders and their compact source artifacts.
    builders: dict[str, Any] = {
        "ic_summary": lambda: save_ic_summary_figure(train_ic, test_ic, REPORT_FIGURE_DIR / "ic_summary_bar.png"),
        "checkpoint": lambda: save_checkpoint_selection_figure(REPORT_FIGURE_DIR / "checkpoint_selection_metrics.png"),
        "signal_quality": lambda: save_signal_quality_figure(REPORT_FIGURE_DIR / "signal_bucket_return_hit_rate.png"),
        "trading_cost": lambda: save_trading_cost_figure(REPORT_FIGURE_DIR / "trading_cost_bridge.png"),
        "time_bucket": lambda: save_time_bucket_figure(REPORT_FIGURE_DIR / "time_bucket_net_bps_per_turnover.png"),
        "liquidity_top": lambda: save_liquidity_top_bucket_figure(REPORT_FIGURE_DIR / "liquidity_top_signal_bucket.png"),
        "pred_target_distribution": lambda: save_prediction_target_distribution_figure(REPORT_FIGURE_DIR / "prediction_target_distribution.png"),
        "pred_target_scale": lambda: save_prediction_target_scale_by_checkpoint_figure(REPORT_FIGURE_DIR / "prediction_target_scale_by_checkpoint.png"),
        "bootstrap_ci": lambda: save_bootstrap_ci_figure(REPORT_FIGURE_DIR / "bootstrap_confidence_intervals.png"),
        "month_stability": lambda: save_month_stability_figure(REPORT_FIGURE_DIR / "month_stability_metrics.png"),
        "regime_stability": lambda: save_regime_stability_figure(REPORT_FIGURE_DIR / "regime_stability_metrics.png"),
        "volatility_stability": lambda: save_volatility_bucket_stability_figure(REPORT_FIGURE_DIR / "volatility_bucket_stability_metrics.png"),
    }
    figure_specs = {
        "ic_summary": (REPORT_FIGURE_DIR / "ic_summary_bar.png", [BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_train.yaml", BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_test.yaml"]),
        "checkpoint": (REPORT_FIGURE_DIR / "checkpoint_selection_metrics.png", [BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv"]),
        "signal_quality": (REPORT_FIGURE_DIR / "signal_bucket_return_hit_rate.png", [BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv"]),
        "trading_cost": (REPORT_FIGURE_DIR / "trading_cost_bridge.png", [BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv"]),
        "time_bucket": (REPORT_FIGURE_DIR / "time_bucket_net_bps_per_turnover.png", [BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv"]),
        "liquidity_top": (REPORT_FIGURE_DIR / "liquidity_top_signal_bucket.png", [BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv"]),
        "pred_target_distribution": (REPORT_FIGURE_DIR / "prediction_target_distribution.png", [BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv"]),
        "pred_target_scale": (REPORT_FIGURE_DIR / "prediction_target_scale_by_checkpoint.png", [BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv"]),
        "bootstrap_ci": (REPORT_FIGURE_DIR / "bootstrap_confidence_intervals.png", [BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv"]),
        "month_stability": (REPORT_FIGURE_DIR / "month_stability_metrics.png", [BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv"]),
        "regime_stability": (REPORT_FIGURE_DIR / "regime_stability_metrics.png", [BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv"]),
        "volatility_stability": (REPORT_FIGURE_DIR / "volatility_bucket_stability_metrics.png", [BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv"]),
    }

    # Render only stale figures from the existing benchmark tables.
    figures = {
        name: build_or_reuse_figure(name, out_path, input_paths, builders)
        for name, (out_path, input_paths) in dict(figure_specs).items()
    }
    return figures


def basic_info_rows() -> dict[str, list[tuple[str, str]]]:
    """Build basic data, model, and method rows for reports."""
    # Reuse the compact report data cache to avoid reparsing the large npz meta YAML.
    cache_output = BENCHMARK_ROOT / "reports" / "cache" / "basic_info_rows.yaml"
    cache_path = BENCHMARK_ROOT / "reports" / "cache" / "basic_info_rows.cache.yaml"
    source_files = [
        BENCHMARK_ROOT / "data" / "npz_meta.yaml",
        BENCHMARK_ROOT / "data" / "normalization_contract.yaml",
        BENCHMARK_ROOT / "model" / "effective_model_summary.yaml",
        BENCHMARK_ROOT / "train" / "train_config.yaml",
        BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml",
    ]
    expected_cache = multi_file_stat_cache_payload("basic_info_rows", 1, source_files, [cache_output])
    if cached_artifacts_are_current(cache_path, expected_cache, [cache_output]):
        cached = read_yaml(cache_output)
        return {str(key): [tuple(item) for item in list(value)] for key, value in dict(cached).items()}

    # Load frozen contracts.
    meta = read_yaml(BENCHMARK_ROOT / "data" / "npz_meta.yaml")
    norm = read_yaml(BENCHMARK_ROOT / "data" / "normalization_contract.yaml")
    model = read_yaml(BENCHMARK_ROOT / "model" / "effective_model_summary.yaml")
    train_cfg = read_yaml(BENCHMARK_ROOT / "train" / "train_config.yaml")
    ckpt = read_yaml(BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml")

    # Summarize data ranges and storage.
    dates = dict(meta["dates"])
    groups = dict(meta["storage"]["groups"])
    data_rows = [
        ("raw_date_range", f"{meta['prep_config']['start_trade_date']} -> {meta['prep_config']['end_trade_date']}"),
        ("train_dates", f"{dates['train'][0]} -> {dates['train'][-1]} ({len(dates['train'])} days)"),
        ("val_dates", f"{dates['val'][0]} -> {dates['val'][-1]} ({len(dates['val'])} days)"),
        ("test_dates", f"{dates['test'][0]} -> {dates['test'][-1]} ({len(dates['test'])} days)"),
        ("train_rows", f"{int(groups['train']['rows']):,}"),
        ("val_rows", f"{int(groups['val']['rows']):,}"),
        ("test_rows", f"{int(groups['test']['rows']):,}"),
        ("feature_dim", str(int(groups["train"]["feature_dim"]))),
        ("features", ", ".join(list(meta["feature_names"]))),
        ("label", str(norm["label"])),
        ("feature_normalization", f"{norm['feature_transform']['stock_norm']['type']} / scope={norm['feature_transform']['stock_norm']['scope']}"),
        ("label_normalization", f"{norm['label_transform']['type']} / scope={norm['label_transform']['scope']}"),
    ]

    # Summarize model architecture and initialization.
    model_rows = [
        ("model_class", str(model["model_class"])),
        ("input_tensor", str(model["input_tensor"])),
        ("gru_input_size", str(model["gru"]["input_size"])),
        ("gru_hidden_size", str(model["gru"]["hidden_size"])),
        ("gru_num_layers", str(model["gru"]["num_layers"])),
        ("gru_bidirectional", str(model["gru"]["bidirectional"])),
        ("mlp_effective_hidden_width", str(model["mlp"]["effective_hidden_width"])),
        ("trainable_parameters", f"{int(model['trainable_parameters']):,}"),
        ("parameter_initialization", "PyTorch default initialization for nn.GRU and nn.Linear; no custom init in prediction_nn2/model.py."),
        ("best_checkpoint", "model/best_checkpoint.pt"),
        ("state_dict_summary", "model/state_dict_summary.yaml"),
    ]

    # Summarize training and evaluation methods.
    method_rows = [
        ("optimizer", str(train_cfg["optimizer"])),
        ("criterion", str(train_cfg["criterion"])),
        ("final_iter", str(train_cfg["stage_manifest"]["final_iter"])),
        ("best_iter", str(train_cfg["stage_manifest"]["best_it"])),
        ("checkpoint_selection", f"{train_cfg['checkpoint_selection']['rule']} {train_cfg['checkpoint_selection']['metric']}"),
        ("checkpoint_saved_to", str(ckpt["best_checkpoint"]["path"])),
        ("prediction_manifest", "predictions/inference_test_manifest.yaml"),
        ("post_train_processing", "eval_train/eval_test/inference_test manifests, IC reports, signal diagnostics."),
        ("backtest_method", "q95_q80_signal_proxy from eval_test prediction buckets; execution backtest is not materialized."),
        ("evaluation_contract", "join validation, IC, signal/time/extreme/normalization/proxy trading metrics, comparison against parent."),
    ]
    # Persist a compact cache for report-only rerenders.
    out = {"data": [list(row) for row in data_rows], "model": [list(row) for row in model_rows], "method": [list(row) for row in method_rows]}
    write_yaml(cache_output, out)
    write_artifact_cache(cache_path, expected_cache)
    return {str(key): [tuple(item) for item in list(value)] for key, value in dict(out).items()}


def markdown_table_from_csv(path: Path, n_rows: int) -> str:
    """Render a small CSV preview as markdown."""
    # Load and truncate the table.
    df = pd.read_csv(Path(path)).head(int(n_rows))

    # Build a compact markdown table.
    headers = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(value) for value in list(row)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def full_report_markdown(summary: dict[str, Any]) -> str:
    """Build the full benchmark evaluation report markdown."""
    # Load core manifests and metric summaries.
    tb_manifest = read_yaml(BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml")
    ckpt_manifest = read_yaml(BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml")
    join_validation = read_yaml(BENCHMARK_ROOT / "evaluation" / "join_validation.yaml")
    train_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_train.yaml")
    test_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_test.yaml")
    rank_turnover = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "test_prediction_rank_turnover.yaml")
    residual = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "test_residual_diagnostics.yaml")
    vol_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "vol_rolling_ic.yaml")
    price_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "price_rolling_ic.yaml")
    comparison = read_yaml(BENCHMARK_ROOT / "evaluation" / "comparison_against_parent.yaml")
    basic = basic_info_rows()

    # Derive retained training artifact details.
    event_files = list(tb_manifest["event_files"])
    tags = list(tb_manifest["tags"])
    retained = [
        "TensorBoard event files and exported scalar parquet.",
        "Checkpoint metrics for validation checkpoint selection.",
        "Best checkpoint metadata and checkpoint file.",
        "Train stage manifest and train config summary.",
        "Train loss curve exported from TensorBoard.",
        "Train/test/inference prediction manifests.",
    ]

    # Compose the report body.
    metrics = dict(summary["headline_metrics"])
    lines = [
        "# Prediction-NN-2 Baseline 完整 Evaluation 报告",
        "",
        "## 结论摘要",
        "",
        f"- benchmark id: `{BENCHMARK_ID}`.",
        f"- replay status: `{read_yaml(BENCHMARK_ROOT / 'replay.yaml')['status']}`.",
        f"- top-minus-bottom: `{float(metrics['top_minus_bottom_bps']):.3f}` bps.",
        f"- high-liq top decile net proxy: `{float(metrics['high_liq_top_decile_net_proxy_bps']):.3f}` bps.",
        f"- q95_q80 net daily return: `{float(metrics['q95_q80_net_daily_return_bps']):.3f}` bps.",
        f"- q95_q80 net bps per turnover: `{float(metrics['q95_q80_net_bps_per_turnover']):.3f}`.",
        f"- best time bucket by net bps/turnover: `{metrics['best_time_bucket_by_net_bps_per_turnover']}`.",
        f"- worst time bucket by net bps/turnover: `{metrics['worst_time_bucket_by_net_bps_per_turnover']}`.",
        "",
        "## Key Takeaways",
        "",
        f"- test daily Pearson IC mean 为 `{float(test_ic['pearson_ic']['mean']):.4f}`, ICIR 为 `{float(test_ic['pearson_ic']['icir']):.2f}`, 统计信号稳定。",
        f"- signal top-minus-bottom 为 `{float(metrics['top_minus_bottom_bps']):.2f}` bps, bottom decile 明显弱于 top decile。",
        f"- q95_q80 gross daily return 为 `{float(metrics['q95_q80_gross_daily_return_bps']):.2f}` bps, 但 net daily return 为 `{float(metrics['q95_q80_net_daily_return_bps']):.2f}` bps, cost 是主要约束。",
        f"- best time bucket 是 `{metrics['best_time_bucket_by_net_bps_per_turnover']}`, worst time bucket 是 `{metrics['worst_time_bucket_by_net_bps_per_turnover']}`。",
        "- 后续实验应复用本报告的 comparison schema, 每次追加 experiment row 和 parent delta。",
        "",
        "## Experiment Comparison Template",
        "",
        "| experiment | date | model | features | label | best_iter | test_ic | top-bottom_bps | net_bps/turnover | cost_model | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| {BENCHMARK_ID} | 2026-05-19 | GruMlpRegressor | 14 | log_close[t+10] - log_close[t+1] | {BEST_CHECKPOINT_ITER} | {float(test_ic['pearson_ic']['mean']):.4f} | {float(metrics['top_minus_bottom_bps']):.2f} | {float(metrics['q95_q80_net_bps_per_turnover']):.4f} | signal proxy | baseline reference |",
        "| future_experiment |  |  |  |  |  |  |  |  |  |  |",
        "",
        "## 已覆盖的 Evaluation",
        "",
        "- checkpoint selector, parameter norm, update norm, train-val gap, runtime scalar summary。",
        "- join validation, IC, signal/liquidity/time/extreme/normalization diagnostics。",
        "- bootstrap CI, month/regime/volatility stability, label availability, turnover, capacity, short-side diagnostics。",
        "- trading metrics and comparison against parent baseline。",
        "",
        "## 训练中保留的数据",
        "",
    ]
    lines.extend([f"- {item}" for item in retained])
    lines.extend(
        [
            "",
            "## 基础信息",
            "",
            "### 数据与 Normalization",
            "",
            "\n".join([f"- {key}: `{value}`." for key, value in basic["data"]]),
            "",
            "### 模型与参数",
            "",
            "\n".join([f"- {key}: `{value}`." for key, value in basic["model"]]),
            "",
            "### 训练、后处理与回测方法",
            "",
            "\n".join([f"- {key}: `{value}`." for key, value in basic["method"]]),
            "",
            "## TensorBoard 与训练监控",
            "",
            f"- TensorBoard source dir: `{tb_manifest['source_dir']}`.",
            f"- exported scalar parquet: `{tb_manifest['scalar_parquet']}`.",
            f"- scalar rows: `{tb_manifest['scalar_rows']}`.",
            f"- event file count: `{len(event_files)}`.",
            f"- best checkpoint iter: `{ckpt_manifest['best_checkpoint_iter']}`.",
            f"- checkpoint selection metric: `{ckpt_manifest['best_selection_metric']}`.",
            "",
            "TensorBoard scalar tags:",
            "",
        ]
    )
    lines.extend([f"- `{tag}`" for tag in tags])
    lines.extend(
        [
            "",
            "Checkpoint metrics preview:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "train" / "checkpoint_metrics.csv", 8),
            "",
            "Checkpoint selector table:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv", 8),
            "",
            "Parameter / update norm diagnostics:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_parameter_update_norms.csv", 8),
            "",
            "Train vs val gap:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_val_gap.csv", 8),
            "",
            "Runtime scalar summary:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_runtime_scalar_summary.csv", 12),
            "",
            "## Join Validation",
            "",
            "```yaml",
            yaml.safe_dump(join_validation, sort_keys=False, allow_unicode=True).strip(),
            "```",
            "",
            "## IC Evaluation",
            "",
            f"- train daily Pearson IC mean: `{float(train_ic['pearson_ic']['mean']):.6f}`, ICIR: `{float(train_ic['pearson_ic']['icir']):.3f}`, positive ratio: `{float(train_ic['pearson_ic']['positive_ratio']):.3f}`.",
            f"- test daily Pearson IC mean: `{float(test_ic['pearson_ic']['mean']):.6f}`, ICIR: `{float(test_ic['pearson_ic']['icir']):.3f}`, positive ratio: `{float(test_ic['pearson_ic']['positive_ratio']):.3f}`.",
            f"- annual test Pearson IC: see `evaluation/model_ic/annual_ic.csv`.",
            "",
            "Intraday IC preview:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "model_ic" / "intraday_ic.csv", 8),
            "",
            "## Rolling IC 与 Rank Turnover",
            "",
            f"- volatility rolling IC max: `{float(vol_ic['pearson_ic_max']['value']):.6f}` at rank `{float(vol_ic['pearson_ic_max']['group_center_rank']):.3f}`.",
            f"- volatility rolling IC min: `{float(vol_ic['pearson_ic_min']['value']):.6f}` at rank `{float(vol_ic['pearson_ic_min']['group_center_rank']):.3f}`.",
            f"- price rolling IC max: `{float(price_ic['pearson_ic_max']['value']):.6f}` at rank `{float(price_ic['pearson_ic_max']['group_center_rank']):.3f}`.",
            f"- price rolling IC min: `{float(price_ic['pearson_ic_min']['value']):.6f}` at rank `{float(price_ic['pearson_ic_min']['group_center_rank']):.3f}`.",
            f"- mean rank turnover: `{float(rank_turnover['mean_rank_turnover']):.6f}`.",
            f"- highest rank turnover time: `{rank_turnover['highest_turnover_time']}`.",
            f"- lowest rank turnover time: `{rank_turnover['lowest_turnover_time']}`.",
            "",
            "## Residual Diagnostics",
            "",
            f"- residual mean: `{float(residual['residual_mean']):.8f}`.",
            f"- residual std: `{float(residual['residual_std']):.8f}`.",
            f"- residual skew: `{float(residual['residual_skew']):.3f}`.",
            f"- residual kurtosis: `{float(residual['residual_kurtosis']):.3f}`.",
            f"- RMSE: `{float(residual['rmse']):.8f}`.",
            "",
            "## Signal Bucket Evaluation",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv", 12),
            "",
            "## Liquidity Evaluation",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv", 18),
            "",
            "## Time Bucket Attribution",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv", 20),
            "",
            "## Extreme Value Diagnostics",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.csv", 20),
            "",
            "## Normalization Diagnostics",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", 30),
            "",
            "## Bootstrap Confidence Intervals",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv", 10),
            "",
            "## Stability Diagnostics",
            "",
            "Month stability:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv", 20),
            "",
            "Regime stability:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv", 10),
            "",
            "Volatility bucket stability:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv", 10),
            "",
            "## Label Availability Coverage",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "label_availability_coverage.csv", 12),
            "",
            "## Turnover, Capacity, Short-side Diagnostics",
            "",
            "Turnover decomposition:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "turnover_decomposition.csv", 10),
            "",
            "Capacity sensitivity:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "capacity_sensitivity_metrics.csv", 12),
            "",
            "Short-side / avoid-bad-stock diagnostics:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "short_side_avoid_bad_stock_metrics.csv", 10),
            "",
            "## Trading 与 Backtest",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv", 8),
            "",
            "Alpha per turnover:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "alpha_per_turnover.csv", 8),
            "",
            "## Comparison Against Parent",
            "",
            "```yaml",
            yaml.safe_dump(comparison, sort_keys=False, allow_unicode=True).strip(),
            "```",
            "",
            "## 可复用代码",
            "",
            "本报告由 `portfolio_backtest/scripts/run_baseline_evaluation.py` 生成。下次实验应复用该脚本中的 contract、metrics 和 report 函数，只替换 `BENCHMARK_ROOT`、`SOURCE_RUN_ROOT`、`SOURCE_SIGNAL_ROOT`、`SOURCE_TRADING_ROOT` 和 `BENCHMARK_ID` 对应的新实验路径。",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_kv_table(rows: list[tuple[str, Any]]) -> str:
    """Render key-value rows as a markdown table."""
    # Convert every value through the report formatter for consistent scanning.
    lines = ["| raw field name | value |", "| --- | --- |"]
    for key, value in list(rows):
        lines.append(f"| `{key}` | `{format_report_value(str(key), value)}` |")
    return "\n".join(lines)


def markdown_rows_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render in-memory rows as a markdown table."""
    # Build the markdown header from raw field names.
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    # Format every row through the report formatter.
    for row in list(rows):
        values = [format_report_cell(str(col), value) for col, value in zip(list(headers), list(row), strict=True)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def markdown_prediction_target_scale_table() -> str:
    """Render prediction/target scale check as markdown."""
    # Build a two-column raw field/value table.
    return markdown_rows_table(["raw field name", "value"], [[field, value] for field, value in prediction_target_scale_rows()])


def markdown_field_notes(raw_fields: list[str], section_note: str) -> str:
    """Render field explanations as markdown bullets."""
    # Start with the section-level note before individual field definitions.
    lines = ["字段解释:", "", f"- {section_note}"]

    # Emit one raw-field definition per requested field.
    for raw_field in list(raw_fields):
        field = FIELD_DEFINITIONS.get(str(raw_field))
        if field is None:
            lines.append(f"- `{raw_field}`: 当前 raw field 暂未登记详细解释。")
            continue
        lines.append(f"- `{raw_field}`: {field['explanation']} 单位: {field['unit']}; 读数方向: {field['direction']}.")
    return "\n".join(lines)


def markdown_comparison_delta_table(comparison: dict[str, Any]) -> str:
    """Render baseline delta comparison as a markdown table."""
    # Skip metrics that the comparison section should no longer display.
    excluded_metrics = {"q95_q80_net_bps_per_turnover", "top_minus_bottom_bps", "high_liq_top_decile_net_proxy_bps"}

    # Convert the primary metric first when it is still eligible.
    primary = dict(comparison["primary_metric"])
    rows = []
    if str(primary["name"]) not in excluded_metrics:
        rows.append([
            str(primary["name"]),
            format_report_value(str(primary["name"]), primary["baseline"]),
            format_report_value(str(primary["name"]), primary["experiment"]),
            format_report_value(str(primary["name"]), primary["absolute_change"]),
            "same baseline",
        ])

    # Append secondary metrics using the baseline self-comparison convention.
    for key, value in dict(comparison["secondary_metrics"]).items():
        if str(key) in excluded_metrics:
            continue
        rows.append([str(key), format_report_value(str(key), value), format_report_value(str(key), value), "0.0000", "reference"])

    # Render the compact markdown table.
    lines = ["| metric | baseline | current | delta | note |", "| --- | --- | --- | --- | --- |"]
    for metric, baseline, current, delta, note in rows:
        lines.append(f"| `{metric}` | `{baseline}` | `{current}` | `{delta}` | {note} |")
    return "\n".join(lines)


def outline_aligned_report_markdown(summary: dict[str, Any]) -> str:
    """Build the benchmark report markdown with the same outline as HTML."""
    # Load the same artifacts used by the HTML report.
    basic = basic_info_rows()
    tb_manifest = read_yaml(BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml")
    join_validation = read_yaml(BENCHMARK_ROOT / "evaluation" / "join_validation.yaml")
    comparison = read_yaml(BENCHMARK_ROOT / "evaluation" / "comparison_against_parent.yaml")
    train_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_train.yaml")["pearson_ic"]
    test_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_test.yaml")["pearson_ic"]
    pooled_ic = source_pooled_ic_values()

    # Assemble data pipeline rows.
    data_basic_fields = ["train_dates", "val_dates", "test_dates", "feature_dim", "features", "label", "feature_normalization", "label_normalization"]
    join_basic_fields = ["prediction_rows", "feature_rows"]
    basic_data = dict(basic["data"])
    data_basic_rows = [(field, basic_data[field]) for field in data_basic_fields] + [(field, join_validation[field]) for field in join_basic_fields]
    normalization_contract_rows = [row for row in data_basic_rows if row[0] in {"feature_normalization", "label_normalization", "label", "feature_dim", "features"}]

    # Compose markdown in the outline order.
    lines = [
        "# Prediction-NN-2 Baseline 完整 Evaluation 报告",
        "",
        "## 1. Executive Summary",
        "",
        markdown_kv_table(
            [
                ("benchmark", BENCHMARK_ID),
                ("selected_checkpoint", BEST_CHECKPOINT_ITER),
                ("test_ic", test_ic["mean"]),
                ("train_ic", train_ic["mean"]),
                ("test_pooled_ic", pooled_ic["test_pooled_ic"]),
                ("train_pooled_ic", pooled_ic["train_pooled_ic"]),
                ("model_class", "GruMlpRegressor"),
            ]
        ),
        "",
        markdown_field_notes(["benchmark", "selected_checkpoint", "test_ic", "train_ic", "test_pooled_ic", "train_pooled_ic", "model_class"], "Executive Summary 展示当前 benchmark 的核心统计信号和模型上下文。"),
        "",
        "## 2. Data Pipeline",
        "",
        "### 2.1. Data Basics",
        "",
        markdown_kv_table(data_basic_rows),
        "",
        markdown_field_notes([key for key, _ in data_basic_rows], "Data Basics 展示本次实验的数据范围、feature/label 设定和 prediction/feature 数据规模。"),
        "",
        "### 2.2. Data Normalization",
        "",
        "Normalization contract:",
        "",
        markdown_kv_table(normalization_contract_rows),
        "",
        "Normalization diagnostics:",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", 30),
        "",
        "Prediction/target scale check:",
        "",
        markdown_prediction_target_scale_table(),
        "",
        "Figure:",
        "",
        "- `evaluation/figures/test_residual_diagnostics.png`: Residual distribution diagnostics.",
        "",
        markdown_field_notes(["feature_normalization", "label_normalization", "label", "feature_dim", "features", "metric_name", "value", "split", "note", "prediction_mean", "prediction_std", "target_mean", "target_std", "pred_std_over_target_std", "prediction_p01", "prediction_p50", "prediction_p99", "target_p01", "target_p50", "target_p99"], "Data Normalization 解释 feature/label normalization contract, 并检查 prediction/target scale 是否合理。"),
        "",
        "## 3. Experiment Comparison",
        "",
        "### 3.1. Experiment Rows",
        "",
        "| experiment | date | model | features | label | best_iter | test_ic | cost_model | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| {BENCHMARK_ID} | 2026-05-19 | GruMlpRegressor | 14 | log_close[t+10] - log_close[t+1] | {BEST_CHECKPOINT_ITER} | {format_report_value('test_ic', test_ic['mean'])} | signal proxy | baseline reference |",
        "| future_experiment |  |  |  |  |  |  |  |  |",
        "",
        markdown_field_notes(["experiment", "date", "model", "features", "label", "best_iter", "test_ic", "cost_model", "verdict"], "Experiment Rows 预留后续实验对比位置。"),
        "",
        "### 3.2. Metric Delta Against Baseline",
        "",
        markdown_comparison_delta_table(comparison),
        "",
        "<details><summary>raw comparison YAML</summary>",
        "",
        "```yaml",
        yaml.safe_dump(comparison, sort_keys=False, allow_unicode=True).strip(),
        "```",
        "",
        "</details>",
        "",
        markdown_field_notes(["metric", "baseline", "current", "delta", "note", "daily_turnover"], "Metric Delta Against Baseline 比较当前 experiment 与 baseline。"),
        "",
        "## 4. Evaluation",
        "",
        "### 4.1. IC Summary",
        "",
        markdown_rows_table(
            ["split", "count", "mean_ic", "std_ic", "t_stat", "positive_ratio", "icir", "pooled_ic", "pooled_rank_ic"],
            [
                ["train", train_ic["count"], train_ic["mean"], train_ic["std"], train_ic["t_stat"], train_ic["positive_ratio"], train_ic["icir"], pooled_ic["train_pooled_ic"], float("nan")],
                ["test", test_ic["count"], test_ic["mean"], test_ic["std"], test_ic["t_stat"], test_ic["positive_ratio"], test_ic["icir"], pooled_ic["test_pooled_ic"], float("nan")],
            ],
        ),
        "",
        markdown_field_notes(["split", "count", "mean_ic", "std_ic", "t_stat", "positive_ratio", "icir", "pooled_ic", "pooled_rank_ic"], "IC Summary 用来判断模型样本外统计信号是否稳定。"),
        "",
        "### 4.2. Signal Quality",
        "",
        "#### Bucket Construction Principle",
        "",
        "每个 evaluation 截面内按 `prediction` 从低到高排序, 再切成 `signal_bucket`. `signal_bucket=1` 代表最低 prediction 分组, `signal_bucket=10` 代表最高 prediction 分组。读表时应检查 `mean_prediction` 是否随 bucket 单调上升, 以及 `mean_return_bps` 和 `hit_rate` 是否随 bucket 改善。",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv", 12),
        "",
        markdown_field_notes(["signal_bucket", "row_count", "mean_prediction", "mean_return_bps", "t_stat", "hit_rate", "mean_spread_bps"], "Signal Quality 检查 prediction 排序后收益是否随 bucket 单调变强。"),
        "",
        "### 4.3. Trading And Cost",
        "",
        "#### Cost Model",
        "",
        "本报告当前使用 signal proxy 成本模型: 0519 最新训练产物尚未 materialize execution backtest, 因此 trading 表使用 top-minus-bottom signal bucket 作为 unit-turnover proxy, spread_cost_bps 和 fee_cost_bps 置为 0。它适合先检查 B-label 模型排序能力, 但不能替代后续真实 spread + fee execution backtest。",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv", 8),
        "",
        markdown_field_notes(["strategy_name", "open_quantile", "close_quantile", "gross_daily_return_bps", "net_daily_return_bps", "gross_sharpe", "net_sharpe", "max_drawdown", "daily_turnover", "spread_cost_bps", "fee_cost_bps", "gross_bps_per_turnover", "net_bps_per_turnover", "active_names"], "Trading And Cost 判断统计 signal 转成交易后是否还能覆盖 spread 和 fee。"),
        "",
        "### 4.4. Time And Liquidity Attribution",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv", 20),
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv", 18),
        "",
        markdown_field_notes(["time_bucket", "gross_return_bps", "turnover", "net_return_bps", "liq_bucket", "entry_spread_cost_bps", "entry_fee_cost_bps", "entry_net_proxy_bps", "mean_signal_amount"], "Time 与 liquidity attribution 用来定位哪些时段和流动性区域贡献收益。"),
        "",
        "### 4.5. Robustness Diagnostics",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv", 10),
        "",
        "Figure: `reports/figures/bootstrap_confidence_intervals.png`.",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv", 20),
        "",
        "Figure: `reports/figures/month_stability_metrics.png`.",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv", 10),
        "",
        "Figure: `reports/figures/regime_stability_metrics.png`.",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv", 10),
        "",
        "Figure: `reports/figures/volatility_bucket_stability_metrics.png`.",
        "",
        "Figure: `evaluation/figures/vol_rolling_ic.png`.",
        "",
        "Figure: `evaluation/figures/price_rolling_ic.png`.",
        "",
        markdown_field_notes(["metric", "mean", "std", "ci_025", "ci_500", "ci_975", "sample_count", "bootstrap_samples", "month", "positive_top_minus_bottom_ratio", "regime", "volatility_bucket", "mean_rank_ic"], "Robustness Diagnostics 检查核心指标在 bootstrap、月份、market regime 和 volatility bucket 下是否稳定。"),
        "",
        "## 5. Train Metrics",
        "",
        "### 5.1. Checkpoint Selection",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv", 8),
        "",
        markdown_field_notes(["iter", "selected", "val/objective/mse", "val/quality/global_ic", "val/quality/rank_ic", "val/dist/pred_std_over_target_std", "metric_rank", "decision"], "Checkpoint Selection 解释为什么选中 best checkpoint。"),
        "",
        "### 5.2. Model And Method",
        "",
        markdown_kv_table(basic["model"] + basic["method"]),
        "",
        markdown_field_notes([key for key, _ in basic["model"] + basic["method"]], "Model And Method 记录模型结构、训练目标、checkpoint 选择规则和 evaluation contract。"),
        "",
        "### 5.3. Training Monitoring",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_val_gap.csv", 8),
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_runtime_scalar_summary.csv", 12),
        "",
        f"- TensorBoard source dir: `{tb_manifest['source_dir']}`.",
        f"- scalar rows: `{tb_manifest['scalar_rows']}`.",
        "",
        markdown_field_notes(["step", "val_mse", "train_loss_mean", "val_minus_train", "val_over_train", "tag", "last_step", "last_value", "tail100_mean", "tail100_std", "min_value", "max_value"], "Training Monitoring 检查 train/validation loss gap、runtime scalar 和参数更新范数。"),
        "",
        "## 6. Appendix Diagnostics",
        "",
        "- `intraday IC table`: `evaluation/model_ic/intraday_ic.csv`.",
        "- `extreme value metrics`: `evaluation/extreme_value_metrics.csv`.",
        "- `normalization metrics`: `evaluation/normalization_metrics.csv`.",
        "- `label availability coverage`: `evaluation/label_availability_coverage.csv`.",
        "- `turnover decomposition`: `evaluation/turnover_decomposition.csv`.",
        "- `capacity sensitivity`: `evaluation/capacity_sensitivity_metrics.csv`.",
        "- `short-side diagnostics`: `evaluation/short_side_avoid_bad_stock_metrics.csv`.",
    ]
    return "\n".join(lines) + "\n"


def write_full_benchmark_report(summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write the full benchmark report into benchmark and repo report dirs."""
    # Build and persist the markdown report.
    markdown = outline_aligned_report_markdown(summary)
    report_md = BENCHMARK_ROOT / "reports" / "full_evaluation_report.md"
    report_md.write_text(markdown, encoding="utf-8")

    # Render a self-contained HTML report with native sections and tables.
    basic = basic_info_rows()
    tb_manifest = read_yaml(BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml")
    join_validation = read_yaml(BENCHMARK_ROOT / "evaluation" / "join_validation.yaml")
    comparison = read_yaml(BENCHMARK_ROOT / "evaluation" / "comparison_against_parent.yaml")
    train_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_train.yaml")["pearson_ic"]
    test_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_test.yaml")["pearson_ic"]
    pooled_ic = source_pooled_ic_values()
    figures = build_report_figures(summary, dict(train_ic), dict(test_ic))

    # Derive the first-screen summary values.
    summary_cards = [
        ("benchmark", BENCHMARK_ID, "current frozen baseline"),
        ("selected_checkpoint", str(BEST_CHECKPOINT_ITER), "minimum validation MSE"),
        ("test_ic", format_report_value("test_ic", test_ic["mean"]), f"ICIR {format_report_value('icir', test_ic['icir'])}"),
        ("train_ic", format_report_value("train_ic", train_ic["mean"]), f"ICIR {format_report_value('icir', train_ic['icir'])}"),
        ("test_pooled_ic", format_report_value("test_pooled_ic", pooled_ic["test_pooled_ic"]), "pooled Pearson IC"),
        ("train_pooled_ic", format_report_value("train_pooled_ic", pooled_ic["train_pooled_ic"]), "pooled Pearson IC"),
        ("model_class", "GruMlpRegressor", "model architecture"),
    ]

    # Write concise investment-style takeaways.
    takeaways = [
        f"模型 IC 在 test split 上为 {format_report_value('test_ic', test_ic['mean'])}, daily ICIR 为 {format_report_value('icir', test_ic['icir'])}, 统计信号稳定。",
        f"当前报告使用 iter {BEST_CHECKPOINT_ITER} 作为 selected checkpoint, 选择规则是 minimum validation MSE。",
        "数据范围、normalization 与 prediction/feature 规模在 Data Pipeline 中展开, signal 与 trading 指标在 Evaluation 中展开。",
        "本报告已预留 Experiment Comparison 行, 后续实验可以直接追加到同一 schema。",
    ]

    # Build the data pipeline rows from data contracts and join validation.
    data_basic_fields = ["train_dates", "val_dates", "test_dates", "feature_dim", "features", "label", "feature_normalization", "label_normalization"]
    join_basic_fields = ["prediction_rows", "feature_rows"]
    basic_data = dict(basic["data"])
    data_basic_rows = [(field, basic_data[field]) for field in data_basic_fields] + [(field, format_report_value(field, join_validation[field])) for field in join_basic_fields]
    normalization_contract_rows = [row for row in data_basic_rows if row[0] in {"feature_normalization", "label_normalization", "label", "feature_dim", "features"}]

    # Render sections from high-level conclusion to appendices.
    data_pipeline_body = (
        render_subsection(
            "2.1. Data Basics",
            render_value_rows(data_basic_rows)
            + render_field_notes(data_basic_fields + join_basic_fields, "Data Basics 展示本次实验的数据范围、feature/label 设定和 prediction/feature 数据规模。"),
        )
        + render_subsection(
            "2.2. Data Normalization",
            render_block_title("normalization contract")
            + render_value_rows(normalization_contract_rows)
            + render_block_title("normalization_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", 30)
            + render_block_title("prediction/target scale check")
            + prediction_target_scale_table()
            + render_embedded_figure("Residual Diagnostics", BENCHMARK_ROOT / "evaluation" / "figures" / "test_residual_diagnostics.png", "Residual distribution diagnostics.")
            + render_field_notes(["feature_normalization", "label_normalization", "label", "feature_dim", "features", "metric_name", "value", "split", "note", "prediction_mean", "prediction_std", "target_mean", "target_std", "pred_std_over_target_std", "prediction_p01", "prediction_p50", "prediction_p99", "target_p01", "target_p50", "target_p99"], "Data Normalization 解释 feature/label normalization contract, 并检查 prediction/target scale 是否合理。"),
        )
    )
    experiment_body = (
        render_subsection(
            "3.1. Experiment Rows",
            experiment_comparison_placeholder(summary, dict(test_ic))
            + render_field_notes(["experiment", "date", "model", "features", "label", "best_iter", "test_ic", "cost_model", "verdict"], "Experiment Rows 预留后续实验对比位置。后续新实验应使用同一 schema 追加一行。"),
        )
        + render_subsection(
            "3.2. Metric Delta Against Baseline",
            comparison_delta_table(comparison)
            + render_details_code("raw comparison YAML", yaml.safe_dump(comparison, sort_keys=False, allow_unicode=True).strip())
            + render_field_notes(["metric", "baseline", "current", "delta", "note", "test_ic", "mean_ic", "std_ic", "t_stat", "ic_positive_ratio", "icir", "top_decile_return_bps", "daily_turnover"], "Metric Delta Against Baseline 比较当前 experiment 与 baseline。当前报告是 baseline self-comparison, delta 为 0。")
        )
    )
    evaluation_body = (
        render_subsection(
            "4.1. IC Summary",
            ic_summary_table(dict(train_ic), dict(test_ic))
            + render_embedded_figure("IC Summary Bar", figures["ic_summary"], "Train/test mean_ic 与 icir 对比。")
            + render_embedded_figure("Intraday IC", BENCHMARK_ROOT / "evaluation" / "figures" / "intraday_ic.png", "Train/test intraday IC.")
            + render_field_notes(["split", "count", "mean_ic", "std_ic", "t_stat", "positive_ratio", "icir", "pooled_ic", "pooled_rank_ic"], "IC Summary 用来判断模型样本外统计信号是否稳定。test IC 越高越好, ICIR 越高说明 daily IC 的波动更小。"),
        )
        + render_subsection(
            "4.2. Signal Quality",
            render_block_title("Bucket Construction Principle")
            + render_field_notes([], "每个 evaluation 截面内按 prediction 从低到高排序, 再切成 signal_bucket。signal_bucket=1 代表最低 prediction 分组, signal_bucket=10 代表最高 prediction 分组。读表时应检查 mean_prediction 是否随 bucket 单调上升, 以及 mean_return_bps 和 hit_rate 是否随 bucket 改善。")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv", 12)
            + render_embedded_figure("Signal Bucket Return Hit Rate", figures["signal_quality"], "每个 signal_bucket 的 mean_return_bps 和 hit_rate。")
            + render_embedded_figure("Signal Decile Return", BENCHMARK_ROOT / "evaluation" / "figures" / "model_signal_decile_return.png", "Signal decile return.")
            + render_embedded_figure("Signal Top Minus Bottom", BENCHMARK_ROOT / "evaluation" / "figures" / "model_signal_top_minus_bottom.png", "Daily top-minus-bottom signal spread.")
            + render_field_notes(["signal_bucket", "row_count", "mean_prediction", "mean_return_bps", "t_stat", "hit_rate", "mean_spread_bps", "top_minus_bottom_bps"], "Signal Quality 检查 prediction 排序后收益是否随 bucket 单调变强。当前 bottom decile 很弱, top-bottom spread 为正, 说明 signal 有明显排序能力。"),
        )
        + render_subsection(
            "4.3. Trading And Cost",
            render_block_title("Cost Model")
            + render_field_notes([], "本报告当前使用 signal proxy 成本模型: 0519 最新训练产物尚未 materialize execution backtest, 因此 trading 表使用 top-minus-bottom signal bucket 作为 unit-turnover proxy, spread_cost_bps 和 fee_cost_bps 置为 0。它适合先检查 B-label 模型排序能力, 但不能替代后续真实 spread + fee execution backtest。")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv", 8)
            + render_embedded_figure("Trading Cost Bridge", figures["trading_cost"], "q95/q80 策略从 gross 到 net 的成本拆解。")
            + render_field_notes(["strategy_name", "open_quantile", "close_quantile", "gross_daily_return_bps", "net_daily_return_bps", "gross_sharpe", "net_sharpe", "max_drawdown", "daily_turnover", "spread_cost_bps", "fee_cost_bps", "gross_bps_per_turnover", "net_bps_per_turnover", "active_names"], "Trading And Cost 判断统计 signal 转成交易后是否还能覆盖 spread 和 fee。当前 gross 为正但 net 为负, 说明交易规则或成本控制仍需改进。"),
        )
        + render_subsection(
            "4.4. Time And Liquidity Attribution",
            render_block_title("time_bucket_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv", 20)
            + render_embedded_figure("Time Bucket Net Bps Per Turnover", figures["time_bucket"], "按 time_bucket 展示 net_bps_per_turnover。")
            + render_block_title("liquidity_bucket_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv", 18)
            + render_embedded_figure("Liquidity Top Signal Bucket", figures["liquidity_top"], "Top signal_bucket 在不同 liq_bucket 下的 entry_net_proxy_bps。")
            + render_field_notes(["time_bucket", "row_count", "gross_return_bps", "turnover", "spread_cost_bps", "fee_cost_bps", "net_return_bps", "gross_bps_per_turnover", "net_bps_per_turnover", "active_names", "liq_bucket", "signal_bucket", "entry_spread_cost_bps", "entry_fee_cost_bps", "entry_net_proxy_bps", "mean_signal_amount", "mean_spread_bps"], "Time 与 liquidity attribution 用来定位哪些时段和流动性区域贡献收益, 哪些区域需要过滤或降低 turnover。"),
        )
        + render_subsection(
            "4.5. Robustness Diagnostics",
            render_block_title("bootstrap_confidence_intervals.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv", 10)
            + render_embedded_figure("Bootstrap Confidence Intervals", figures["bootstrap_ci"], "Bootstrap 区间展示核心指标的不确定性。")
            + render_block_title("month_stability_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv", 20)
            + render_embedded_figure("Month Stability Metrics", figures["month_stability"], "按月份展示 top_minus_bottom_bps 和 q95/q80 net bps per turnover。")
            + render_block_title("regime_stability_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv", 10)
            + render_embedded_figure("Regime Stability Metrics", figures["regime_stability"], "按 market regime 对比 gross_daily_bps 与 net_daily_bps。")
            + render_block_title("volatility_bucket_stability_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv", 10)
            + render_embedded_figure("Volatility Bucket Stability Metrics", figures["volatility_stability"], "按 volatility_bucket 对比 mean_ic 与 mean_rank_ic。")
            + render_embedded_figure("Vol Rolling IC", BENCHMARK_ROOT / "evaluation" / "figures" / "vol_rolling_ic.png", "IC by volatility rank.")
            + render_embedded_figure("Price Rolling IC", BENCHMARK_ROOT / "evaluation" / "figures" / "price_rolling_ic.png", "IC by price rank.")
            + render_field_notes(["metric", "mean", "std", "ci_025", "ci_500", "ci_975", "sample_count", "bootstrap_samples", "month", "positive_top_minus_bottom_ratio", "q95_q80_net_daily_bps", "q95_q80_net_bps_per_turnover", "daily_turnover", "regime", "day_count", "gross_daily_bps", "net_daily_bps", "net_bps_per_turnover", "turnover", "volatility_bucket", "row_count", "mean_ic", "mean_rank_ic", "center_rank_min", "center_rank_max"], "Robustness Diagnostics 检查核心指标在 bootstrap、月份、market regime 和 volatility bucket 下是否稳定。区间跨 0 的指标需要谨慎。"),
        )
    )
    train_body = (
        render_subsection(
            "5.1. Checkpoint Selection",
            checkpoint_summary_table()
            + render_embedded_figure("Checkpoint Selection Metrics", figures["checkpoint"], "val/objective/mse 和 IC 随 iter 的变化, 红线为 selected checkpoint。")
            + render_field_notes(["iter", "selected", "val/objective/mse", "val/quality/global_ic", "val/quality/rank_ic", "val/dist/pred_std_over_target_std", "metric_rank", "decision"], f"Checkpoint Selection 解释为什么选中 iter {BEST_CHECKPOINT_ITER}。selector 以 validation MSE 为主, 同时保留 IC 与 prediction scale 作为 sanity check。")
            + render_details_table("full checkpoint selector", render_csv_table(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv", 8)),
        )
        + render_subsection(
            "5.2. Model And Method",
            render_block_title("Model")
            + render_value_rows(basic["model"])
            + render_block_title("Method")
            + render_value_rows(basic["method"])
            + render_field_notes([row[0] for row in basic["model"] + basic["method"]], "Model And Method 记录模型结构、训练目标、checkpoint 选择规则和 evaluation contract, 用于复现实验。"),
        )
        + render_subsection(
            "5.3. Training Monitoring",
            render_block_title("train_val_gap.csv")
            + render_csv_table(BENCHMARK_ROOT / "train" / "diagnostics" / "train_val_gap.csv", 8)
            + render_block_title("train_runtime_scalar_summary.csv")
            + render_csv_table(BENCHMARK_ROOT / "train" / "diagnostics" / "train_runtime_scalar_summary.csv", 12)
            + render_embedded_figure("Train Loss", BENCHMARK_ROOT / "train" / "train_loss_curve.png", "TensorBoard loss curve.")
            + render_details_table("parameter and update norm", render_csv_table(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_parameter_update_norms.csv", 8))
            + render_details_table("TensorBoard storage", render_value_rows([("source_dir", str(tb_manifest["source_dir"])), ("scalar_parquet", str(tb_manifest["scalar_parquet"])), ("scalar_rows", str(tb_manifest["scalar_rows"])), ("tags", ", ".join(list(tb_manifest["tags"])))]))
            + render_field_notes(["step", "val_mse", "train_loss_mean", "val_minus_train", "val_over_train", "tag", "last_step", "last_value", "tail100_mean", "tail100_std", "min_value", "max_value"], "Training Monitoring 检查 train/validation loss gap、runtime scalar 和参数更新范数, 用于发现训练不稳定或过拟合迹象。"),
        )
    )
    appendix_body = (
        render_details_table("intraday IC table", render_csv_table(BENCHMARK_ROOT / "evaluation" / "model_ic" / "intraday_ic.csv", 12))
        + render_details_table("extreme value metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.csv", 20))
        + render_details_table("normalization metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", 30))
        + render_details_table("label availability coverage", render_csv_table(BENCHMARK_ROOT / "evaluation" / "label_availability_coverage.csv", 12))
        + render_details_table("turnover decomposition", render_csv_table(BENCHMARK_ROOT / "evaluation" / "turnover_decomposition.csv", 10))
        + render_details_table("capacity sensitivity", render_csv_table(BENCHMARK_ROOT / "evaluation" / "capacity_sensitivity_metrics.csv", 12))
        + render_details_table("short-side diagnostics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "short_side_avoid_bad_stock_metrics.csv", 10))
        + render_field_notes([], "Appendix Diagnostics 保留长表和辅助诊断, 用于追查主线指标背后的数据质量、normalization、turnover、capacity 和 short-side 行为。")
    )
    sections = [
        render_section("1. Executive Summary", render_summary_cards(summary_cards) + render_takeaways(takeaways)),
        render_section("2. Data Pipeline", data_pipeline_body),
        render_section("3. Experiment Comparison", experiment_body),
        render_section("4. Evaluation", evaluation_body),
        render_section("5. Train Metrics", train_body),
        render_section("6. Appendix Diagnostics", appendix_body),
    ]
    html = build_page("Prediction-NN-2 Baseline 完整 Evaluation 报告", "训练中监测、IC、signal diagnostics、normalization、extreme values 与 backtest 汇总。", sections)
    report_html = BENCHMARK_ROOT / "reports" / "full_evaluation_report.html"
    report_html.write_text(html, encoding="utf-8")

    # Copy the lightweight public reports into repo report/benchmarks.
    copy_file(report_md, REPO_BENCHMARK_REPORT_DIR / "current_baseline_full_evaluation_report.md")
    copy_file(report_html, REPO_BENCHMARK_REPORT_DIR / "current_baseline_full_evaluation_report.html")
    return report_md, report_html


def write_train_monitoring_html() -> Path:
    """Write the training monitoring HTML report."""
    # Load train artifacts.
    tb_manifest = read_yaml(BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml")
    checkpoint_manifest = read_yaml(BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml")

    # Render training monitoring sections.
    sections = [
        render_section("TensorBoard Manifest", render_code_block(yaml.safe_dump(tb_manifest, sort_keys=False, allow_unicode=True))),
        render_section("Checkpoint Manifest", render_code_block(yaml.safe_dump(checkpoint_manifest, sort_keys=False, allow_unicode=True))),
        render_section("Checkpoint Metrics", render_csv_table(BENCHMARK_ROOT / "train" / "checkpoint_metrics.csv", 12)),
        render_section("Train Loss Curve", render_embedded_figure("Train Loss Curve", BENCHMARK_ROOT / "train" / "train_loss_curve.png", "TensorBoard train/objective/loss_mean exported as PNG.")),
    ]
    html = build_page("Baseline Train Monitoring", "TensorBoard and checkpoint monitoring for the frozen baseline.", sections)
    out = BENCHMARK_ROOT / "reports" / "train_monitoring.html"
    out.write_text(html, encoding="utf-8")
    return out


def write_model_signal_evaluation_html(summary: dict[str, Any]) -> Path:
    """Write the model signal evaluation HTML report."""
    # Render model signal sections.
    sections = [
        render_section("Headline Metrics", render_value_rows([(key, str(value)) for key, value in dict(summary["headline_metrics"]).items()])),
        render_section("Join Validation", render_code_block((BENCHMARK_ROOT / "evaluation" / "join_validation.yaml").read_text(encoding="utf-8"))),
        render_section("Signal Bucket Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv", 12)),
        render_section("Liquidity Bucket Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv", 18)),
        render_section("Time Bucket Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv", 30)),
        render_section("Extreme Value Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.csv", 20)),
        render_section("Normalization Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", 30)),
        render_section("Signal Decile Figure", render_embedded_figure("Signal Decile Return", BENCHMARK_ROOT / "evaluation" / "figures" / "model_signal_decile_return.png", "Existing signal decile return diagnostics.")),
    ]
    html = build_page("Baseline Model Signal Evaluation", "Post-training statistical evaluation for the frozen baseline.", sections)
    out = BENCHMARK_ROOT / "reports" / "model_signal_evaluation.html"
    out.write_text(html, encoding="utf-8")
    return out


def write_trading_evaluation_html() -> Path:
    """Write the trading evaluation HTML report."""
    # Render trading sections.
    sections = [
        render_section("Trading Rule Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv", 12)),
        render_section("Alpha Per Turnover", render_csv_table(BENCHMARK_ROOT / "evaluation" / "alpha_per_turnover.csv", 12)),
        render_section("Trading Baseline Report", render_code_block((BENCHMARK_ROOT / "reports" / "trading_baseline_report.md").read_text(encoding="utf-8"))),
    ]
    html = build_page("Baseline Trading Evaluation", "Trading-rule evaluation for the frozen baseline.", sections)
    out = BENCHMARK_ROOT / "reports" / "trading_evaluation.html"
    out.write_text(html, encoding="utf-8")
    return out


def write_evaluation_card_html(summary: dict[str, Any], comparison: dict[str, Any]) -> Path:
    """Write the top-level evaluation card HTML report."""
    # Render the top-level report card.
    sections = [
        render_section("Headline Metrics", render_value_rows([(key, str(value)) for key, value in dict(summary["headline_metrics"]).items()])),
        render_section("Evaluation Summary", render_code_block(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))),
        render_section("Comparison Against Parent", render_code_block(yaml.safe_dump(comparison, sort_keys=False, allow_unicode=True))),
        render_section("Train Monitoring", render_code_block("reports/train_monitoring.html")),
        render_section("Model Signal Evaluation", render_code_block("reports/model_signal_evaluation.html")),
        render_section("Trading Evaluation", render_code_block("reports/trading_evaluation.html")),
    ]
    html = build_page("Baseline Evaluation Card", "Complete baseline evaluation entry point.", sections)
    out = BENCHMARK_ROOT / "reports" / "evaluation_card.html"
    out.write_text(html, encoding="utf-8")
    copy_file(out, REPO_BENCHMARK_REPORT_DIR / "current_baseline_evaluation_card.html")
    return out
