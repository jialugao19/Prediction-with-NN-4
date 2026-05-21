"""Render benchmark reports from existing evaluation artifacts."""

from __future__ import annotations

from portfolio_backtest.benchmark.evaluation_builder import BENCHMARK_ROOT, REPO_BENCHMARK_REPORT_DIR, read_yaml, update_benchmark_and_replay
from portfolio_backtest.benchmark.report_renderer import (
    write_evaluation_card_html,
    write_full_benchmark_report,
    write_model_signal_evaluation_html,
    write_trading_evaluation_html,
    write_train_monitoring_html,
)


def render_baseline_report():
    """Render report files without rebuilding evaluation artifacts."""
    # Ensure report output directories exist.
    (BENCHMARK_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    REPO_BENCHMARK_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing evaluation artifacts as report inputs.
    summary = read_yaml(BENCHMARK_ROOT / "evaluation" / "evaluation_summary.yaml")
    comparison = read_yaml(BENCHMARK_ROOT / "evaluation" / "comparison_against_parent.yaml")

    # Render the report family from existing CSV/YAML/PNG artifacts.
    write_train_monitoring_html()
    write_model_signal_evaluation_html(summary)
    write_trading_evaluation_html()
    card = write_evaluation_card_html(summary, comparison)
    write_full_benchmark_report(summary)

    # Re-run the artifact/readability gate without recomputing metrics.
    replay = update_benchmark_and_replay(summary)
    if replay["status"] != "passed":
        raise RuntimeError(f"replay failed: {replay}")
    return card


if __name__ == "__main__":
    print(render_baseline_report().as_posix())
