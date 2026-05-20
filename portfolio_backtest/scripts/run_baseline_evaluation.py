"""Run the complete benchmark evaluation build."""

from __future__ import annotations

from portfolio_backtest.benchmark.evaluation_builder import (
    BENCHMARK_ROOT,
    REPO_BENCHMARK_REPORT_DIR,
    build_bootstrap_confidence_intervals,
    build_comparison,
    build_evaluation_input_manifest,
    build_evaluation_summary,
    build_extreme_value_metrics,
    build_label_availability_coverage,
    build_normalization_metrics,
    build_short_side_diagnostics,
    build_stability_diagnostics,
    build_time_bucket_metrics,
    build_training_diagnostics,
    build_turnover_and_capacity_diagnostics,
    copy_training_post_eval_artifacts,
    run_join_validation,
    standardize_liquidity_metrics,
    standardize_signal_metrics,
    standardize_trading_rule_metrics,
    sync_latest_training_artifacts,
    update_benchmark_and_replay,
)
from portfolio_backtest.benchmark.report_renderer import (
    write_evaluation_card_html,
    write_full_benchmark_report,
    write_model_signal_evaluation_html,
    write_trading_evaluation_html,
    write_train_monitoring_html,
)


def run_baseline_evaluation():
    """Run the complete baseline evaluation build."""
    # Create the target directories.
    (BENCHMARK_ROOT / "evaluation").mkdir(parents=True, exist_ok=True)
    (BENCHMARK_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    REPO_BENCHMARK_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Sync the latest training artifacts before deriving report tables.
    sync_latest_training_artifacts()

    # Build contract and metrics artifacts.
    build_evaluation_input_manifest()
    join_validation = run_join_validation()
    signal = standardize_signal_metrics()
    liquidity = standardize_liquidity_metrics()
    time_bucket = build_time_bucket_metrics()
    build_extreme_value_metrics()
    build_normalization_metrics()
    copy_training_post_eval_artifacts()
    build_training_diagnostics()
    build_bootstrap_confidence_intervals()
    build_stability_diagnostics()
    build_label_availability_coverage()
    build_turnover_and_capacity_diagnostics()
    build_short_side_diagnostics()
    trading = standardize_trading_rule_metrics()
    summary = build_evaluation_summary(join_validation, signal, liquidity, time_bucket, trading)
    comparison = build_comparison(summary)

    # Render reports and update replay.
    write_train_monitoring_html()
    write_model_signal_evaluation_html(summary)
    write_trading_evaluation_html()
    card = write_evaluation_card_html(summary, comparison)
    write_full_benchmark_report(summary)
    replay = update_benchmark_and_replay(summary)
    if replay["status"] != "passed":
        raise RuntimeError(f"replay failed: {replay}")
    return card


if __name__ == "__main__":
    print(run_baseline_evaluation().as_posix())
