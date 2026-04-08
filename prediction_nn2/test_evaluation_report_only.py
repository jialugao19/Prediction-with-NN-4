"""Rebuild `test_evaluation_report.html` from an existing test manifest without rerunning training."""

from prediction_nn2.pipeline import _default_config, run_test_evaluation_report_postprocess_only


def main() -> None:
    """Run the test-evaluation-report rebuild entrypoint with the module-level default config."""
    # Reuse the shared default config and override behavior via the dedicated entry function.
    cfg = _default_config()
    run_test_evaluation_report_postprocess_only(cfg)


if __name__ == "__main__":
    main()
