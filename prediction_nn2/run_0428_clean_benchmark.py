"""Run the 0428 clean benchmark and publish the final HTML reports."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import yaml

from portfolio_backtest.contract import build_default_portfolio_backtest_config
from portfolio_backtest.run_portfolio_backtest import run_portfolio_backtest
from prediction_nn2.pipeline import _default_config, run_pipeline


REPO_ROOT = Path("/home/maomao/prediction-NN-2")
RUN_ROOT = Path("/data-cache/nn/0428")
SPLIT_ROOT = RUN_ROOT / "date_ranges"
PIPELINE_RUN_ROOT = SPLIT_ROOT / "run"
PORTFOLIO_OUTPUT_DIR = RUN_ROOT / "portfolio_backtest"
REPORT_DIR = REPO_ROOT / "report" / "0428"


def build_pipeline_config():
    """Build the fixed clean benchmark pipeline config."""
    # Reuse the project default and pin the benchmark root.
    base = _default_config()
    cfg = replace(
        base,
        pipeline_mode="test_full",
        root_dir=RUN_ROOT,
    )
    return cfg


def read_yaml(path: Path) -> dict:
    """Read one YAML payload from disk."""
    # Load the file as UTF-8 YAML.
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def find_inference_manifest() -> Path:
    """Find the clean inference manifest selected by the test-evaluation stage."""
    # Read the stage manifest because it records the selected best checkpoint path.
    stage_manifest_candidates = [
        SPLIT_ROOT / "manifests" / "test_evaluation.yaml",
        SPLIT_ROOT / "stage_manifests" / "test_evaluation.yaml",
    ]
    stage_manifest = next(path for path in stage_manifest_candidates if path.exists())
    payload = read_yaml(stage_manifest)
    manifest_path = Path(str(payload["inference_manifest_path"]))

    # Require the manifest to be the formal inference output.
    manifest = read_yaml(manifest_path)
    columns = list(manifest["columns"])
    if columns != ["prediction", "code", "date", "time"]:
        raise RuntimeError(f"Unexpected inference manifest columns: {columns}")
    if "compat_mode" in dict(manifest):
        raise RuntimeError(f"Refusing compatibility inference manifest: {manifest_path}")
    if str(manifest["group"]) != "inference_test":
        raise RuntimeError(f"Unexpected inference manifest group: {manifest['group']}")
    return manifest_path


def pipeline_outputs_exist() -> bool:
    """Check whether the expensive pipeline stages already finished."""
    # Require the HTML reports and manifest that downstream stages consume.
    required_paths = [
        SPLIT_ROOT / "train_report.html",
        SPLIT_ROOT / "test_evaluation_report.html",
        SPLIT_ROOT / "manifests" / "test_evaluation.yaml",
    ]
    return all(path.exists() for path in required_paths)


def run_clean_pipeline() -> Path:
    """Run the clean training, eval, inference, and test-report pipeline."""
    # Reuse completed pipeline outputs when resuming the wrapper.
    if pipeline_outputs_exist():
        return find_inference_manifest()

    # Execute the project pipeline with the fixed 0428 config.
    cfg = build_pipeline_config()
    run_pipeline(cfg)

    # Return the validated formal inference manifest.
    return find_inference_manifest()


def run_clean_portfolio_backtest(inference_manifest_path: Path) -> Path:
    """Run portfolio_backtest against the clean inference manifest."""
    # Build the canonical config and replace only run-specific IO fields.
    base = build_default_portfolio_backtest_config()
    config = replace(
        base,
        inference_manifest_path=Path(inference_manifest_path),
        output_dir=PORTFOLIO_OUTPUT_DIR,
        feature_db_path=PORTFOLIO_OUTPUT_DIR / "portfolio_backtest.duckdb",
        feature_chunk_dir=PORTFOLIO_OUTPUT_DIR / "feature_chunks",
        feature_manifest_path=PORTFOLIO_OUTPUT_DIR / "feature_manifest.yaml",
        report_title="0428 Clean Benchmark: Execution-Aware Strategy Backtest",
    )

    # Run the full backtest and return the generated markdown report path.
    return run_portfolio_backtest(config)


def publish_html_reports() -> dict[str, str]:
    """Copy final HTML reports into the repository report/0428 directory."""
    # Create the final report directory in the repository.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy the user-facing HTML reports with stable names.
    outputs = {
        "portfolio_backtest_html": REPORT_DIR / "portfolio_backtest_report_0428.html",
        "train_report_html": REPORT_DIR / "train_report_0428.html",
        "test_evaluation_report_html": REPORT_DIR / "test_evaluation_report_0428.html",
    }
    sources = {
        "portfolio_backtest_html": PORTFOLIO_OUTPUT_DIR / "research_report.html",
        "train_report_html": SPLIT_ROOT / "train_report.html",
        "test_evaluation_report_html": SPLIT_ROOT / "test_evaluation_report.html",
    }
    for key, src in sources.items():
        shutil.copy2(Path(src), Path(outputs[key]))

    # Return a serializable mapping for the run summary.
    return {str(key): str(path.as_posix()) for key, path in outputs.items()}


def write_run_summary(inference_manifest_path: Path, portfolio_report_path: Path, html_outputs: dict[str, str]) -> Path:
    """Write the 0428 benchmark summary as YAML."""
    # Assemble the key paths needed to audit this run.
    summary = {
        "run_root": str(RUN_ROOT.as_posix()),
        "split_root": str(SPLIT_ROOT.as_posix()),
        "pipeline_run_root": str(PIPELINE_RUN_ROOT.as_posix()),
        "inference_manifest_path": str(Path(inference_manifest_path).as_posix()),
        "portfolio_output_dir": str(PORTFOLIO_OUTPUT_DIR.as_posix()),
        "portfolio_report_path": str(Path(portfolio_report_path).as_posix()),
        "repo_report_dir": str(REPORT_DIR.as_posix()),
        "html_outputs": dict(html_outputs),
    }

    # Persist the summary in both artifact and final report locations.
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_summary_path = RUN_ROOT / "clean_benchmark_summary.yaml"
    report_summary_path = REPORT_DIR / "clean_benchmark_summary.yaml"
    payload = yaml.safe_dump(summary, allow_unicode=True, sort_keys=False)
    artifact_summary_path.write_text(payload, encoding="utf-8")
    report_summary_path.write_text(payload, encoding="utf-8")
    return artifact_summary_path


def main() -> None:
    """Run the full 0428 clean benchmark workflow."""
    # Run pipeline stages and require a formal inference manifest.
    inference_manifest_path = run_clean_pipeline()

    # Run portfolio backtest and publish the final HTML reports.
    portfolio_report_path = run_clean_portfolio_backtest(inference_manifest_path)
    html_outputs = publish_html_reports()
    summary_path = write_run_summary(inference_manifest_path, portfolio_report_path, html_outputs)
    print(summary_path.as_posix())


if __name__ == "__main__":
    main()
