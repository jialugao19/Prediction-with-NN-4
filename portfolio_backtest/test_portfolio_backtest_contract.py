"""Unit tests for the stable portfolio backtest contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path("/home/maomao/prediction-NN-2")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import portfolio_backtest
from portfolio_backtest.contract import (
    build_input_contract,
    load_chunk_manifest_glob,
    load_inference_manifest,
    validate_required_artifacts,
    write_chunk_manifest,
    write_runtime_contract,
)


def test_write_runtime_contract_emits_stable_yaml(tmp_path: Path):
    """Ensure the runtime contract YAML is written with the expected stable fields."""
    # Write one runtime contract and load it back for inspection.
    contract_path = write_runtime_contract(tmp_path)
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

    # Require the key contract fields to stay pinned.
    assert contract_path.name == "portfolio_backtest_contract.yaml"
    assert payload["input_contract"]["expected_manifest_filename"] == "inference_manifest.yaml"
    assert payload["input_contract"]["expected_inference_split"] == "inference_test"
    assert "strategy_summary.yaml" in list(payload["output_contract"]["required_artifacts"])
    assert "research_report.html" in list(payload["output_contract"]["required_artifacts"])


def test_load_inference_manifest_accepts_canonical_layout(tmp_path: Path):
    """Ensure the input manifest validator accepts the canonical inference layout."""
    # Write one tiny canonical inference manifest and its chunk payload.
    manifest_dir = Path(tmp_path) / "inference_test" / "iter_1"
    chunk_dir = manifest_dir / "predict_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": [0.1], "code": [1], "date": [260101], "time": [93000]}).to_parquet(chunk_dir / "part_000000.parquet", index=False)
    manifest_path = manifest_dir / "inference_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "format": "parquet_chunks",
                "columns": list(build_input_contract().required_manifest_columns),
                "chunk_files": ["predict_chunks/part_000000.parquet"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    # Require the loader to accept the canonical path and schema.
    manifest = load_inference_manifest(manifest_path)
    assert list(manifest["columns"]) == ["prediction", "code", "date", "time"]


def test_validate_required_artifacts_checks_contract(tmp_path: Path):
    """Ensure the output validator succeeds only when every contract artifact exists."""
    # Materialize the required artifact names as empty files.
    contract_path = write_runtime_contract(tmp_path)
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    for artifact_name in list(payload["output_contract"]["required_artifacts"]):
        artifact_path = Path(tmp_path) / str(artifact_name)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if not artifact_path.exists():
            artifact_path.write_text("", encoding="utf-8")

    # Require the validator to accept the complete artifact set.
    validate_required_artifacts(tmp_path)


def test_generic_chunk_manifest_glob_supports_feature_outputs(tmp_path: Path):
    """Ensure the generic chunk manifest reader works for non-inference feature chunks."""
    # Materialize one fake feature chunk and its manifest.
    manifest_dir = Path(tmp_path)
    chunk_dir = manifest_dir / "feature_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": [0.1], "code": [1], "date": [260101], "time": [93000]}).to_parquet(chunk_dir / "part_000000.parquet", index=False)
    manifest_path = manifest_dir / "feature_manifest.yaml"
    write_chunk_manifest(manifest_path, [chunk_dir / "part_000000.parquet"], "feature_chunks", ["prediction", "code", "date", "time"])

    # Require the generic manifest reader to recover the parquet glob without inference validation.
    parquet_glob = load_chunk_manifest_glob(manifest_path)
    assert parquet_glob.endswith("feature_chunks/*.parquet")


def test_package_exports_stable_entrypoints():
    """Ensure the package root exports the stable contract and entrypoint symbols."""
    # Check the package-level public API so callers can import from one stable root.
    assert callable(portfolio_backtest.run_portfolio_backtest)
    assert callable(portfolio_backtest.load_inference_manifest)
    assert isinstance(portfolio_backtest.INFERENCE_MANIFEST_COLUMNS, list)


def test_package_code_does_not_import_backtest2():
    """Ensure the canonical package code does not depend on the compatibility wrapper."""
    # Scan the package python sources once.
    package_dir = REPO_ROOT / "portfolio_backtest"
    python_files = [path for path in sorted(package_dir.glob("*.py")) if not path.name.startswith("test_")]

    # Require every implementation module to avoid importing backtest2.
    for path in list(python_files):
        text = path.read_text(encoding="utf-8")
        assert "import backtest2" not in text
        assert "from backtest2" not in text


def test_canonical_html_script_lives_under_repo_and_writes_to_data_cache():
    """Ensure the canonical HTML generator stays in-repo and targets /data-cache/nn outputs."""
    # Resolve the canonical script path once.
    script_path = REPO_ROOT / "portfolio_backtest" / "scripts" / "generate_portfolio_backtest_self_constrained_html.py"

    # Read the script and pin the canonical output root.
    text = script_path.read_text(encoding="utf-8")
    assert script_path.exists()
    assert "/data-cache/nn/0424" in text
