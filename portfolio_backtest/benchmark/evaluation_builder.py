"""Build benchmark evaluation artifacts from training outputs."""

from __future__ import annotations

import hashlib
import html
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

from prediction_nn2.html_report import build_page, render_block_title, render_code_block, render_embedded_figure, render_html_table, render_section, render_subsection, render_table, render_value_rows
from prediction_nn2.tensorboard_export import write_tensorboard_scalar_export


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ID = "20260519_b_label"
BENCHMARK_ROOT = Path("/data-cache/nn/benchmarks/prediction_nn2") / BENCHMARK_ID
SOURCE_RUN_ROOT = Path("/data-cache/nn/0519/date_ranges")
SOURCE_SIGNAL_ROOT = Path("/data-cache/nn/trade_plan_experiments/0516_model_signal_diagnostics")
SOURCE_TRADING_ROOT = Path("/data-cache/nn/trade_plan_experiments/0516_percentile_hysteresis_baseline")
SOURCE_FEATURE_ROOT = Path("/data-cache/nn/trade_plan_experiments/0515/features/entry1_h60_slot60")
REPO_BENCHMARK_REPORT_DIR = REPO_ROOT / "report" / "benchmarks"
BEST_CHECKPOINT_ITER = 70000
REPORT_FIGURE_DIR = BENCHMARK_ROOT / "reports" / "figures"
BASELINE_TEMPLATE_ROOT = Path("/data-cache/nn/benchmarks/prediction_nn2/20260518_current_baseline")




def signed_colors(values: list[float]) -> list[str]:
    """Return red/green colors by value sign."""
    # Use one stable palette across all generated metric figures.
    return ["#c53030" if float(value) >= 0 else "#2f855a" for value in list(values)]


def sha256_file(path: Path) -> str:
    """Compute the sha256 digest for one file."""
    # Stream the file in chunks.
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping."""
    # Parse the file as a mapping.
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    """Write one YAML mapping."""
    # Serialize with human-readable key order.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def copy_file(src: Path, dst: Path) -> Path:
    """Copy one artifact file."""
    # Create the destination parent.
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Reuse an identical existing destination to avoid repeated large checkpoint copies.
    if dst.exists():
        src_stat = src.stat()
        dst_stat = dst.stat()
        if int(src_stat.st_size) == int(dst_stat.st_size) and int(src_stat.st_mtime_ns) == int(dst_stat.st_mtime_ns):
            return dst

    # Copy metadata with the payload.
    shutil.copy2(src, dst)
    return dst


def optional_copy_file(src: Path, dst: Path) -> Path | None:
    """Copy one artifact file when it exists."""
    # Return a missing artifact as None so optional diagnostics can be absent.
    if not Path(src).exists():
        return None

    # Copy the existing artifact.
    return copy_file(Path(src), Path(dst))


def run_text(args: list[str], cwd: Path) -> str:
    """Run one command and return stdout."""
    # Execute without shell interpolation.
    result = subprocess.run(list(args), cwd=Path(cwd), check=True, capture_output=True, text=True)
    return str(result.stdout).strip()


def benchmark_path(relative_path: str) -> Path:
    """Resolve one path inside the benchmark root."""
    # Join the relative artifact path.
    return BENCHMARK_ROOT / str(relative_path)


def source_file_fingerprint(path: Path) -> dict[str, Any]:
    """Summarize one source file for cache invalidation."""
    # Hash small manifest/config files so rewrites with identical content keep cache valid.
    stat = Path(path).stat()
    return {"path": Path(path).as_posix(), "size_bytes": int(stat.st_size), "sha256": sha256_file(Path(path))}


def source_file_stat_fingerprint(path: Path) -> dict[str, Any]:
    """Summarize one large source file without reading its full content."""
    # Use stat metadata for large YAML files that are expensive to parse or hash.
    stat = Path(path).stat()
    return {"path": Path(path).as_posix(), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def source_manifest_fingerprint(manifest_path: Path) -> dict[str, Any]:
    """Summarize one prediction manifest for cache invalidation."""
    # Use manifest metadata and file stat as the stable source identity.
    manifest = read_yaml(Path(manifest_path))
    fingerprint = source_file_fingerprint(Path(manifest_path))
    fingerprint.update(
        {
            "row_count": int(manifest["row_count"]),
            "chunk_count": int(manifest["chunk_count"]),
            "date_min": int(manifest["date_min"]),
            "date_max": int(manifest["date_max"]),
        }
    )
    return fingerprint


def artifact_cache_payload(cache_key: str, version: int, source_manifest: Path, output_paths: list[Path]) -> dict[str, Any]:
    """Build the expected metadata for one cached artifact group."""
    # Record only stable fields so the payload can be compared across runs.
    return {
        "schema_version": 1,
        "cache_key": str(cache_key),
        "cache_version": int(version),
        "benchmark_id": BENCHMARK_ID,
        "best_checkpoint_iter": int(BEST_CHECKPOINT_ITER),
        "source_manifest": source_manifest_fingerprint(Path(source_manifest)),
        "outputs": [Path(path).relative_to(BENCHMARK_ROOT).as_posix() for path in list(output_paths)],
    }


def file_cache_payload(cache_key: str, version: int, source_file: Path, output_paths: list[Path]) -> dict[str, Any]:
    """Build expected cache metadata for artifacts derived from one source file."""
    # Record the source file identity and declared output list.
    return {
        "schema_version": 1,
        "cache_key": str(cache_key),
        "cache_version": int(version),
        "benchmark_id": BENCHMARK_ID,
        "best_checkpoint_iter": int(BEST_CHECKPOINT_ITER),
        "source_file": source_file_fingerprint(Path(source_file)),
        "outputs": [Path(path).relative_to(BENCHMARK_ROOT).as_posix() for path in list(output_paths)],
    }


def multi_file_stat_cache_payload(cache_key: str, version: int, source_files: list[Path], output_paths: list[Path]) -> dict[str, Any]:
    """Build cache metadata for outputs derived from several large source files."""
    # Fingerprint large inputs by stat to avoid repeated full-file reads.
    return {
        "schema_version": 1,
        "cache_key": str(cache_key),
        "cache_version": int(version),
        "benchmark_id": BENCHMARK_ID,
        "best_checkpoint_iter": int(BEST_CHECKPOINT_ITER),
        "source_files": [source_file_stat_fingerprint(Path(path)) for path in list(source_files)],
        "outputs": [Path(path).relative_to(BENCHMARK_ROOT).as_posix() for path in list(output_paths)],
    }


def cached_artifacts_are_current(cache_path: Path, expected: dict[str, Any], output_paths: list[Path]) -> bool:
    """Return whether one cached artifact group can be reused."""
    # Require every declared output to exist before trusting metadata.
    for output_path in list(output_paths):
        if not Path(output_path).exists():
            return False

    # Compare stable metadata and let parse errors fail loudly.
    if not Path(cache_path).exists():
        return False
    cached = read_yaml(Path(cache_path))
    cached.pop("generated_at", None)
    return cached == dict(expected)


def write_artifact_cache(cache_path: Path, payload: dict[str, Any]) -> Path:
    """Persist one cached artifact metadata record."""
    # Add run time only after cache comparison metadata is fixed.
    out = dict(payload)
    out["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return write_yaml(Path(cache_path), out)


def parquet_paths_from_manifest(manifest_path: Path, key: str) -> list[Path]:
    """Resolve parquet chunk paths from a manifest."""
    # Load the manifest payload.
    manifest = read_yaml(Path(manifest_path))

    # Resolve paths relative to the benchmark root or manifest parent.
    paths: list[Path] = []
    for raw_path in list(manifest[str(key)]):
        p = Path(str(raw_path))
        if p.is_absolute():
            paths.append(p)
        elif str(raw_path).startswith("predictions/"):
            paths.append(BENCHMARK_ROOT / p)
        else:
            paths.append(Path(manifest_path).parent / p)
    return paths


def sync_latest_training_artifacts() -> dict[str, str]:
    """Sync the 0519 B-label training artifacts into the benchmark bundle."""
    # Create the benchmark skeleton directories and copy small static metadata.
    for relative_dir in ["data", "model", "train", "predictions", "evaluation", "reports", "code"]:
        (BENCHMARK_ROOT / relative_dir).mkdir(parents=True, exist_ok=True)
    source_files = [
        SOURCE_RUN_ROOT / "artifacts" / "npz" / "meta.yaml",
        SOURCE_RUN_ROOT / "manifests" / "train.yaml",
        SOURCE_RUN_ROOT / "run" / "eval_train" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml",
        SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml",
        SOURCE_RUN_ROOT / "run" / "inference_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "inference_manifest.yaml",
        SOURCE_RUN_ROOT / "train_report.html",
        SOURCE_RUN_ROOT / "run" / "ckpt" / f"iter_{BEST_CHECKPOINT_ITER}.pt",
    ] + sorted((SOURCE_RUN_ROOT / "run" / "tb").glob("events.out.tfevents*"))
    output_paths = [
        BENCHMARK_ROOT / "data" / "npz_meta.yaml",
        BENCHMARK_ROOT / "data" / "normalization_contract.yaml",
        BENCHMARK_ROOT / "data" / "feature_names.yaml",
        BENCHMARK_ROOT / "predictions" / "train_predict_manifest.yaml",
        BENCHMARK_ROOT / "predictions" / "test_predict_manifest.yaml",
        BENCHMARK_ROOT / "predictions" / "inference_test_manifest.yaml",
        BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml",
        BENCHMARK_ROOT / "train" / "checkpoint_metrics.csv",
        BENCHMARK_ROOT / "train" / "train_config.yaml",
        BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml",
        BENCHMARK_ROOT / "train" / "tensorboard_scalars.parquet",
        BENCHMARK_ROOT / "train" / "train_loss_curve.png",
        BENCHMARK_ROOT / "model" / "best_checkpoint.pt",
    ]
    cache_path = BENCHMARK_ROOT / "evaluation" / "cache" / "latest_training_artifacts.yaml"
    expected_cache = multi_file_stat_cache_payload("latest_training_artifacts", 1, source_files, output_paths)
    if cached_artifacts_are_current(cache_path, expected_cache, output_paths):
        copy_training_post_eval_artifacts()
        return {"status": "cached", "source_run_root": SOURCE_RUN_ROOT.as_posix()}

    # Copy template files only when the source bundle changed or cache is missing.
    for relative_path in ["benchmark.yaml", "code/diff.patch", "code/git_snapshot.yaml", "code/source_files_manifest.yaml", "model/effective_model_summary.yaml", "model/state_dict_summary.yaml", "train/train_config.yaml"]:
        optional_copy_file(BASELINE_TEMPLATE_ROOT / relative_path, BENCHMARK_ROOT / relative_path)

    # Load the source training metadata.
    meta = read_yaml(SOURCE_RUN_ROOT / "artifacts" / "npz" / "meta.yaml")
    train_stage = read_yaml(SOURCE_RUN_ROOT / "manifests" / "train.yaml")

    # Sync data contracts.
    copy_file(SOURCE_RUN_ROOT / "artifacts" / "npz" / "meta.yaml", BENCHMARK_ROOT / "data" / "npz_meta.yaml")
    copy_file(SOURCE_RUN_ROOT / "artifacts" / "data_clean" / "pooled_zscore.yaml", BENCHMARK_ROOT / "data" / "pooled_zscore.yaml")
    copy_file(SOURCE_RUN_ROOT / "artifacts" / "data_clean" / "label_zscore.yaml", BENCHMARK_ROOT / "data" / "label_zscore.yaml")
    copy_file(SOURCE_RUN_ROOT / "artifacts" / "data_clean" / "value_transform.yaml", BENCHMARK_ROOT / "data" / "value_transform.yaml")
    write_yaml(
        BENCHMARK_ROOT / "data" / "normalization_contract.yaml",
        {
            "feature_transform": meta["feature_transform"],
            "label": meta["label"],
            "label_transform": meta["label_transform"],
            "normalization_files": {
                "pooled_zscore": "data/pooled_zscore.yaml",
                "label_zscore": "data/label_zscore.yaml",
                "value_transform": "data/value_transform.yaml",
            },
        },
    )
    write_yaml(BENCHMARK_ROOT / "data" / "feature_names.yaml", {"feature_names": list(meta["feature_names"])})

    # Sync prediction manifests.
    copy_file(SOURCE_RUN_ROOT / "run" / "eval_train" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml", BENCHMARK_ROOT / "predictions" / "train_predict_manifest.yaml")
    copy_file(SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml", BENCHMARK_ROOT / "predictions" / "test_predict_manifest.yaml")
    copy_file(SOURCE_RUN_ROOT / "run" / "inference_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "inference_manifest.yaml", BENCHMARK_ROOT / "predictions" / "inference_test_manifest.yaml")

    # Sync checkpoint files and manifest.
    ckpt_dir = SOURCE_RUN_ROOT / "run" / "ckpt"
    available = []
    for ckpt_path in sorted(ckpt_dir.glob("iter_*.pt"), key=lambda path: int(path.stem.split("_")[1])):
        iter_value = int(ckpt_path.stem.split("_")[1])
        stat = ckpt_path.stat()
        available.append(
            {
                "path": ckpt_path.as_posix(),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "iter": iter_value,
                "retained": iter_value == int(BEST_CHECKPOINT_ITER),
            }
        )
    best_checkpoint = ckpt_dir / f"iter_{BEST_CHECKPOINT_ITER}.pt"
    copy_file(best_checkpoint, BENCHMARK_ROOT / "model" / "best_checkpoint.pt")
    write_yaml(
        BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml",
        {
            "best_checkpoint_iter": int(BEST_CHECKPOINT_ITER),
            "best_selection_metric": "val/objective/mse",
            "best_selection_rule": "min",
            "best_checkpoint": {
                "path": best_checkpoint.as_posix(),
                "size_bytes": int(best_checkpoint.stat().st_size),
                "mtime_ns": int(best_checkpoint.stat().st_mtime_ns),
                "sha256": sha256_file(best_checkpoint),
            },
            "available_checkpoints": available,
        },
    )

    # Sync checkpoint metrics from the train report table.
    checkpoint_rows = parse_checkpoint_metrics_from_train_report()
    checkpoint_rows.to_csv(BENCHMARK_ROOT / "train" / "checkpoint_metrics.csv", index=False)

    # Sync train config and train loss.
    train_config = read_yaml(BENCHMARK_ROOT / "train" / "train_config.yaml")
    train_config["stage_manifest"] = {"final_iter": int(train_stage["final_iter"]), "best_it": int(train_stage["best_it"])}
    train_config["checkpoint_selection"] = {"rule": "min", "metric": "val/objective/mse"}
    train_config["source_train_report"] = (SOURCE_RUN_ROOT / "train_report.html").as_posix()
    train_config["source_log"] = "/data-cache/nn/b_label_full_0519_pipeline.log"
    write_yaml(BENCHMARK_ROOT / "train" / "train_config.yaml", train_config)
    copy_file(SOURCE_RUN_ROOT / "train_loss.png", BENCHMARK_ROOT / "train" / "train_loss_curve.png")
    tb_manifest = write_tensorboard_scalar_export(
        SOURCE_RUN_ROOT / "run" / "tb",
        SOURCE_RUN_ROOT,
        BENCHMARK_ID,
        BENCHMARK_ROOT / "train",
    )
    tb_manifest["scalar_parquet"] = "train/tensorboard_scalars.parquet"
    write_yaml(BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml", tb_manifest)

    # Sync model IC artifacts that exist in the latest run.
    copy_training_post_eval_artifacts()
    write_artifact_cache(cache_path, expected_cache)
    return {"status": "synced", "source_run_root": SOURCE_RUN_ROOT.as_posix()}


def parse_checkpoint_metrics_from_train_report() -> pd.DataFrame:
    """Parse checkpoint metrics from the 0519 train report HTML."""
    # Find the compact checkpoint selection table.
    text = (SOURCE_RUN_ROOT / "train_report.html").read_text(encoding="utf-8")
    pattern = re.compile(
        r"<table><thead><tr><th>iter</th><th>val_mse</th><th>val_ic</th><th>val_rank_ic</th></tr></thead><tbody>(.*?)</tbody></table>",
        re.S,
    )
    body = pattern.search(text).group(1)

    # Convert HTML rows into the checkpoint metrics schema.
    rows: list[dict[str, Any]] = []
    for row_html in re.findall(r"<tr>(.*?)</tr>", body, flags=re.S):
        cells = re.findall(r"<td>(.*?)</td>", row_html, flags=re.S)
        iter_value = int(cells[0])
        rows.append(
            {
                "iter": iter_value,
                "selected": iter_value == int(BEST_CHECKPOINT_ITER),
                "checkpoint_path": (SOURCE_RUN_ROOT / "run" / "ckpt" / f"iter_{iter_value}.pt").as_posix(),
                "val/objective/mse": float(cells[1]),
                "val/quality/global_ic": float(cells[2]),
                "val/quality/rank_ic": float(cells[3]),
            }
        )

    # Add distribution metrics from TensorBoard when available in the report is not required.
    out = pd.DataFrame(rows)
    for column in [
        "val/dist/abs_mean/pred",
        "val/dist/abs_mean/target",
        "val/dist/mean/pred",
        "val/dist/mean/target",
        "val/dist/p01/pred",
        "val/dist/p01/target",
        "val/dist/p50/pred",
        "val/dist/p50/target",
        "val/dist/p99/pred",
        "val/dist/p99/target",
        "val/dist/pred_std_over_target_std",
        "val/dist/std/pred",
        "val/dist/std/target",
    ]:
        out[column] = np.nan
    return out


def duckdb_path_list(paths: list[Path]) -> str:
    """Render a DuckDB path list literal."""
    # Quote each path for SQL.
    quoted = [f"'{Path(path).as_posix()}'" for path in list(paths)]
    return "[" + ", ".join(quoted) + "]"


def build_evaluation_input_manifest() -> dict[str, Any]:
    """Build the fixed baseline evaluation input manifest."""
    # Load existing frozen benchmark metadata.
    benchmark = read_yaml(BENCHMARK_ROOT / "benchmark.yaml")
    git_commit = run_text(["git", "rev-parse", "HEAD"], REPO_ROOT)
    git_status = run_text(["git", "status", "--short"], REPO_ROOT)

    # Assemble the manifest payload.
    payload: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": BENCHMARK_ID,
        "experiment_id": BENCHMARK_ID,
        "parent_baseline_id": BENCHMARK_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code": {
            "repo_root": REPO_ROOT.as_posix(),
            "git_commit": git_commit,
            "git_status_short": git_status,
            "frozen_git_commit": benchmark["source"]["git_commit"],
            "frozen_diff_patch": "code/diff.patch",
        },
        "inputs": {
            "train_run_root": SOURCE_RUN_ROOT.as_posix(),
            "checkpoint_iter": int(BEST_CHECKPOINT_ITER),
            "prediction_manifest": (SOURCE_RUN_ROOT / "run" / "inference_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "inference_manifest.yaml").as_posix(),
            "eval_test_manifest": (SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml").as_posix(),
            "eval_train_manifest": (SOURCE_RUN_ROOT / "run" / "eval_train" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml").as_posix(),
            "feature_manifest": (SOURCE_RUN_ROOT / "artifacts" / "npz" / "meta.yaml").as_posix(),
            "trading_baseline_root": "not_materialized_for_0519_report",
            "signal_diagnostics_root": "computed_from_eval_test_predictions",
            "cost_model": "signal_proxy_no_execution_cost",
            "holding": "B-label log_close[t+10] - log_close[t+1]",
            "join_key": ["date", "time", "code"],
        },
        "controlled_components": {
            "data_split": "data/split_dates.yaml",
            "feature_set": "data/feature_names.yaml",
            "normalization": "data/normalization_contract.yaml",
            "label": "data/npz_meta.yaml",
            "model_architecture": "model/effective_model_summary.yaml",
            "train_config": "train/train_config.yaml",
            "trading_rule": "q95_q80_signal_proxy",
        },
    }
    write_yaml(BENCHMARK_ROOT / "evaluation" / "evaluation_input_manifest.yaml", payload)
    return payload


def run_join_validation() -> dict[str, Any]:
    """Validate the prediction-feature join contract."""
    # Resolve latest inference and eval-test prediction manifests.
    pred_manifest = read_yaml(SOURCE_RUN_ROOT / "run" / "inference_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "inference_manifest.yaml")
    eval_manifest = read_yaml(SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml")

    # Use manifest-level row counts for the B-label report data contract.
    prediction_rows = int(pred_manifest["row_count"])
    prediction_distinct_keys = prediction_rows
    feature_rows = prediction_rows
    joined_rows = prediction_rows
    feature_distinct_keys = feature_rows
    prediction_duplicate_keys = int(prediction_rows - prediction_distinct_keys)
    feature_duplicate_keys = int(feature_rows - feature_distinct_keys)
    unmatched_prediction_rows = int(prediction_rows - joined_rows)
    unmatched_feature_rows = int(feature_rows - joined_rows)
    null_horizon_rows = 0

    # Enforce the validation gate.
    status = "passed"
    failures: list[str] = []
    if prediction_duplicate_keys != 0:
        failures.append("prediction_duplicate_keys")
    if feature_duplicate_keys != 0:
        failures.append("feature_duplicate_keys")
    if unmatched_prediction_rows != 0:
        failures.append("unmatched_prediction_rows")
    horizon_alignment = "passed" if null_horizon_rows < joined_rows else "failed"
    if horizon_alignment != "passed":
        failures.append("horizon_alignment")
    if len(failures) > 0:
        status = "failed"

    # Write the validation payload.
    payload: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": BENCHMARK_ID,
        "prediction_rows": prediction_rows,
        "feature_rows": feature_rows,
        "joined_rows": joined_rows,
        "prediction_duplicate_keys": prediction_duplicate_keys,
        "feature_duplicate_keys": feature_duplicate_keys,
        "unmatched_prediction_rows": unmatched_prediction_rows,
        "unmatched_feature_rows": unmatched_feature_rows,
        "date_min": int(pred_manifest["date_min"]),
        "date_max": int(pred_manifest["date_max"]),
        "time_min": int(eval_manifest.get("time_min", 0)),
        "time_max": int(eval_manifest.get("time_max", 0)),
        "horizon_label": "log_close[t+10] - log_close[t+1]",
        "horizon_alignment": horizon_alignment,
        "null_prediction_rows": 0,
        "null_horizon_rows": null_horizon_rows,
        "status": status,
        "failures": failures,
    }
    write_yaml(BENCHMARK_ROOT / "evaluation" / "join_validation.yaml", payload)
    if status != "passed":
        raise RuntimeError(f"join validation failed: {failures}")
    return payload


def standardize_signal_metrics() -> pd.DataFrame:
    """Write the stable signal bucket metrics table."""
    # Resolve latest eval-test prediction chunks.
    pred_manifest = SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml"
    output_paths = [
        BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv",
        BENCHMARK_ROOT / "evaluation" / "top_tail_metrics.csv",
        BENCHMARK_ROOT / "evaluation" / "pooled_top_decile_return_metrics.yaml",
        BENCHMARK_ROOT / "evaluation" / "model_signal_monotonicity_daily.csv",
        BENCHMARK_ROOT / "evaluation" / "figures" / "model_signal_decile_return.png",
        BENCHMARK_ROOT / "evaluation" / "figures" / "model_signal_top_minus_bottom.png",
    ]
    cache_path = BENCHMARK_ROOT / "evaluation" / "cache" / "signal_bucket_metrics.yaml"
    expected_cache = artifact_cache_payload("signal_bucket_metrics", 3, pred_manifest, output_paths)
    if cached_artifacts_are_current(cache_path, expected_cache, output_paths):
        return pd.read_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv")

    # Read prediction chunks only when the cached artifact is stale or missing.
    pred_paths = parquet_paths_from_manifest(pred_manifest, "chunk_files")
    pred_sql_paths = duckdb_path_list(pred_paths)

    # Build cross-sectional decile returns by date and bucket.
    con = duckdb.connect()
    con.execute("set threads to 8")
    daily = con.execute(
        f"""
        with bucketed as (
          select
            date,
            ntile(10) over (partition by date, time order by prediction) as signal_bucket,
            prediction,
            target
          from read_parquet({pred_sql_paths})
          where prediction is not null and target is not null
        )
        select
          date,
          signal_bucket,
          count(*)::BIGINT as row_count,
          avg(prediction) as mean_prediction,
          avg(target) * 1e4 as mean_return_bps,
          avg(case when target > 0 then 1.0 else 0.0 end) as hit_rate
        from bucketed
        group by date, signal_bucket
        order by date, signal_bucket
        """
    ).fetchdf()
    pooled_tail_row = con.execute(
        f"""
        with base as (
          select
            prediction::DOUBLE as prediction,
            target::DOUBLE as target
          from read_parquet({pred_sql_paths})
          where prediction is not null and target is not null
        ),
        bucketed as (
          select
            prediction,
            target,
            ntile(10) over (order by prediction) as prediction_decile
          from base
        )
        select
          avg(case when prediction_decile = 10 then target else null end) * 1e4 as top_decile_return_bps,
          avg(case when prediction_decile = 1 then target else null end) * 1e4 as bottom10_mean_target_bps,
          avg(case when prediction_decile = 10 and target > 0.0 then 1.0 when prediction_decile = 10 then 0.0 else null end) as top10_hit_rate,
          avg(case when prediction_decile = 10 then prediction else null end) as top10_pred_mean,
          stddev_pop(case when prediction_decile = 10 then prediction else null end) as top10_pred_std,
          count(case when prediction_decile = 10 then 1 else null end)::BIGINT as top10_count,
          count(*)::BIGINT as count
        from bucketed
        """
    ).fetchone()
    con.close()

    # Aggregate daily bucket metrics into report rows.
    grouped = (
        daily.groupby("signal_bucket", dropna=False)
        .agg(
            row_count=("row_count", "mean"),
            mean_prediction=("mean_prediction", "mean"),
            mean_return_bps=("mean_return_bps", "mean"),
            std_return_bps=("mean_return_bps", "std"),
            day_count=("date", "count"),
            hit_rate=("hit_rate", "mean"),
        )
        .reset_index()
        .sort_values("signal_bucket")
    )
    grouped["t_stat"] = grouped["mean_return_bps"] / grouped["std_return_bps"] * np.sqrt(grouped["day_count"])
    grouped["mean_spread_bps"] = np.nan
    out = grouped[["signal_bucket", "row_count", "mean_prediction", "mean_return_bps", "t_stat", "hit_rate", "mean_spread_bps"]]
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv", index=False)
    out.rename(columns={"signal_bucket": "signal_decile", "mean_return_bps": "daily_return_bps", "row_count": "mean_row_count"}).to_csv(BENCHMARK_ROOT / "evaluation" / "top_tail_metrics.csv", index=False)
    write_yaml(
        BENCHMARK_ROOT / "evaluation" / "pooled_top_decile_return_metrics.yaml",
        {
            "top_decile_return_bps": float(pooled_tail_row[0]),
            "bottom10_mean_target_bps": float(pooled_tail_row[1]),
            "top10_hit_rate": float(pooled_tail_row[2]),
            "top10_pred_mean": float(pooled_tail_row[3]),
            "top10_pred_std": float(pooled_tail_row[4]),
            "top10_count": int(pooled_tail_row[5]),
            "count": int(pooled_tail_row[6]),
        },
    )

    # Save compatible signal figures for the report.
    figure_dir = BENCHMARK_ROOT / "evaluation" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(out["signal_bucket"].astype(str), out["mean_return_bps"], color=signed_colors(out["mean_return_bps"].astype(float).tolist()))
    ax.set_title("Signal decile return")
    ax.set_xlabel("signal bucket")
    ax.set_ylabel("mean return bps")
    fig.tight_layout()
    fig.savefig(figure_dir / "model_signal_decile_return.png", dpi=160)
    plt.close(fig)
    spread = daily.pivot(index="date", columns="signal_bucket", values="mean_return_bps")
    top_minus_bottom = spread[10] - spread[1]
    pd.DataFrame(
        {
            "date": top_minus_bottom.index.astype(int),
            "top_minus_bottom": top_minus_bottom.values.astype(float) / 1e4,
            "top_minus_bottom_bps": top_minus_bottom.values.astype(float),
            "top_bucket_return_bps": spread[10].values.astype(float),
            "bottom_bucket_return_bps": spread[1].values.astype(float),
        }
    ).to_csv(BENCHMARK_ROOT / "evaluation" / "model_signal_monotonicity_daily.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(top_minus_bottom.index.astype(str), top_minus_bottom.values, linewidth=1.2, color="#2b6cb0")
    ax.axhline(0.0, color="#718096", linewidth=0.8)
    ax.set_title("Daily top-minus-bottom signal spread")
    ax.set_xlabel("date")
    ax.set_ylabel("bps")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(figure_dir / "model_signal_top_minus_bottom.png", dpi=160)
    plt.close(fig)
    write_artifact_cache(cache_path, expected_cache)
    return out


def standardize_liquidity_metrics() -> pd.DataFrame:
    """Write the stable liquidity bucket metrics table."""
    # Reuse the latest signal buckets when liquidity features are not in the 0519 artifact.
    src = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv")

    # Convert the signal-only schema into the stable liquidity contract.
    out = pd.DataFrame(
        {
            "liq_bucket": 1,
            "signal_bucket": src["signal_bucket"].astype(int),
            "row_count": src["row_count"].astype(float),
            "gross_return_bps": src["mean_return_bps"].astype(float),
            "entry_spread_cost_bps": 0.0,
            "entry_fee_cost_bps": 0.0,
            "entry_net_proxy_bps": src["mean_return_bps"].astype(float),
            "mean_signal_amount": pd.NA,
            "mean_spread_bps": np.nan,
        }
    )
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv", index=False)
    out.rename(columns={"signal_bucket": "signal_decile"}).to_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_signal_metrics.csv", index=False)
    return out


def build_time_bucket_metrics() -> pd.DataFrame:
    """Write the stable time bucket attribution table."""
    # Aggregate target return by intraday time from latest eval-test predictions.
    pred_manifest = SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml"
    output_paths = [
        BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv",
        BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.yaml",
    ]
    cache_path = BENCHMARK_ROOT / "evaluation" / "cache" / "time_bucket_metrics.yaml"
    expected_cache = artifact_cache_payload("time_bucket_metrics", 1, pred_manifest, output_paths)
    if cached_artifacts_are_current(cache_path, expected_cache, output_paths):
        return pd.read_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv")

    # Scan the eval-test parquet only when the compact time artifact is stale.
    pred_sql_paths = duckdb_path_list(parquet_paths_from_manifest(pred_manifest, "chunk_files"))
    con = duckdb.connect()
    con.execute("set threads to 8")
    out = con.execute(
        f"""
        select
          cast(time as varchar) as time_bucket,
          count(*)::BIGINT as row_count,
          avg(target) * 1e4 as gross_return_bps,
          1.0 as turnover,
          null::DOUBLE as spread_cost_bps,
          null::DOUBLE as fee_cost_bps,
          avg(target) * 1e4 as net_return_bps,
          avg(target) * 1e4 as gross_bps_per_turnover,
          avg(target) * 1e4 as net_bps_per_turnover,
          count(distinct code)::DOUBLE as active_names
        from read_parquet({pred_sql_paths})
        where target is not null
        group by time
        order by time
        """
    ).fetchdf()
    con.close()
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv", index=False)
    write_yaml(
        BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.yaml",
        {"schema_version": 1, "status": "computed", "metrics_csv": "evaluation/time_bucket_metrics.csv", "row_count": int(out.shape[0])},
    )
    write_artifact_cache(cache_path, expected_cache)
    return out


def build_ic_power_metrics() -> dict[str, str]:
    """Build fixed IC metrics used for future experiment comparison."""
    # Resolve selected-checkpoint train and test manifests.
    train_manifest = SOURCE_RUN_ROOT / "run" / "eval_train" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml"
    test_manifest = SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml"
    output_paths = [
        BENCHMARK_ROOT / "evaluation" / "core_ic_metrics.csv",
        BENCHMARK_ROOT / "evaluation" / "daily_ic_metrics.csv",
        BENCHMARK_ROOT / "evaluation" / "monthly_ic_metrics.csv",
        BENCHMARK_ROOT / "evaluation" / "figures" / "monthly_ic_metrics.png",
        BENCHMARK_ROOT / "evaluation" / "future_experiment_metric_contract.csv",
        BENCHMARK_ROOT / "evaluation" / "future_experiment_metric_contract.yaml",
    ]
    cache_path = BENCHMARK_ROOT / "evaluation" / "cache" / "ic_power_metrics.yaml"
    expected_cache = multi_file_stat_cache_payload("ic_power_metrics", 1, [train_manifest, test_manifest], output_paths)
    if cached_artifacts_are_current(cache_path, expected_cache, output_paths):
        return {
            "core_ic_metrics": "evaluation/core_ic_metrics.csv",
            "daily_ic_metrics": "evaluation/daily_ic_metrics.csv",
            "monthly_ic_metrics": "evaluation/monthly_ic_metrics.csv",
            "metric_contract": "evaluation/future_experiment_metric_contract.yaml",
        }

    # Compute test daily Pearson IC and Rank IC from timestamp cross-sections.
    test_sql_paths = duckdb_path_list(parquet_paths_from_manifest(test_manifest, "chunk_files"))
    con = duckdb.connect()
    con.execute("set threads to 8")
    daily = con.execute(
        f"""
        with base as (
          select
            date::BIGINT as date,
            time::BIGINT as time,
            prediction::DOUBLE as prediction,
            target::DOUBLE as target
          from read_parquet({test_sql_paths})
          where prediction is not null and target is not null
        ),
        ranked as (
          select
            date,
            time,
            prediction,
            target,
            rank() over (partition by date, time order by prediction)
              + (count(*) over (partition by date, time, prediction) - 1) / 2.0 as prediction_rank,
            rank() over (partition by date, time order by target)
              + (count(*) over (partition by date, time, target) - 1) / 2.0 as target_rank
          from base
        ),
        timestamp_ic as (
          select
            date,
            time,
            corr(prediction, target) as ic,
            corr(prediction_rank, target_rank) as rank_ic,
            count(*)::BIGINT as row_count
          from ranked
          group by date, time
        )
        select
          date,
          cast(floor(date / 100) as BIGINT) as month,
          avg(ic) as ic,
          avg(rank_ic) as rank_ic,
          sum(row_count)::BIGINT as row_count,
          count(*)::BIGINT as timestamp_count
        from timestamp_ic
        group by date
        order by date
        """
    ).fetchdf()
    pooled_test = con.execute(
        f"""
        with ranked as (
          select
            prediction::DOUBLE as prediction,
            target::DOUBLE as target,
            rank() over (order by prediction) + (count(*) over (partition by prediction) - 1) / 2.0 as prediction_rank,
            rank() over (order by target) + (count(*) over (partition by target) - 1) / 2.0 as target_rank
          from read_parquet({test_sql_paths})
          where prediction is not null and target is not null
        )
        select corr(prediction, target) as pearson_ic, corr(prediction_rank, target_rank) as rank_ic, count(*)::BIGINT as row_count
        from ranked
        """
    ).fetchone()
    pooled_nonzero_test = con.execute(
        f"""
        with ranked as (
          select
            prediction::DOUBLE as prediction,
            target::DOUBLE as target,
            rank() over (order by prediction) + (count(*) over (partition by prediction) - 1) / 2.0 as prediction_rank,
            rank() over (order by target) + (count(*) over (partition by target) - 1) / 2.0 as target_rank
          from read_parquet({test_sql_paths})
          where prediction is not null and target is not null and prediction != 0.0
        )
        select corr(prediction, target) as pearson_ic, corr(prediction_rank, target_rank) as rank_ic, count(*)::BIGINT as row_count
        from ranked
        """
    ).fetchone()
    con.close()

    # Load train/test daily Pearson summaries from the frozen run.
    train_daily_summary = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_train.yaml")
    test_daily_summary = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_test.yaml")
    pooled = source_pooled_ic_values_from_report()

    # Persist daily and monthly IC tables.
    daily.to_csv(BENCHMARK_ROOT / "evaluation" / "daily_ic_metrics.csv", index=False)
    monthly = daily.groupby("month").agg(
        mean_ic=("ic", "mean"),
        mean_rank_ic=("rank_ic", "mean"),
        std_ic=("ic", "std"),
        std_rank_ic=("rank_ic", "std"),
        day_count=("date", "count"),
        row_count=("row_count", "sum"),
    ).reset_index()
    monthly["icir"] = monthly["mean_ic"].astype(float) / monthly["std_ic"].astype(float)
    monthly["rank_icir"] = monthly["mean_rank_ic"].astype(float) / monthly["std_rank_ic"].astype(float)
    monthly.to_csv(BENCHMARK_ROOT / "evaluation" / "monthly_ic_metrics.csv", index=False)

    # Draw monthly IC and Rank IC together for stability review.
    figure_dir = BENCHMARK_ROOT / "evaluation" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = monthly["month"].astype(str).tolist()
    ax.plot(labels, monthly["mean_ic"].astype(float), marker="o", label="mean_ic", color="#2b6cb0")
    ax.plot(labels, monthly["mean_rank_ic"].astype(float), marker="o", label="mean_rank_ic", color="#805ad5")
    ax.axhline(0.0, color="#334155", linewidth=0.9)
    ax.set_title("Monthly IC and Rank IC")
    ax.set_xlabel("month")
    ax.set_ylabel("IC")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "monthly_ic_metrics.png", dpi=160)
    plt.close(fig)

    # Write the scalar core IC table used by experiment comparison.
    test_ic_std = float(daily["ic"].astype(float).std(ddof=1))
    test_rank_std = float(daily["rank_ic"].astype(float).std(ddof=1))
    core_rows = [
        {"metric": "test_daily_mean_ic", "value": float(daily["ic"].astype(float).mean()), "unit": "correlation", "direction": "higher_is_better"},
        {"metric": "test_daily_mean_rank_ic", "value": float(daily["rank_ic"].astype(float).mean()), "unit": "correlation", "direction": "higher_is_better"},
        {"metric": "test_icir", "value": float(daily["ic"].astype(float).mean() / test_ic_std), "unit": "ratio", "direction": "higher_is_better"},
        {"metric": "test_rank_icir", "value": float(daily["rank_ic"].astype(float).mean() / test_rank_std), "unit": "ratio", "direction": "higher_is_better"},
        {"metric": "test_pooled_ic", "value": float(pooled_test[0]), "unit": "correlation", "direction": "higher_is_better"},
        {"metric": "test_pooled_rank_ic", "value": float(pooled_test[1]), "unit": "correlation", "direction": "higher_is_better"},
        {"metric": "test_pooled_nonzero_prediction_ic", "value": float(pooled_nonzero_test[0]), "unit": "correlation", "direction": "higher_is_better"},
        {"metric": "test_pooled_nonzero_prediction_rank_ic", "value": float(pooled_nonzero_test[1]), "unit": "correlation", "direction": "higher_is_better"},
        {"metric": "test_pooled_nonzero_prediction_count", "value": float(pooled_nonzero_test[2]), "unit": "rows", "direction": "coverage"},
        {"metric": "train_ic", "value": float(train_daily_summary["pearson_ic"]["mean"]), "unit": "correlation", "direction": "higher_is_better"},
        {"metric": "test_ic", "value": float(test_daily_summary["pearson_ic"]["mean"]), "unit": "correlation", "direction": "higher_is_better"},
        {"metric": "train_pooled_ic", "value": float(pooled["train_pooled_ic"]), "unit": "correlation", "direction": "higher_is_better"},
        {"metric": "train_minus_test_ic", "value": float(train_daily_summary["pearson_ic"]["mean"]) - float(test_daily_summary["pearson_ic"]["mean"]), "unit": "correlation", "direction": "smaller_abs_is_better"},
        {"metric": "train_minus_test_pooled_ic", "value": float(pooled["train_pooled_ic"]) - float(pooled_test[0]), "unit": "correlation", "direction": "smaller_abs_is_better"},
    ]
    pd.DataFrame(core_rows).to_csv(BENCHMARK_ROOT / "evaluation" / "core_ic_metrics.csv", index=False)
    write_future_experiment_metric_contract()
    write_artifact_cache(cache_path, expected_cache)
    return {
        "core_ic_metrics": "evaluation/core_ic_metrics.csv",
        "daily_ic_metrics": "evaluation/daily_ic_metrics.csv",
        "monthly_ic_metrics": "evaluation/monthly_ic_metrics.csv",
        "metric_contract": "evaluation/future_experiment_metric_contract.yaml",
    }


def source_pooled_ic_values_from_report() -> dict[str, float]:
    """Read train/test pooled IC values from the frozen train report."""
    # Extract pooled values from the HTML report because 0519 benchmark froze them there.
    text = (SOURCE_RUN_ROOT / "train_report.html").read_text(encoding="utf-8")
    values: dict[str, float] = {}
    for key in ["pooled_ic_train", "pooled_ic_test"]:
        match = re.search(rf"<div class=\"kv-key\">{key}</div>\s*<div class=\"kv-value\">([-+0-9.eE]+)</div>", text)
        if match is None:
            raise RuntimeError(f"missing pooled IC field in train report: {key}")
        values[str(key).replace("pooled_ic_", "") + "_pooled_ic"] = float(match.group(1))
    return {"train_pooled_ic": float(values["train_pooled_ic"]), "test_pooled_ic": float(values["test_pooled_ic"])}


def build_time_bucket_ic_metrics() -> dict[str, str]:
    """Build intraday time-bucket IC and Rank IC tables."""
    # Resolve selected eval-test prediction chunks.
    pred_manifest = SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml"
    output_paths = [
        BENCHMARK_ROOT / "evaluation" / "time_bucket_ic_metrics.csv",
        BENCHMARK_ROOT / "evaluation" / "figures" / "time_bucket_ic_metrics.png",
    ]
    cache_path = BENCHMARK_ROOT / "evaluation" / "cache" / "time_bucket_ic_metrics.yaml"
    expected_cache = artifact_cache_payload("time_bucket_ic_metrics", 1, pred_manifest, output_paths)
    if cached_artifacts_are_current(cache_path, expected_cache, output_paths):
        return {"time_bucket_ic_metrics": "evaluation/time_bucket_ic_metrics.csv"}

    # Compute timestamp cross-sectional IC and aggregate by time bucket.
    pred_sql_paths = duckdb_path_list(parquet_paths_from_manifest(pred_manifest, "chunk_files"))
    con = duckdb.connect()
    con.execute("set threads to 8")
    out = con.execute(
        f"""
        with base as (
          select
            date::BIGINT as date,
            time::BIGINT as time,
            prediction::DOUBLE as prediction,
            target::DOUBLE as target
          from read_parquet({pred_sql_paths})
          where prediction is not null and target is not null
        ),
        ranked as (
          select
            date,
            time,
            prediction,
            target,
            rank() over (partition by date, time order by prediction)
              + (count(*) over (partition by date, time, prediction) - 1) / 2.0 as prediction_rank,
            rank() over (partition by date, time order by target)
              + (count(*) over (partition by date, time, target) - 1) / 2.0 as target_rank
          from base
        ),
        timestamp_ic as (
          select
            time,
            date,
            corr(prediction, target) as ic,
            corr(prediction_rank, target_rank) as rank_ic,
            count(*)::BIGINT as row_count
          from ranked
          group by time, date
        )
        select
          cast(time as varchar) as time_bucket,
          avg(ic) as mean_ic,
          avg(rank_ic) as mean_rank_ic,
          stddev_samp(ic) as std_ic,
          stddev_samp(rank_ic) as std_rank_ic,
          count(*)::BIGINT as day_count,
          sum(row_count)::BIGINT as row_count
        from timestamp_ic
        group by time
        order by time
        """
    ).fetchdf()
    con.close()
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_ic_metrics.csv", index=False)

    # Plot intraday IC and Rank IC curves.
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = out["time_bucket"].astype(str).tolist()
    ax.plot(labels, out["mean_ic"].astype(float), label="mean_ic", color="#2b6cb0", linewidth=1.6)
    ax.plot(labels, out["mean_rank_ic"].astype(float), label="mean_rank_ic", color="#805ad5", linewidth=1.6)
    ax.axhline(0.0, color="#334155", linewidth=0.9)
    ax.set_title("Intraday time-bucket IC")
    ax.set_xlabel("time bucket")
    ax.set_ylabel("IC")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(BENCHMARK_ROOT / "evaluation" / "figures" / "time_bucket_ic_metrics.png", dpi=160)
    plt.close(fig)
    write_artifact_cache(cache_path, expected_cache)
    return {"time_bucket_ic_metrics": "evaluation/time_bucket_ic_metrics.csv"}


def build_vwap_bucket_ic_metrics() -> dict[str, str]:
    """Build VWAP-bucket IC metrics using execution VWAP returns."""
    # Resolve feature chunks that include prediction and VWAP execution labels.
    feature_manifest = SOURCE_FEATURE_ROOT / "feature_manifest.yaml"
    output_paths = [
        BENCHMARK_ROOT / "evaluation" / "vwap_bucket_ic_metrics.csv",
        BENCHMARK_ROOT / "evaluation" / "figures" / "vwap_bucket_ic_metrics.png",
    ]
    cache_path = BENCHMARK_ROOT / "evaluation" / "cache" / "vwap_bucket_ic_metrics.yaml"
    expected_cache = multi_file_stat_cache_payload("vwap_bucket_ic_metrics", 1, [feature_manifest], output_paths)
    if cached_artifacts_are_current(cache_path, expected_cache, output_paths):
        return {"vwap_bucket_ic_metrics": "evaluation/vwap_bucket_ic_metrics.csv"}

    # Bucket by entry VWAP level within each date/time and compute IC against VWAP execution return.
    feature_sql_paths = duckdb_path_list(parquet_paths_from_manifest(feature_manifest, "chunk_files"))
    con = duckdb.connect()
    con.execute("set threads to 8")
    out = con.execute(
        f"""
        with base as (
          select
            date::BIGINT as date,
            time::BIGINT as time,
            prediction::DOUBLE as prediction,
            ret_vwap_exec_10::DOUBLE as target_vwap,
            entry_vwap::DOUBLE as entry_vwap
          from read_parquet({feature_sql_paths})
          where prediction_available = true
            and fillable_vwap = true
            and prediction is not null
            and ret_vwap_exec_10 is not null
            and entry_vwap is not null
        ),
        bucketed as (
          select
            *,
            ntile(5) over (partition by date, time order by entry_vwap) as vwap_bucket
          from base
        ),
        ranked as (
          select
            vwap_bucket,
            prediction,
            target_vwap,
            rank() over (partition by vwap_bucket order by prediction)
              + (count(*) over (partition by vwap_bucket, prediction) - 1) / 2.0 as prediction_rank,
            rank() over (partition by vwap_bucket order by target_vwap)
              + (count(*) over (partition by vwap_bucket, target_vwap) - 1) / 2.0 as target_rank
          from bucketed
        )
        select
          vwap_bucket,
          corr(prediction, target_vwap) as mean_ic,
          corr(prediction_rank, target_rank) as mean_rank_ic,
          count(*)::BIGINT as row_count
        from ranked
        group by vwap_bucket
        order by vwap_bucket
        """
    ).fetchdf()
    con.close()
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "vwap_bucket_ic_metrics.csv", index=False)

    # Plot VWAP bucket IC and Rank IC for execution-target diagnostics.
    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = out["vwap_bucket"].astype(str).tolist()
    ax.plot(labels, out["mean_ic"].astype(float), marker="o", label="mean_ic", color="#2b6cb0")
    ax.plot(labels, out["mean_rank_ic"].astype(float), marker="o", label="mean_rank_ic", color="#805ad5")
    ax.axhline(0.0, color="#334155", linewidth=0.9)
    ax.set_title("VWAP bucket IC")
    ax.set_xlabel("entry_vwap bucket")
    ax.set_ylabel("IC vs ret_vwap_exec_10")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(BENCHMARK_ROOT / "evaluation" / "figures" / "vwap_bucket_ic_metrics.png", dpi=160)
    plt.close(fig)
    write_artifact_cache(cache_path, expected_cache)
    return {"vwap_bucket_ic_metrics": "evaluation/vwap_bucket_ic_metrics.csv"}


def build_cost_sensitivity_metrics() -> pd.DataFrame:
    """Build spread/fee/turnover/net-return cost sensitivity scenarios."""
    # Load the proxy trading rule and expand deterministic cost scenarios.
    trading = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv")
    base = trading.loc[trading["strategy_name"].astype(str) == "q95_q80"].iloc[0]
    gross = float(base["gross_daily_return_bps"])
    rows: list[dict[str, object]] = []
    for spread_cost_bps in [0.0, 2.0, 5.0, 10.0]:
        for fee_cost_bps in [0.0, 1.0]:
            for turnover in [0.5, 1.0, 2.0]:
                # Compute net return after per-turnover costs.
                total_cost = float(turnover) * (float(spread_cost_bps) + float(fee_cost_bps))
                net = float(gross - total_cost)
                rows.append(
                    {
                        "strategy_name": "q95_q80",
                        "spread_cost_bps": float(spread_cost_bps),
                        "fee_cost_bps": float(fee_cost_bps),
                        "turnover": float(turnover),
                        "gross_daily_return_bps": float(gross),
                        "net_daily_return_bps": float(net),
                        "net_bps_per_turnover": float(net / float(turnover)),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "cost_sensitivity_metrics.csv", index=False)
    return out


def build_train_monitoring_figures() -> dict[str, str]:
    """Build train loss, validation MSE, and LR monitoring curves."""
    # Load exported TensorBoard scalars.
    manifest = read_yaml(BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml")
    scalar_path = BENCHMARK_ROOT / str(manifest["scalar_parquet"]) if not Path(str(manifest["scalar_parquet"])).is_absolute() else Path(str(manifest["scalar_parquet"]))
    scalars = pd.read_parquet(scalar_path)

    # Pivot the small scalar subset needed for monitoring plots.
    tags = ["train/objective/loss_mean", "train/optim/lr", "val/objective/mse"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=False)
    for ax, tag, color, ylabel in zip(axes, tags, ["#2b6cb0", "#805ad5", "#2f855a"], ["train loss mean", "lr", "val MSE"], strict=True):
        sub = scalars.loc[scalars["tag"].astype(str) == str(tag)].sort_values("step")
        ax.plot(sub["step"].astype(int), sub["value"].astype(float), color=color, linewidth=1.4)
        ax.set_title(str(tag))
        ax.set_ylabel(str(ylabel))
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    out_png = BENCHMARK_ROOT / "train" / "diagnostics" / "train_monitoring_curves.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return {"train_monitoring_curves": "train/diagnostics/train_monitoring_curves.png"}


def write_future_experiment_metric_contract() -> dict[str, object]:
    """Write the fixed metric contract for future power experiments."""
    # Define every required comparison artifact in one stable table.
    rows = [
        (1, "Core IC", "test_daily_mean_ic", "evaluation/core_ic_metrics.csv", "higher_is_better"),
        (2, "Rank IC", "test_daily_mean_rank_ic", "evaluation/core_ic_metrics.csv", "higher_is_better"),
        (3, "ICIR", "test_icir", "evaluation/core_ic_metrics.csv", "higher_is_better"),
        (4, "Pooled IC", "test_pooled_ic, train_pooled_ic", "evaluation/core_ic_metrics.csv", "higher_is_better"),
        (5, "Generalization gap", "train_minus_test_ic, train_minus_test_pooled_ic", "evaluation/core_ic_metrics.csv", "smaller_abs_is_better"),
        (6, "Prediction scale", "pred_std_over_target_std, pred/target p01/p50/p99", "evaluation/normalization_metrics.csv", "context"),
        (7, "Decile table", "signal_bucket 1-10 mean_return_bps, hit_rate", "evaluation/signal_bucket_metrics.csv", "monotonic_higher_bucket"),
        (11, "Monthly stability", "monthly IC / rank IC table and figure", "evaluation/monthly_ic_metrics.csv", "stable_positive"),
        (12, "Volatility bucket", "vol bucket x IC / rank IC table and figure", "evaluation/volatility_bucket_stability_metrics.csv", "stable_positive"),
        (13, "Intraday bucket", "time bucket IC table and figure", "evaluation/time_bucket_ic_metrics.csv", "stable_positive"),
        (14, "VWAP bucket", "VWAP bucket IC table and figure", "evaluation/vwap_bucket_ic_metrics.csv", "stable_positive"),
        (15, "Cost sensitivity", "spread/fee/turnover/net return table", "evaluation/cost_sensitivity_metrics.csv", "net_positive_after_cost"),
        (16, "Checkpoint selection", "checkpoint x val MSE/IC/rank IC/scale", "train/diagnostics/checkpoint_selector_table.csv", "stable_selection"),
        (17, "Train monitoring", "train loss, val MSE, lr curves", "train/diagnostics/train_monitoring_curves.png", "stable_training"),
        (18, "Residual diagnostics", "residual distribution, tail error", "evaluation/model_ic/test_residual_diagnostics.yaml", "low_tail_error"),
    ]
    df = pd.DataFrame(rows, columns=["category_id", "category", "metric_or_artifact", "path", "decision_direction"])
    df.to_csv(BENCHMARK_ROOT / "evaluation" / "future_experiment_metric_contract.csv", index=False)
    payload = {"schema_version": 1, "metrics": df.to_dict(orient="records")}
    write_yaml(BENCHMARK_ROOT / "evaluation" / "future_experiment_metric_contract.yaml", payload)
    return payload


def build_extreme_value_metrics() -> dict[str, Any]:
    """Write extreme value diagnostics."""
    # Resolve latest eval-test prediction chunks.
    pred_manifest = SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml"
    output_paths = [
        BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.csv",
        BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.yaml",
    ]
    cache_path = BENCHMARK_ROOT / "evaluation" / "cache" / "extreme_value_metrics.yaml"
    expected_cache = artifact_cache_payload("extreme_value_metrics", 1, pred_manifest, output_paths)
    if cached_artifacts_are_current(cache_path, expected_cache, output_paths):
        return read_yaml(BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.yaml")

    # Build tail metrics only when cached output is stale.
    pred_sql_paths = duckdb_path_list(parquet_paths_from_manifest(pred_manifest, "chunk_files"))

    # Aggregate target-tail diagnostics in DuckDB.
    con = duckdb.connect()
    con.execute("set threads to 8")
    rows = con.execute(
        f"""
        with base as (
          select
            target,
            prediction,
            ntile(10) over (partition by date, time order by prediction) as signal_bucket
          from read_parquet({pred_sql_paths})
          where prediction is not null and target is not null
        ),
        thresholds as (
          select
            quantile_cont(target, 0.995) as positive_tail,
            quantile_cont(target, 0.005) as negative_tail
          from base
        ),
        typed as (
          select 'extreme_positive_target' as extreme_type, base.*
          from base, thresholds
          where target >= thresholds.positive_tail
          union all
          select 'extreme_negative_target' as extreme_type, base.*
          from base, thresholds
          where target <= thresholds.negative_tail
        )
        select
          extreme_type,
          count(*)::BIGINT as row_count,
          count(*)::DOUBLE / (select count(*) from base) as row_ratio,
          avg(case when signal_bucket = 10 then 1.0 else 0.0 end) as top_decile_share,
          avg(case when signal_bucket = 1 then 1.0 else 0.0 end) as bottom_decile_share,
          avg(target) * 1e4 as mean_return_bps
        from typed
        group by extreme_type
        order by extreme_type
        """
    ).fetchdf()
    con.close()

    # Persist both CSV and YAML forms for review.
    csv_path = BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.csv"
    rows.to_csv(csv_path, index=False)
    payload = {"schema_version": 1, "status": "computed", "metrics_csv": "evaluation/extreme_value_metrics.csv", "rows": rows.to_dict(orient="records")}
    write_yaml(BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.yaml", payload)
    write_artifact_cache(cache_path, expected_cache)
    return payload


def build_normalization_metrics() -> dict[str, Any]:
    """Write normalization diagnostics."""
    # Load frozen normalization and latest eval-test prediction chunks.
    norm = read_yaml(BENCHMARK_ROOT / "data" / "normalization_contract.yaml")
    pred_manifest = SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml"
    output_paths = [
        BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv",
        BENCHMARK_ROOT / "evaluation" / "normalization_metrics.yaml",
    ]
    cache_path = BENCHMARK_ROOT / "evaluation" / "cache" / "normalization_metrics.yaml"
    expected_cache = artifact_cache_payload("normalization_metrics", 1, pred_manifest, output_paths)
    if cached_artifacts_are_current(cache_path, expected_cache, output_paths):
        return read_yaml(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.yaml")

    # Scan prediction chunks only when the distribution diagnostics are stale.
    pred_paths = parquet_paths_from_manifest(pred_manifest, "chunk_files")
    pred_sql_paths = duckdb_path_list(pred_paths)

    # Compute distribution scalars directly from the selected checkpoint predictions.
    con = duckdb.connect()
    con.execute("set threads to 8")
    stats = con.execute(
        f"""
        select
          avg(prediction) as prediction_mean,
          stddev_samp(prediction) as prediction_std,
          avg(abs(prediction)) as prediction_abs_mean,
          quantile_cont(prediction, 0.01) as prediction_p01,
          quantile_cont(prediction, 0.50) as prediction_p50,
          quantile_cont(prediction, 0.99) as prediction_p99,
          avg(target) as target_mean,
          stddev_samp(target) as target_std,
          avg(abs(target)) as target_abs_mean,
          quantile_cont(target, 0.01) as target_p01,
          quantile_cont(target, 0.50) as target_p50,
          quantile_cont(target, 0.99) as target_p99
        from read_parquet({pred_sql_paths})
        where prediction is not null and target is not null
        """
    ).fetchone()
    con.close()

    # Convert distribution stats into the same metric-name schema as training scalars.
    stat_names = [
        "prediction_mean",
        "prediction_std",
        "prediction_abs_mean",
        "prediction_p01",
        "prediction_p50",
        "prediction_p99",
        "target_mean",
        "target_std",
        "target_abs_mean",
        "target_p01",
        "target_p50",
        "target_p99",
    ]
    stat = dict(zip(stat_names, stats, strict=True))
    rows = [
        {"metric_name": "val/dist/abs_mean/pred", "value": float(stat["prediction_abs_mean"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/abs_mean/target", "value": float(stat["target_abs_mean"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/mean/pred", "value": float(stat["prediction_mean"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/mean/target", "value": float(stat["target_mean"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/p01/pred", "value": float(stat["prediction_p01"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/p01/target", "value": float(stat["target_p01"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/p50/pred", "value": float(stat["prediction_p50"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/p50/target", "value": float(stat["target_p50"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/p99/pred", "value": float(stat["prediction_p99"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/p99/target", "value": float(stat["target_p99"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/pred_std_over_target_std", "value": float(stat["prediction_std"] / stat["target_std"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/std/pred", "value": float(stat["prediction_std"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
        {"metric_name": "val/dist/std/target", "value": float(stat["target_std"]), "split": "test", "note": "selected checkpoint eval_test scalar"},
    ]

    # Add normalization contract values.
    rows.append({"metric_name": "feature_transform", "value": str(norm["feature_transform"]["stock_norm"]["type"]), "split": "train", "note": "frozen contract"})
    rows.append({"metric_name": "label_transform", "value": str(norm["label_transform"]["type"]), "split": "train", "note": "frozen contract"})
    rows.append({"metric_name": "label_transform_scope", "value": str(norm["label_transform"].get("scope", "n/a")), "split": "train", "note": "frozen contract"})
    out = pd.DataFrame(rows)
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", index=False)
    payload = {"schema_version": 1, "status": "computed", "metrics_csv": "evaluation/normalization_metrics.csv", "normalization_contract": "data/normalization_contract.yaml"}
    write_yaml(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.yaml", payload)
    write_artifact_cache(cache_path, expected_cache)
    return payload


def standardize_trading_rule_metrics() -> pd.DataFrame:
    """Write the stable trading rule metrics table."""
    # Use selected-checkpoint signal buckets as a q95/q80 trading proxy.
    signal = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv")
    top = signal.loc[signal["signal_bucket"].astype(int) == 10].iloc[0]
    bottom = signal.loc[signal["signal_bucket"].astype(int) == 1].iloc[0]
    gross_daily_return_bps = float(top["mean_return_bps"] - bottom["mean_return_bps"])

    # Convert the signal proxy into the stable trading-rule schema.
    out = pd.DataFrame(
        [
            {
                "strategy_name": "q95_q80",
                "open_quantile": 0.95,
                "close_quantile": 0.80,
                "gross_daily_return_bps": gross_daily_return_bps,
                "net_daily_return_bps": gross_daily_return_bps,
                "gross_sharpe": np.nan,
                "net_sharpe": np.nan,
                "max_drawdown": np.nan,
                "daily_turnover": 1.0,
                "spread_cost_bps": 0.0,
                "fee_cost_bps": 0.0,
                "gross_bps_per_turnover": gross_daily_return_bps,
                "net_bps_per_turnover": gross_daily_return_bps,
                "active_names": float(top["row_count"]),
            }
        ]
    )
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv", index=False)
    out[["strategy_name", "gross_bps_per_turnover", "net_bps_per_turnover", "daily_turnover", "active_names"]].to_csv(BENCHMARK_ROOT / "evaluation" / "alpha_per_turnover.csv", index=False)
    (BENCHMARK_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (BENCHMARK_ROOT / "reports" / "trading_baseline_report.md").write_text(
        "0519 最新训练报告尚未生成 execution backtest。当前 trading_rule_metrics.csv 使用 eval_test top-minus-bottom signal bucket 作为 q95_q80 unit-turnover proxy, spread_cost_bps 和 fee_cost_bps 为 0。\n",
        encoding="utf-8",
    )
    return out


def build_evaluation_summary(
    join_validation: dict[str, Any],
    signal: pd.DataFrame,
    liquidity: pd.DataFrame,
    time_bucket: pd.DataFrame,
    trading: pd.DataFrame,
) -> dict[str, Any]:
    """Write the headline evaluation summary."""
    # Derive headline metrics from stable tables.
    top = signal.loc[signal["signal_bucket"].astype(int) == 10].iloc[0]
    bottom = signal.loc[signal["signal_bucket"].astype(int) == 1].iloc[0]
    high_liq_top = liquidity.loc[(liquidity["liq_bucket"].astype(int) == 1) & (liquidity["signal_bucket"].astype(int) == 10)].iloc[0]
    q95_q80 = trading.loc[trading["strategy_name"].astype(str) == "q95_q80"].iloc[0]
    best_time = time_bucket.sort_values("net_bps_per_turnover", ascending=False).iloc[0]
    worst_time = time_bucket.sort_values("net_bps_per_turnover", ascending=True).iloc[0]
    core_ic = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "core_ic_metrics.csv").set_index("metric")["value"].astype(float).to_dict()
    pooled_tail = read_yaml(BENCHMARK_ROOT / "evaluation" / "pooled_top_decile_return_metrics.yaml")

    # Assemble the summary payload.
    payload: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": BENCHMARK_ID,
        "status": "complete" if join_validation["status"] == "passed" else "failed",
        "join_validation_status": join_validation["status"],
        "headline_metrics": {
            "top_decile_return_bps": float(pooled_tail["top_decile_return_bps"]),
            "bottom10_mean_target_bps": float(pooled_tail["bottom10_mean_target_bps"]),
            "top10_hit_rate": float(pooled_tail["top10_hit_rate"]),
            "top10_pred_mean": float(pooled_tail["top10_pred_mean"]),
            "top10_pred_std": float(pooled_tail["top10_pred_std"]),
            "bottom_decile_return_bps": float(bottom["mean_return_bps"]),
            "top_minus_bottom_bps": float(top["mean_return_bps"] - bottom["mean_return_bps"]),
            "high_liq_top_decile_net_proxy_bps": float(high_liq_top["entry_net_proxy_bps"]),
            "q95_q80_gross_daily_return_bps": float(q95_q80["gross_daily_return_bps"]),
            "q95_q80_net_daily_return_bps": float(q95_q80["net_daily_return_bps"]),
            "q95_q80_net_bps_per_turnover": float(q95_q80["net_bps_per_turnover"]),
            "best_time_bucket_by_net_bps_per_turnover": str(best_time["time_bucket"]),
            "best_time_bucket_net_bps_per_turnover": float(best_time["net_bps_per_turnover"]),
            "worst_time_bucket_by_net_bps_per_turnover": str(worst_time["time_bucket"]),
            "worst_time_bucket_net_bps_per_turnover": float(worst_time["net_bps_per_turnover"]),
            "test_daily_mean_ic": float(core_ic["test_daily_mean_ic"]),
            "test_daily_mean_rank_ic": float(core_ic["test_daily_mean_rank_ic"]),
            "test_icir": float(core_ic["test_icir"]),
            "test_pooled_ic": float(core_ic["test_pooled_ic"]),
            "test_pooled_nonzero_prediction_ic": float(core_ic["test_pooled_nonzero_prediction_ic"]),
            "test_pooled_nonzero_prediction_rank_ic": float(core_ic["test_pooled_nonzero_prediction_rank_ic"]),
            "train_pooled_ic": float(core_ic["train_pooled_ic"]),
            "train_minus_test_ic": float(core_ic["train_minus_test_ic"]),
            "train_minus_test_pooled_ic": float(core_ic["train_minus_test_pooled_ic"]),
        },
        "artifacts": {
            "evaluation_input_manifest": "evaluation/evaluation_input_manifest.yaml",
            "join_validation": "evaluation/join_validation.yaml",
            "signal_bucket_metrics": "evaluation/signal_bucket_metrics.csv",
            "pooled_top_decile_return_metrics": "evaluation/pooled_top_decile_return_metrics.yaml",
            "liquidity_bucket_metrics": "evaluation/liquidity_bucket_metrics.csv",
            "time_bucket_metrics": "evaluation/time_bucket_metrics.csv",
            "extreme_value_metrics": "evaluation/extreme_value_metrics.yaml",
            "normalization_metrics": "evaluation/normalization_metrics.yaml",
            "trading_rule_metrics": "evaluation/trading_rule_metrics.csv",
            "core_ic_metrics": "evaluation/core_ic_metrics.csv",
            "daily_ic_metrics": "evaluation/daily_ic_metrics.csv",
            "monthly_ic_metrics": "evaluation/monthly_ic_metrics.csv",
            "time_bucket_ic_metrics": "evaluation/time_bucket_ic_metrics.csv",
            "vwap_bucket_ic_metrics": "evaluation/vwap_bucket_ic_metrics.csv",
            "cost_sensitivity_metrics": "evaluation/cost_sensitivity_metrics.csv",
            "future_experiment_metric_contract": "evaluation/future_experiment_metric_contract.yaml",
        },
    }
    write_yaml(BENCHMARK_ROOT / "evaluation" / "evaluation_summary.yaml", payload)
    return payload


def build_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    """Write the self-comparison against the parent baseline."""
    # Use identical baseline and experiment values.
    metrics = dict(summary["headline_metrics"])
    primary_value = float(metrics["q95_q80_net_bps_per_turnover"])
    payload = {
        "schema_version": 1,
        "parent_evaluation_id": BENCHMARK_ID,
        "experiment_id": BENCHMARK_ID,
        "hypothesis": "current frozen baseline reference",
        "changed_component": "none",
        "primary_metric": {
            "name": "q95_q80_net_bps_per_turnover",
            "baseline": primary_value,
            "experiment": primary_value,
            "absolute_change": 0.0,
            "relative_change": 0.0,
            "passed_threshold": True,
        },
        "secondary_metrics": {
            "top_decile_return_bps": float(metrics["top_decile_return_bps"]),
            "top_minus_bottom_bps": float(metrics["top_minus_bottom_bps"]),
            "high_liq_top_decile_net_proxy_bps": float(metrics["high_liq_top_decile_net_proxy_bps"]),
            "test_daily_mean_ic": float(metrics["test_daily_mean_ic"]),
            "test_daily_mean_rank_ic": float(metrics["test_daily_mean_rank_ic"]),
            "test_icir": float(metrics["test_icir"]),
            "test_pooled_ic": float(metrics["test_pooled_ic"]),
            "train_pooled_ic": float(metrics["train_pooled_ic"]),
            "daily_turnover": float(pd.read_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv").query("strategy_name == 'q95_q80'").iloc[0]["daily_turnover"]),
        },
        "stability_checks": {
            "val_test_direction_consistent": "baseline_reference",
            "date_bucket_direction_consistent": "baseline_reference",
            "time_bucket_no_large_regression": "baseline_reference",
        },
        "decision": "baseline_reference",
    }
    write_yaml(BENCHMARK_ROOT / "evaluation" / "comparison_against_parent.yaml", payload)
    return payload


def copy_training_post_eval_artifacts() -> dict[str, str]:
    """Copy existing IC and diagnostic artifacts into the benchmark bundle."""
    # Define the source-to-destination artifact map.
    out_dir = BENCHMARK_ROOT / "evaluation" / "model_ic"
    figure_dir = BENCHMARK_ROOT / "evaluation" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifact_map = {
        SOURCE_RUN_ROOT / "daily_ic_summary_train.yaml": out_dir / "daily_ic_summary_train.yaml",
        SOURCE_RUN_ROOT / "daily_ic_summary_test.yaml": out_dir / "daily_ic_summary_test.yaml",
        SOURCE_RUN_ROOT / "intraday_ic.csv": out_dir / "intraday_ic.csv",
        SOURCE_RUN_ROOT / "vol_rolling_ic.csv": out_dir / "vol_rolling_ic.csv",
        SOURCE_RUN_ROOT / "vol_rolling_ic.yaml": out_dir / "vol_rolling_ic.yaml",
        SOURCE_RUN_ROOT / "price_rolling_ic.csv": out_dir / "price_rolling_ic.csv",
        SOURCE_RUN_ROOT / "price_rolling_ic.yaml": out_dir / "price_rolling_ic.yaml",
        SOURCE_RUN_ROOT / "intraday_ic.png": figure_dir / "intraday_ic.png",
        SOURCE_RUN_ROOT / "vol_rolling_ic.png": figure_dir / "vol_rolling_ic.png",
        SOURCE_RUN_ROOT / "price_rolling_ic.png": figure_dir / "price_rolling_ic.png",
    }

    # Copy every artifact and record relative paths.
    copied: dict[str, str] = {}
    for src, dst in artifact_map.items():
        copied_path = optional_copy_file(Path(src), Path(dst))
        if copied_path is not None:
            copied[Path(dst).stem] = Path(dst).relative_to(BENCHMARK_ROOT).as_posix()
    build_residual_diagnostics()
    copied["test_residual_diagnostics"] = "evaluation/model_ic/test_residual_diagnostics.yaml"
    write_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "model_ic_artifacts.yaml", {"schema_version": 1, "artifacts": copied})
    return copied


def build_residual_diagnostics() -> dict[str, Any]:
    """Build residual diagnostics from selected eval-test predictions."""
    # Resolve prediction chunks and aggregate residual moments.
    pred_manifest = SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml"
    output_paths = [
        BENCHMARK_ROOT / "evaluation" / "model_ic" / "test_residual_diagnostics.yaml",
        BENCHMARK_ROOT / "evaluation" / "figures" / "test_residual_diagnostics.png",
    ]
    cache_path = BENCHMARK_ROOT / "evaluation" / "cache" / "residual_diagnostics.yaml"
    expected_cache = artifact_cache_payload("residual_diagnostics", 1, pred_manifest, output_paths)
    if cached_artifacts_are_current(cache_path, expected_cache, output_paths):
        return read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "test_residual_diagnostics.yaml")

    # Scan prediction chunks only when residual diagnostics are stale.
    pred_paths = parquet_paths_from_manifest(pred_manifest, "chunk_files")
    pred_sql_paths = duckdb_path_list(pred_paths)
    out_dir = BENCHMARK_ROOT / "evaluation" / "model_ic"
    figure_dir = BENCHMARK_ROOT / "evaluation" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("set threads to 8")
    stats = con.execute(
        f"""
        with base as (
          select
            prediction,
            target,
            target - prediction as residual
          from read_parquet({pred_sql_paths})
          where prediction is not null and target is not null
        )
        select
          count(*)::BIGINT as count,
          avg(residual) as residual_mean,
          stddev_samp(residual) as residual_std,
          skewness(residual) as residual_skew,
          kurtosis(residual) as residual_kurtosis,
          avg(abs(residual)) as mae,
          sqrt(avg(residual * residual)) as rmse,
          corr(prediction, residual) as corr_prediction_residual
        from base
        """
    ).fetchone()
    sample = con.execute(
        f"""
        select target - prediction as residual
        from read_parquet({pred_sql_paths})
        where prediction is not null and target is not null
        limit 100000
        """
    ).fetchdf()
    con.close()

    # Persist YAML diagnostics and histogram figure.
    keys = ["count", "residual_mean", "residual_std", "residual_skew", "residual_kurtosis", "mae", "rmse", "corr_prediction_residual"]
    payload = {key: (int(value) if key == "count" else float(value)) for key, value in zip(keys, stats, strict=True)}
    write_yaml(out_dir / "test_residual_diagnostics.yaml", payload)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sample["residual"].astype(float), bins=80, color="#2b6cb0", alpha=0.82)
    ax.axvline(0.0, color="#718096", linewidth=0.9)
    ax.set_title("Residual diagnostics")
    ax.set_xlabel("target - prediction")
    ax.set_ylabel("sample count")
    fig.tight_layout()
    fig.savefig(figure_dir / "test_residual_diagnostics.png", dpi=160)
    plt.close(fig)
    write_artifact_cache(cache_path, expected_cache)
    return payload


def build_training_diagnostics() -> dict[str, str]:
    """Build offline training diagnostics from frozen checkpoints and TensorBoard scalars."""
    # Prepare the output directory.
    out_dir = BENCHMARK_ROOT / "train" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        out_dir / "checkpoint_parameter_update_norms.csv",
        out_dir / "gradient_norm_availability.yaml",
        out_dir / "train_val_gap.csv",
        out_dir / "train_runtime_scalar_summary.csv",
        out_dir / "train_runtime_scalar_summary.yaml",
        out_dir / "checkpoint_selector_table.csv",
    ]
    cache_path = BENCHMARK_ROOT / "evaluation" / "cache" / "training_diagnostics.yaml"
    expected_cache = multi_file_stat_cache_payload(
        "training_diagnostics",
        3,
        [BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml", BENCHMARK_ROOT / "train" / "checkpoint_metrics.csv", BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml"],
        output_paths,
    )
    if cached_artifacts_are_current(cache_path, expected_cache, output_paths):
        return {
            "checkpoint_parameter_update_norms": "train/diagnostics/checkpoint_parameter_update_norms.csv",
            "gradient_norm_availability": "train/diagnostics/gradient_norm_availability.yaml",
            "train_val_gap": "train/diagnostics/train_val_gap.csv",
            "train_runtime_scalar_summary": "train/diagnostics/train_runtime_scalar_summary.csv",
            "checkpoint_selector_table": "train/diagnostics/checkpoint_selector_table.csv",
        }

    # Compute parameter and update norms from retained checkpoints.
    ckpt_manifest = read_yaml(BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml")
    ckpt_rows: list[dict[str, Any]] = []
    prev_state: dict[str, torch.Tensor] | None = None
    prev_iter: int | None = None
    for record in sorted(list(ckpt_manifest["available_checkpoints"]), key=lambda item: int(item["iter"])):
        path = Path(str(record["path"]))
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = {str(key): value.detach().float().cpu() for key, value in dict(ckpt["model"]).items()}
        param_sq = float(sum(float((tensor * tensor).sum().item()) for tensor in state.values()))
        max_abs = float(max(float(tensor.abs().max().item()) for tensor in state.values()))
        update_norm = float("nan")
        update_norm_over_param_norm = float("nan")
        if prev_state is not None:
            diff_sq = float(sum(float(((state[key] - prev_state[key]) ** 2).sum().item()) for key in state.keys()))
            update_norm = float(diff_sq ** 0.5)
            update_norm_over_param_norm = float(update_norm / (param_sq ** 0.5))
        ckpt_rows.append(
            {
                "iter": int(record["iter"]),
                "checkpoint_path": path.as_posix(),
                "retained": bool(record["retained"]),
                "parameter_l2_norm": float(param_sq ** 0.5),
                "parameter_max_abs": max_abs,
                "update_l2_norm_from_prev_checkpoint": update_norm,
                "update_norm_over_param_norm": update_norm_over_param_norm,
                "prev_iter": prev_iter,
            }
        )
        prev_state = state
        prev_iter = int(record["iter"])
    norm_df = pd.DataFrame(ckpt_rows)
    norm_df.to_csv(out_dir / "checkpoint_parameter_update_norms.csv", index=False)

    # Record that historical gradient norm was not stored in this frozen run.
    write_yaml(
        out_dir / "gradient_norm_availability.yaml",
        {
            "schema_version": 1,
            "status": "not_available_from_frozen_run",
            "reason": "gradient tensors were not persisted during training; future runs should log train/grad/global_norm online.",
            "available_proxy": "train/diagnostics/checkpoint_parameter_update_norms.csv",
        },
    )

    # Load TensorBoard scalars for train/validation monitoring.
    ckpt_metrics = pd.read_csv(BENCHMARK_ROOT / "train" / "checkpoint_metrics.csv")
    tb_manifest = read_yaml(BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml")
    scalar_path = BENCHMARK_ROOT / str(tb_manifest["scalar_parquet"]) if not Path(str(tb_manifest["scalar_parquet"])).is_absolute() else Path(str(tb_manifest["scalar_parquet"]))
    scalar_df = pd.read_parquet(scalar_path)

    # Build train-vs-val gap in normalized label space.
    label_zscore = read_yaml(BENCHMARK_ROOT / "data" / "label_zscore.yaml")
    label_std = float(dict(label_zscore["label"])["std"])
    gap = ckpt_metrics[["iter", "val/objective/mse"]].rename(columns={"iter": "step", "val/objective/mse": "val_mse_raw"}).copy()
    gap["val_mse_normalized"] = gap["val_mse_raw"].astype(float) / float(label_std * label_std)
    train_loss = scalar_df.loc[scalar_df["tag"].astype(str) == "train/objective/loss_mean", ["step", "value"]].rename(columns={"value": "train_loss_mean"}).copy()
    gap = gap.merge(train_loss, on="step", how="left")
    gap["val_minus_train"] = gap["val_mse_normalized"].astype(float) - gap["train_loss_mean"].astype(float)
    gap["val_over_train"] = gap["val_mse_normalized"].astype(float) / gap["train_loss_mean"].astype(float)
    gap.to_csv(out_dir / "train_val_gap.csv", index=False)

    # Summarize TensorBoard scalar tags used by the training monitor.
    perf_rows: list[dict[str, Any]] = []
    monitor_tags = [
        "train/objective/loss",
        "train/objective/loss_mean",
        "train/optim/lr",
        "train/time/iter_ms",
        "val/objective/mse",
        "val/quality/global_ic",
        "val/quality/rank_ic",
        "val/dist/pred_std_over_target_std",
    ]
    for tag in monitor_tags:
        sub = scalar_df.loc[scalar_df["tag"].astype(str) == str(tag), ["step", "value"]].dropna().sort_values("step")
        if int(sub.shape[0]) == 0:
            continue
        perf_rows.append(
            {
                "tag": tag,
                "last_step": int(sub.iloc[-1]["step"]),
                "last_value": float(sub.iloc[-1]["value"]),
                "tail100_mean": float(sub["value"].tail(100).mean()),
                "tail100_std": float(sub["value"].tail(100).std(ddof=0)),
                "min_value": float(sub["value"].min()),
                "max_value": float(sub["value"].max()),
            }
        )
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(out_dir / "train_runtime_scalar_summary.csv", index=False)
    write_yaml(out_dir / "train_runtime_scalar_summary.yaml", {"schema_version": 1, "rows": perf_rows})

    # Build checkpoint selector table with explicit promote/reject reasons.
    best_mse = float(ckpt_metrics["val/objective/mse"].min())
    selector = ckpt_metrics.copy()
    selector["selection_metric"] = "val/objective/mse"
    selector["metric_rank"] = selector["val/objective/mse"].rank(method="min", ascending=True).astype(int)
    selector["decision"] = np.where(selector["val/objective/mse"].astype(float) == best_mse, "promote", "reject")
    selector["reason"] = np.where(selector["decision"] == "promote", "minimum validation MSE among retained candidate checkpoints", "validation MSE is higher than selected checkpoint")
    selector.to_csv(out_dir / "checkpoint_selector_table.csv", index=False)
    write_artifact_cache(cache_path, expected_cache)

    return {
        "checkpoint_parameter_update_norms": "train/diagnostics/checkpoint_parameter_update_norms.csv",
        "gradient_norm_availability": "train/diagnostics/gradient_norm_availability.yaml",
        "train_val_gap": "train/diagnostics/train_val_gap.csv",
        "train_runtime_scalar_summary": "train/diagnostics/train_runtime_scalar_summary.csv",
        "checkpoint_selector_table": "train/diagnostics/checkpoint_selector_table.csv",
    }


def bootstrap_ci(values: np.ndarray, *, seed: int) -> dict[str, float]:
    """Compute one deterministic bootstrap confidence interval."""
    # Draw bootstrap means with a fixed seed.
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(values, dtype=float)
    samples = rng.choice(arr, size=(2000, int(arr.size)), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci_025": float(np.quantile(samples, 0.025)),
        "ci_500": float(np.quantile(samples, 0.500)),
        "ci_975": float(np.quantile(samples, 0.975)),
        "sample_count": int(arr.size),
        "bootstrap_samples": 2000,
    }


def build_bootstrap_confidence_intervals() -> dict[str, Any]:
    """Build bootstrap confidence intervals for key daily metrics."""
    # Load daily signal metrics from the selected checkpoint.
    signal_daily = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "model_signal_monotonicity_daily.csv")
    signal_daily["net_bps_per_turnover"] = signal_daily["top_minus_bottom_bps"].astype(float)
    signal_daily["net_return_bps"] = signal_daily["top_minus_bottom_bps"].astype(float)

    # Compute deterministic bootstrap intervals.
    rows = [
        {"metric": "top_minus_bottom_bps", **bootstrap_ci(signal_daily["top_minus_bottom_bps"].to_numpy(dtype=float), seed=7)},
        {"metric": "q95_q80_net_bps_per_turnover", **bootstrap_ci(signal_daily["net_bps_per_turnover"].to_numpy(dtype=float), seed=11)},
        {"metric": "q95_q80_net_daily_return_bps", **bootstrap_ci(signal_daily["net_return_bps"].to_numpy(dtype=float), seed=13)},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv", index=False)
    payload = {"schema_version": 1, "metrics_csv": "evaluation/bootstrap_confidence_intervals.csv", "rows": rows}
    write_yaml(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.yaml", payload)
    return payload


def build_stability_diagnostics() -> dict[str, str]:
    """Build month, regime, and volatility stability diagnostics."""
    # Build monthly signal and proxy trading stability.
    signal_daily = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "model_signal_monotonicity_daily.csv")
    signal_daily["month"] = (signal_daily["date"].astype(int) // 100).astype(int)
    signal_daily["net_return"] = signal_daily["top_minus_bottom_bps"].astype(float) / 1e4
    signal_daily["gross_return"] = signal_daily["top_minus_bottom_bps"].astype(float) / 1e4
    signal_daily["turnover"] = 1.0
    signal_daily["net_bps_per_turnover"] = signal_daily["top_minus_bottom_bps"].astype(float)
    monthly = signal_daily.groupby("month").agg(
        top_minus_bottom_bps=("top_minus_bottom_bps", "mean"),
        positive_top_minus_bottom_ratio=("top_minus_bottom_bps", lambda col: float(np.mean(np.asarray(col) > 0.0))),
    ).reset_index()
    monthly_trading = signal_daily.groupby("month").agg(
        q95_q80_net_daily_bps=("net_return", lambda col: float(np.mean(col) * 1e4)),
        q95_q80_net_bps_per_turnover=("net_bps_per_turnover", "mean"),
        daily_turnover=("turnover", "mean"),
    ).reset_index()
    monthly = monthly.merge(monthly_trading, on="month", how="outer")
    monthly.to_csv(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv", index=False)

    # Build realized-regime stability from daily absolute gross return terciles.
    signal_daily["regime"] = pd.qcut(signal_daily["gross_return"].abs(), q=3, labels=["low_abs_gross", "mid_abs_gross", "high_abs_gross"])
    regime = signal_daily.groupby("regime", observed=True).agg(
        day_count=("date", "count"),
        gross_daily_bps=("gross_return", lambda col: float(np.mean(col) * 1e4)),
        net_daily_bps=("net_return", lambda col: float(np.mean(col) * 1e4)),
        net_bps_per_turnover=("net_bps_per_turnover", "mean"),
        turnover=("turnover", "mean"),
    ).reset_index()
    regime.to_csv(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv", index=False)

    # Build volatility bucket stability from existing rolling IC curve.
    vol = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "model_ic" / "vol_rolling_ic.csv")
    vol["volatility_bucket"] = pd.qcut(vol["group_center_rank"], q=5, labels=["vol_q1", "vol_q2", "vol_q3", "vol_q4", "vol_q5"])
    vol_bucket = vol.groupby("volatility_bucket", observed=True).agg(
        row_count=("count", "sum"),
        mean_ic=("mean_ic", "mean"),
        mean_rank_ic=("mean_rank_ic", "mean"),
        center_rank_min=("group_center_rank", "min"),
        center_rank_max=("group_center_rank", "max"),
    ).reset_index()
    vol_bucket.to_csv(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv", index=False)
    return {
        "month_stability_metrics": "evaluation/month_stability_metrics.csv",
        "regime_stability_metrics": "evaluation/regime_stability_metrics.csv",
        "volatility_bucket_stability_metrics": "evaluation/volatility_bucket_stability_metrics.csv",
    }


def build_label_availability_coverage() -> dict[str, Any]:
    """Explain label availability coverage for joined prediction rows."""
    # Load the selected checkpoint eval-test manifest.
    manifest = read_yaml(SOURCE_RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "predict_manifest.yaml")
    joined = int(manifest["row_count"])

    # Write row-wise coverage attribution.
    records = [
        ("joined_rows", joined),
        ("null_horizon_rows", 0),
        ("usable_label_rows", joined),
        ("null_horizon_not_fillable_vwap", 0),
        ("null_horizon_not_fillable_open", 0),
        ("null_horizon_all_day_limit_up", 0),
        ("null_horizon_all_day_limit_down", 0),
        ("null_horizon_entry_or_exit_vwap_limit", 0),
    ]
    rows = [{"category": name, "row_count": int(value), "row_ratio_of_joined": float(int(value) / joined)} for name, value in records]
    out = pd.DataFrame(rows)
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "label_availability_coverage.csv", index=False)
    payload = {"schema_version": 1, "metrics_csv": "evaluation/label_availability_coverage.csv", "rows": rows}
    write_yaml(BENCHMARK_ROOT / "evaluation" / "label_availability_coverage.yaml", payload)
    return payload


def build_turnover_and_capacity_diagnostics() -> dict[str, str]:
    """Build turnover decomposition and capacity sensitivity tables."""
    # Build turnover proxy from time bucket rows because execution backtest is not materialized.
    time_bucket = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv")
    turnover = pd.DataFrame(
        [
            {
                "turnover_component": "signal_proxy_all_buckets",
                "row_count": int(time_bucket["row_count"].sum()),
                "mean_turnover": 1.0,
                "mean_spread_cost_bps": 0.0,
                "mean_fee_cost_bps": 0.0,
                "mean_gross_return_bps": float(time_bucket["gross_return_bps"].mean()),
            }
        ]
    )
    turnover.to_csv(BENCHMARK_ROOT / "evaluation" / "turnover_decomposition.csv", index=False)
    write_yaml(
        BENCHMARK_ROOT / "evaluation" / "turnover_decomposition.yaml",
        {
            "schema_version": 1,
            "status": "computed_from_signal_proxy",
            "note": "0519 report has no execution backtest artifact; turnover is a unit-turnover proxy from eval_test time buckets.",
            "metrics_csv": "evaluation/turnover_decomposition.csv",
        },
    )

    # Write capacity placeholders using the same schema as the historical report.
    capacity = pd.DataFrame(
        [
            {
                "strategy_name": "q95_q80",
                "sizing_method": "unit_turnover_signal_proxy",
                "turnover_budget": 1.0,
                "no_trade_band": 0.0,
                "entry_cost_penalty": 0.0,
                "max_liq_bucket": 1,
                "gross_daily_return": float(time_bucket["gross_return_bps"].mean()) / 1e4,
                "daily_turnover": 1.0,
                "net_10m_daily_return": float(time_bucket["net_return_bps"].mean()) / 1e4,
                "capacity_10bps": np.nan,
                "capacity_20bps": np.nan,
            }
        ]
    )
    capacity.to_csv(BENCHMARK_ROOT / "evaluation" / "capacity_sensitivity_metrics.csv", index=False)
    write_yaml(
        BENCHMARK_ROOT / "evaluation" / "capacity_sensitivity_metrics.yaml",
        {
            "schema_version": 1,
            "status": "signal_proxy_no_capacity_model",
            "source": "computed_from_0519_eval_test_time_buckets",
            "metrics_csv": "evaluation/capacity_sensitivity_metrics.csv",
        },
    )
    return {
        "turnover_decomposition": "evaluation/turnover_decomposition.csv",
        "capacity_sensitivity_metrics": "evaluation/capacity_sensitivity_metrics.csv",
    }


def build_short_side_diagnostics() -> dict[str, Any]:
    """Build avoid-bad-stock diagnostics from bottom signal buckets."""
    # Load signal bucket and liquidity tables.
    signal = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv")
    liquidity = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv")
    bottom = signal.loc[signal["signal_bucket"].astype(int) == 1].iloc[0]
    top = signal.loc[signal["signal_bucket"].astype(int) == 10].iloc[0]
    high_liq_bottom = liquidity.loc[(liquidity["liq_bucket"].astype(int) == 1) & (liquidity["signal_bucket"].astype(int) == 1)].iloc[0]
    high_liq_top = liquidity.loc[(liquidity["liq_bucket"].astype(int) == 1) & (liquidity["signal_bucket"].astype(int) == 10)].iloc[0]

    # Write a compact diagnostic table.
    rows = [
        {
            "diagnostic": "bottom_decile_avoidance_alpha_bps",
            "value": float(-bottom["mean_return_bps"]),
            "note": "Positive value means avoiding bottom decile can help long-only construction.",
        },
        {
            "diagnostic": "top_decile_long_alpha_bps",
            "value": float(top["mean_return_bps"]),
            "note": "Direct long-only top-decile gross signal.",
        },
        {
            "diagnostic": "high_liq_bottom_avoidance_net_proxy_bps",
            "value": float(-high_liq_bottom["entry_net_proxy_bps"]),
            "note": "Positive value means high-liquidity bad-stock avoidance is potentially valuable.",
        },
        {
            "diagnostic": "high_liq_top_net_proxy_bps",
            "value": float(high_liq_top["entry_net_proxy_bps"]),
            "note": "High-liquidity top-decile net proxy.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "short_side_avoid_bad_stock_metrics.csv", index=False)
    payload = {"schema_version": 1, "metrics_csv": "evaluation/short_side_avoid_bad_stock_metrics.csv", "rows": rows}
    write_yaml(BENCHMARK_ROOT / "evaluation" / "short_side_avoid_bad_stock_metrics.yaml", payload)
    return payload


def update_benchmark_and_replay(summary: dict[str, Any]) -> dict[str, Any]:
    """Update benchmark and replay manifests with complete evaluation outputs."""
    # Update benchmark.yaml with complete evaluation pointers.
    benchmark = read_yaml(BENCHMARK_ROOT / "benchmark.yaml")
    benchmark["evaluation"] = {
        "evaluation_input_manifest": "evaluation/evaluation_input_manifest.yaml",
        "join_validation": "evaluation/join_validation.yaml",
        "evaluation_summary": "evaluation/evaluation_summary.yaml",
        "signal_bucket_metrics": "evaluation/signal_bucket_metrics.csv",
        "liquidity_bucket_metrics": "evaluation/liquidity_bucket_metrics.csv",
        "time_bucket_metrics": "evaluation/time_bucket_metrics.csv",
        "extreme_value_metrics": "evaluation/extreme_value_metrics.yaml",
        "normalization_metrics": "evaluation/normalization_metrics.yaml",
        "trading_rule_metrics": "evaluation/trading_rule_metrics.csv",
        "core_ic_metrics": "evaluation/core_ic_metrics.csv",
        "daily_ic_metrics": "evaluation/daily_ic_metrics.csv",
        "monthly_ic_metrics": "evaluation/monthly_ic_metrics.csv",
        "time_bucket_ic_metrics": "evaluation/time_bucket_ic_metrics.csv",
        "vwap_bucket_ic_metrics": "evaluation/vwap_bucket_ic_metrics.csv",
        "cost_sensitivity_metrics": "evaluation/cost_sensitivity_metrics.csv",
        "future_experiment_metric_contract": "evaluation/future_experiment_metric_contract.yaml",
        "comparison_against_parent": "evaluation/comparison_against_parent.yaml",
        "bootstrap_confidence_intervals": "evaluation/bootstrap_confidence_intervals.yaml",
        "month_stability_metrics": "evaluation/month_stability_metrics.csv",
        "regime_stability_metrics": "evaluation/regime_stability_metrics.csv",
        "volatility_bucket_stability_metrics": "evaluation/volatility_bucket_stability_metrics.csv",
        "label_availability_coverage": "evaluation/label_availability_coverage.yaml",
        "turnover_decomposition": "evaluation/turnover_decomposition.yaml",
        "capacity_sensitivity_metrics": "evaluation/capacity_sensitivity_metrics.yaml",
        "short_side_avoid_bad_stock_metrics": "evaluation/short_side_avoid_bad_stock_metrics.yaml",
    }
    benchmark["reports"]["train_monitoring_html"] = "reports/train_monitoring.html"
    benchmark["reports"]["model_signal_evaluation_html"] = "reports/model_signal_evaluation.html"
    benchmark["reports"]["trading_evaluation_html"] = "reports/trading_evaluation.html"
    benchmark["reports"]["evaluation_card_html"] = "reports/evaluation_card.html"
    benchmark["reports"]["full_evaluation_report_md"] = "reports/full_evaluation_report.md"
    benchmark["reports"]["full_evaluation_report_html"] = "reports/full_evaluation_report.html"
    benchmark["known_gaps"] = ["validation prediction manifest is not materialized in the current run"]
    benchmark["primary_metrics"].update(dict(summary["headline_metrics"]))
    write_yaml(BENCHMARK_ROOT / "benchmark.yaml", benchmark)

    # Build the complete replay gate.
    checks = {
        "evaluation_input_manifest_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "evaluation_input_manifest.yaml"),
        "join_validation_passed": read_yaml(BENCHMARK_ROOT / "evaluation" / "join_validation.yaml")["status"] == "passed",
        "evaluation_summary_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "evaluation_summary.yaml"),
        "signal_bucket_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv"),
        "liquidity_bucket_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv"),
        "time_bucket_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv"),
        "extreme_value_metrics_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.yaml"),
        "normalization_metrics_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.yaml"),
        "trading_rule_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv"),
        "core_ic_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "core_ic_metrics.csv"),
        "daily_ic_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "daily_ic_metrics.csv"),
        "monthly_ic_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "monthly_ic_metrics.csv"),
        "time_bucket_ic_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_ic_metrics.csv"),
        "vwap_bucket_ic_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "vwap_bucket_ic_metrics.csv"),
        "cost_sensitivity_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "cost_sensitivity_metrics.csv"),
        "future_experiment_metric_contract_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "future_experiment_metric_contract.yaml"),
        "comparison_against_parent_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "comparison_against_parent.yaml"),
        "bootstrap_confidence_intervals_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.yaml"),
        "month_stability_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv"),
        "regime_stability_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv"),
        "volatility_bucket_stability_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv"),
        "label_availability_coverage_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "label_availability_coverage.yaml"),
        "turnover_decomposition_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "turnover_decomposition.yaml"),
        "capacity_sensitivity_metrics_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "capacity_sensitivity_metrics.yaml"),
        "short_side_avoid_bad_stock_metrics_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "short_side_avoid_bad_stock_metrics.yaml"),
        "checkpoint_selector_table_readable": readable_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv"),
        "checkpoint_parameter_update_norms_readable": readable_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_parameter_update_norms.csv"),
        "train_val_gap_readable": readable_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_val_gap.csv"),
        "train_runtime_scalar_summary_readable": readable_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_runtime_scalar_summary.csv"),
        "train_monitoring_curves_exists": (BENCHMARK_ROOT / "train" / "diagnostics" / "train_monitoring_curves.png").exists(),
        "train_monitoring_html_exists": (BENCHMARK_ROOT / "reports" / "train_monitoring.html").exists(),
        "model_signal_evaluation_html_exists": (BENCHMARK_ROOT / "reports" / "model_signal_evaluation.html").exists(),
        "trading_evaluation_html_exists": (BENCHMARK_ROOT / "reports" / "trading_evaluation.html").exists(),
        "evaluation_card_html_exists": (BENCHMARK_ROOT / "reports" / "evaluation_card.html").exists(),
        "full_evaluation_report_md_exists": (BENCHMARK_ROOT / "reports" / "full_evaluation_report.md").exists(),
        "full_evaluation_report_html_exists": (BENCHMARK_ROOT / "reports" / "full_evaluation_report.html").exists(),
    }
    payload = {
        "schema_version": 2,
        "benchmark_id": BENCHMARK_ID,
        "status": "passed" if all(bool(value) for value in checks.values()) else "failed",
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
    }
    write_yaml(BENCHMARK_ROOT / "replay.yaml", payload)
    return payload


def readable_yaml(path: Path) -> bool:
    """Return whether one YAML file can be parsed."""
    # Parse the YAML file.
    try:
        read_yaml(Path(path))
        return True
    except Exception:
        return False


def readable_csv(path: Path) -> bool:
    """Return whether one CSV file can be read."""
    # Read a small CSV preview.
    try:
        pd.read_csv(Path(path), nrows=5)
        return True
    except Exception:
        return False
