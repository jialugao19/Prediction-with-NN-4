"""Freeze the current prediction-NN-2 research baseline into a benchmark bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from prediction_nn2.html_report import build_page, render_code_block, render_section, render_table, render_value_rows


REPO_ROOT = Path("/home/maomao/prediction-NN-2")
RUN_ROOT = Path("/data-cache/nn/0428/date_ranges")
TRADE_FEATURE_ROOT = Path("/data-cache/nn/trade_plan_experiments/0515/features/entry1_h60_slot60")
SIGNAL_DIAG_ROOT = Path("/data-cache/nn/trade_plan_experiments/0516_model_signal_diagnostics")
TRADING_BASELINE_ROOT = Path("/data-cache/nn/trade_plan_experiments/0516_percentile_hysteresis_baseline")
BENCHMARK_ID = "20260518_current_baseline"
BENCHMARK_PARENT = Path("/data-cache/nn/benchmarks/prediction_nn2")
BENCHMARK_ROOT = BENCHMARK_PARENT / BENCHMARK_ID
REPO_BENCHMARK_REPORT_DIR = REPO_ROOT / "report" / "benchmarks"
BEST_ITER = 140000


def sha256_file(path: Path) -> str:
    """Compute a sha256 digest for one file."""
    # Stream the file in chunks so checkpoint hashing does not allocate large buffers.
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, include_sha: bool) -> dict[str, object]:
    """Build one file metadata record."""
    # Read filesystem metadata first so missing files fail immediately.
    p = Path(path)
    st = p.stat()

    # Attach sha256 only for files where the caller wants a strong identity.
    record: dict[str, object] = {
        "path": p.as_posix(),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    if bool(include_sha):
        record["sha256"] = sha256_file(p)
    return record


def ensure_clean_dir(path: Path) -> None:
    """Create one directory and keep existing contents append-safe."""
    # Make the benchmark directory tree without deleting prior artifacts.
    Path(path).mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> Path:
    """Copy one small artifact into the benchmark bundle."""
    # Create the parent directory before copying the file with metadata.
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(src), dst)
    return dst


def hardlink_or_copy(src: Path, dst: Path) -> Path:
    """Hardlink one artifact, copying only when hardlinking is impossible."""
    # Ensure the destination parent exists before linking.
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst

    # Prefer hardlinks so large parquet chunks are frozen without duplicating storage.
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    # Parse YAML as a mapping; corrupt or empty files should fail visibly.
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    """Write one YAML mapping with stable human-readable formatting."""
    # Serialize YAML with insertion order preserved for reviewability.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def run_text(args: list[str], cwd: Path) -> str:
    """Run one non-interactive command and return stdout."""
    # Execute command without shell interpolation so recorded snapshots are stable.
    result = subprocess.run(list(args), cwd=Path(cwd), check=True, capture_output=True, text=True)
    return str(result.stdout).strip()


def freeze_code_snapshot() -> dict[str, object]:
    """Freeze git state and source file hashes."""
    # Capture git state and dirty diff for reproducibility.
    code_dir = BENCHMARK_ROOT / "code"
    ensure_clean_dir(code_dir)
    git_commit = run_text(["git", "rev-parse", "HEAD"], REPO_ROOT)
    git_branch = run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], REPO_ROOT)
    git_status = run_text(["git", "status", "--short"], REPO_ROOT)
    diff_text = run_text(["git", "diff", "--", "."], REPO_ROOT)
    (code_dir / "diff.patch").write_text(diff_text, encoding="utf-8")

    # Hash the source files that define data prep, model, evaluation, and backtest behavior.
    source_patterns = [
        "prediction_nn2/*.py",
        "portfolio_backtest/*.py",
        "portfolio_backtest/scripts/*.py",
        "qmodel/qmodel/**/*.py",
    ]
    source_rows: list[dict[str, object]] = []
    for pattern in source_patterns:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if path.is_file():
                source_rows.append(
                    {
                        "path": path.relative_to(REPO_ROOT).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": int(path.stat().st_size),
                    }
                )
    source_manifest = {"files": source_rows}
    write_yaml(code_dir / "source_files_manifest.yaml", source_manifest)

    # Persist the git snapshot as a compact manifest.
    payload = {
        "repo_root": REPO_ROOT.as_posix(),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_status_short": git_status,
        "diff_patch": "code/diff.patch",
    }
    write_yaml(code_dir / "git_snapshot.yaml", payload)
    return payload


def freeze_data_contracts() -> dict[str, object]:
    """Freeze data, normalization, split, and join contracts."""
    # Copy core metadata and normalization contracts from the completed run.
    data_dir = BENCHMARK_ROOT / "data"
    ensure_clean_dir(data_dir)
    meta_src = RUN_ROOT / "artifacts" / "npz" / "meta.yaml"
    meta = load_yaml(meta_src)
    copy_file(meta_src, data_dir / "npz_meta.yaml")
    for name in ["pooled_zscore.yaml", "label_zscore.yaml", "value_transform.yaml"]:
        copy_file(RUN_ROOT / "artifacts" / "data_clean" / name, data_dir / name)

    # Write split dates and feature names separately for quick experiment comparisons.
    split_dates = dict(meta["dates"])
    feature_names = list(meta["feature_transform"]["stock_features"]) + list(meta["feature_transform"]["time_features"])
    write_yaml(data_dir / "split_dates.yaml", split_dates)
    write_yaml(data_dir / "feature_names.yaml", {"feature_names": feature_names})
    normalization_contract = {
        "feature_transform": dict(meta["feature_transform"]),
        "label": dict(meta["label"]),
        "label_transform": dict(meta["label_transform"]),
        "normalization_files": {
            "pooled_zscore": "data/pooled_zscore.yaml",
            "label_zscore": "data/label_zscore.yaml",
            "value_transform": "data/value_transform.yaml",
        },
    }
    write_yaml(data_dir / "normalization_contract.yaml", normalization_contract)

    # Record raw and derived data file identities without hashing huge raw binaries.
    raw_records: list[dict[str, object]] = []
    key_paths = [
        meta_src,
        RUN_ROOT / "artifacts" / "data_clean" / "feature_moments.csv",
        RUN_ROOT / "artifacts" / "data_clean" / "label_audit.csv",
        TRADE_FEATURE_ROOT / "feature_manifest.yaml",
        TRADE_FEATURE_ROOT / "feature_audit.yaml",
    ]
    for path in key_paths:
        raw_records.append(file_record(path, include_sha=True))
    for path in sorted((RUN_ROOT / "artifacts" / "npz").glob("*.f32")) + sorted((RUN_ROOT / "artifacts" / "npz").glob("*.i64")):
        raw_records.append(file_record(path, include_sha=False))
    raw_manifest = {"records": raw_records}
    write_yaml(data_dir / "raw_data_manifest.yaml", raw_manifest)

    # Persist the prediction-to-backtest join contract.
    join_contract = {
        "join_key": ["date", "time", "code"],
        "prediction_source": (RUN_ROOT / "run" / "inference_test" / f"iter_{BEST_ITER}" / "inference_manifest.yaml").as_posix(),
        "enriched_feature_source": (TRADE_FEATURE_ROOT / "feature_manifest.yaml").as_posix(),
        "requirements": [
            "prediction side key must be unique",
            "feature side key must be unique",
            "joined row count and unmatched rows must be recorded",
            "time must be the decision timestamp",
            "prediction horizon must align with ret_vwap_exec_10",
        ],
    }
    write_yaml(data_dir / "join_contract.yaml", join_contract)
    return {
        "npz_meta": "data/npz_meta.yaml",
        "normalization_contract": "data/normalization_contract.yaml",
        "raw_data_manifest": "data/raw_data_manifest.yaml",
        "join_contract": "data/join_contract.yaml",
    }


def summarize_state_dict(ckpt_path: Path) -> dict[str, object]:
    """Summarize checkpoint state_dict keys, shapes, and dtypes."""
    # Load the checkpoint on CPU and inspect only metadata plus light parameter moments.
    ckpt = torch.load(Path(ckpt_path), map_location="cpu")
    model_state = dict(ckpt["model"])
    rows: list[dict[str, object]] = []
    total_params = 0
    for key, tensor in model_state.items():
        arr = tensor.detach().float()
        count = int(tensor.numel())
        total_params += count
        rows.append(
            {
                "key": str(key),
                "shape": [int(v) for v in tensor.shape],
                "dtype": str(tensor.dtype),
                "numel": count,
                "mean": float(arr.mean().item()) if count else 0.0,
                "std": float(arr.std(unbiased=False).item()) if count else 0.0,
            }
        )
    return {
        "checkpoint_iteration": int(ckpt["iteration"]),
        "total_parameter_tensors": int(len(rows)),
        "total_parameters": int(total_params),
        "state_dict": rows,
    }


def freeze_model_and_train() -> dict[str, object]:
    """Freeze model checkpoint, train manifests, checkpoint metrics, and TensorBoard scalars."""
    # Copy best checkpoint and core train manifests.
    model_dir = BENCHMARK_ROOT / "model"
    train_dir = BENCHMARK_ROOT / "train"
    ensure_clean_dir(model_dir)
    ensure_clean_dir(train_dir)
    ckpt_src = RUN_ROOT / "run" / "ckpt" / f"iter_{BEST_ITER}.pt"
    ckpt_dst = hardlink_or_copy(ckpt_src, model_dir / "best_checkpoint.pt")
    train_manifest = load_yaml(RUN_ROOT / "manifests" / "train.yaml")
    copy_file(RUN_ROOT / "manifests" / "train.yaml", train_dir / "train_stage_manifest.yaml")

    # Persist effective model and checkpoint summaries.
    meta = load_yaml(RUN_ROOT / "artifacts" / "npz" / "meta.yaml")
    feature_dim = int(meta["storage"]["groups"]["train"]["feature_dim"])
    effective_model_summary = {
        "model_class": "GruMlpRegressor",
        "input_tensor": f"(B, T=60, F={feature_dim})",
        "gru": {
            "input_size": feature_dim,
            "hidden_size": 256,
            "num_layers": 2,
            "bidirectional": False,
            "dropout": 0.0,
        },
        "mlp": {
            "configured_hidden_dims": "from pipeline config",
            "effective_hidden_width": 512,
            "dropout": "from pipeline config",
            "note": "model.py hard-codes the MLP width to 512.",
        },
        "output_dim": 1,
    }
    state_summary = summarize_state_dict(ckpt_src)
    effective_model_summary["trainable_parameters"] = int(state_summary["total_parameters"])
    write_yaml(model_dir / "effective_model_summary.yaml", effective_model_summary)
    write_yaml(model_dir / "state_dict_summary.yaml", state_summary)

    # Build checkpoint metrics from all validation metric JSON files.
    metric_rows: list[dict[str, object]] = []
    for metrics_path in sorted((RUN_ROOT / "run" / "eval_val").glob("iter_*/metrics.json")):
        iter_value = int(metrics_path.parent.name.split("_")[-1])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"iter": iter_value, "selected": bool(iter_value == int(BEST_ITER)), "checkpoint_path": (RUN_ROOT / "run" / "ckpt" / f"iter_{iter_value}.pt").as_posix()}
        row.update({str(key): float(value) for key, value in metrics.items()})
        metric_rows.append(row)
    checkpoint_metrics_path = train_dir / "checkpoint_metrics.csv"
    write_csv(checkpoint_metrics_path, metric_rows)
    checkpoint_manifest = {
        "best_checkpoint_iter": int(BEST_ITER),
        "best_selection_metric": "val/objective/mse",
        "best_selection_rule": "min",
        "best_checkpoint": file_record(ckpt_dst, include_sha=True),
        "available_checkpoints": [
            file_record(path, include_sha=False) | {"iter": int(path.stem.split("_")[-1]), "retained": bool(path.name == f"iter_{BEST_ITER}.pt")}
            for path in sorted((RUN_ROOT / "run" / "ckpt").glob("iter_*.pt"))
        ],
    }
    write_yaml(train_dir / "checkpoint_manifest.yaml", checkpoint_manifest)

    # Export TensorBoard scalar events into a stable parquet table.
    scalar_df = export_tensorboard_scalars(RUN_ROOT / "run" / "tb")
    scalar_path = train_dir / "train_scalars.parquet"
    scalar_df.to_parquet(scalar_path, index=False)
    tensorboard_manifest = {
        "source_dir": (RUN_ROOT / "run" / "tb").as_posix(),
        "event_files": [file_record(path, include_sha=True) for path in sorted((RUN_ROOT / "run" / "tb").glob("events.out.tfevents*"))],
        "scalar_parquet": "train/train_scalars.parquet",
        "scalar_rows": int(scalar_df.shape[0]),
        "tags": sorted(set(str(value) for value in scalar_df["tag"].tolist())),
    }
    write_yaml(train_dir / "tensorboard_manifest.yaml", tensorboard_manifest)
    copy_file(RUN_ROOT / "train_loss.png", train_dir / "train_loss_curve.png")

    # Write a serializable training config summary.
    train_config = {
        "stage_manifest": dict(train_manifest),
        "optimizer": "AdamW",
        "criterion": "MSELoss",
        "checkpoint_selection": {"metric": "val/objective/mse", "rule": "min", "best_iter": int(BEST_ITER)},
    }
    write_yaml(train_dir / "train_config.yaml", train_config)
    return {
        "best_checkpoint": "model/best_checkpoint.pt",
        "checkpoint_manifest": "train/checkpoint_manifest.yaml",
        "checkpoint_metrics": "train/checkpoint_metrics.csv",
        "train_scalars": "train/train_scalars.parquet",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a list of mapping rows to CSV."""
    # Write an empty marker table when there are no rows.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        path.write_text("", encoding="utf-8")
        return path

    # Preserve the first row's key order as the CSV schema.
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_tensorboard_scalars(tb_dir: Path) -> pd.DataFrame:
    """Export all TensorBoard scalar tags into a parquet-friendly DataFrame."""
    # Load all scalar event data from the TensorBoard directory.
    acc = EventAccumulator(Path(tb_dir).as_posix(), size_guidance={"scalars": 0})
    acc.Reload()

    # Convert each scalar event to a long-form row.
    rows: list[dict[str, object]] = []
    for tag in acc.Tags().get("scalars", []):
        for scalar in acc.Scalars(str(tag)):
            rows.append(
                {
                    "tag": str(tag),
                    "step": int(scalar.step),
                    "wall_time": float(scalar.wall_time),
                    "value": float(scalar.value),
                }
            )
    return pd.DataFrame(rows)


def freeze_predictions() -> dict[str, object]:
    """Freeze prediction manifests and enriched backtest feature chunks."""
    # Copy train/test/inference manifests and record missing validation prediction manifest explicitly.
    pred_dir = BENCHMARK_ROOT / "predictions"
    ensure_clean_dir(pred_dir)
    paths = {
        "train_predict_manifest": RUN_ROOT / "run" / "eval_train" / f"iter_{BEST_ITER}" / "predict_manifest.yaml",
        "test_predict_manifest": RUN_ROOT / "run" / "eval_test" / f"iter_{BEST_ITER}" / "predict_manifest.yaml",
        "inference_test_manifest": RUN_ROOT / "run" / "inference_test" / f"iter_{BEST_ITER}" / "inference_manifest.yaml",
    }
    output: dict[str, object] = {}
    for key, src in paths.items():
        dst = pred_dir / f"{key}.yaml"
        copy_file(src, dst)
        output[key] = f"predictions/{dst.name}"

    # Preserve validation eval metrics even though no val predict_manifest is available.
    val_manifest = {
        "status": "not_materialized",
        "reason": "current run has eval_val metrics and rank0.feather, but no predict_manifest.yaml for val",
        "eval_val_dir": (RUN_ROOT / "run" / "eval_val" / f"iter_{BEST_ITER}").as_posix(),
    }
    write_yaml(pred_dir / "val_predict_manifest.yaml", val_manifest)
    output["val_predict_manifest"] = "predictions/val_predict_manifest.yaml"

    # Hardlink enriched feature chunks because they already contain prediction and backtest fields.
    enriched_dir = pred_dir / "enriched_test_predictions"
    ensure_clean_dir(enriched_dir / "feature_chunks")
    feature_manifest = load_yaml(TRADE_FEATURE_ROOT / "feature_manifest.yaml")
    hardlinked_files: list[str] = []
    for rel_path in list(feature_manifest["chunk_files"]):
        src = TRADE_FEATURE_ROOT / str(rel_path)
        dst = enriched_dir / str(rel_path)
        hardlink_or_copy(src, dst)
        hardlinked_files.append(dst.relative_to(BENCHMARK_ROOT).as_posix())
    enriched_manifest = dict(feature_manifest)
    enriched_manifest["source_manifest"] = (TRADE_FEATURE_ROOT / "feature_manifest.yaml").as_posix()
    enriched_manifest["benchmark_chunk_files"] = hardlinked_files
    enriched_manifest["storage"] = "hardlink_or_copy"
    write_yaml(pred_dir / "enriched_test_predictions_manifest.yaml", enriched_manifest)
    copy_file(TRADE_FEATURE_ROOT / "feature_audit.yaml", pred_dir / "enriched_test_feature_audit.yaml")
    output["enriched_test_predictions"] = "predictions/enriched_test_predictions_manifest.yaml"
    return output


def freeze_existing_reports_and_metrics() -> dict[str, object]:
    """Freeze existing diagnostics, baseline backtest outputs, and report artifacts."""
    # Copy model signal diagnostics tables and figures.
    eval_dir = BENCHMARK_ROOT / "evaluation"
    reports_dir = BENCHMARK_ROOT / "reports"
    backtest_dir = BENCHMARK_ROOT / "backtest"
    ensure_clean_dir(eval_dir)
    ensure_clean_dir(reports_dir)
    ensure_clean_dir(backtest_dir)
    copy_file(SIGNAL_DIAG_ROOT / "model_signal_decile_summary.csv", eval_dir / "top_tail_metrics.csv")
    copy_file(SIGNAL_DIAG_ROOT / "model_signal_liquidity_summary.csv", eval_dir / "liquidity_signal_metrics.csv")
    copy_file(SIGNAL_DIAG_ROOT / "model_signal_alpha_per_turnover.csv", eval_dir / "alpha_per_turnover.csv")
    copy_file(SIGNAL_DIAG_ROOT / "model_signal_monotonicity_daily.csv", eval_dir / "model_signal_monotonicity_daily.csv")
    copy_file(SIGNAL_DIAG_ROOT / "model_signal_diagnostics_summary.yaml", eval_dir / "model_metrics.yaml")
    copy_file(SIGNAL_DIAG_ROOT / "model_signal_diagnostics_report.md", reports_dir / "model_signal_diagnostics.md")
    copy_file(SIGNAL_DIAG_ROOT / "model_signal_diagnostics_report.html", reports_dir / "model_signal_diagnostics.html")
    for path in sorted(SIGNAL_DIAG_ROOT.glob("*.png")):
        copy_file(path, eval_dir / "figures" / path.name)

    # Copy trading baseline summary, reports, curves, and per-variant daily/bar outputs.
    copy_file(TRADING_BASELINE_ROOT / "percentile_hysteresis_baseline_summary.csv", backtest_dir / "baseline_summary.csv")
    copy_file(TRADING_BASELINE_ROOT / "percentile_hysteresis_baseline_summary.yaml", backtest_dir / "baseline_summary.yaml")
    copy_file(TRADING_BASELINE_ROOT / "percentile_hysteresis_baseline_report.md", reports_dir / "trading_baseline_report.md")
    copy_file(TRADING_BASELINE_ROOT / "percentile_hysteresis_baseline_report.html", reports_dir / "trading_baseline_report.html")
    copy_file(TRADING_BASELINE_ROOT / "percentile_hysteresis_baseline_comparison.png", backtest_dir / "figures" / "percentile_hysteresis_baseline_comparison.png")
    for path in sorted(TRADING_BASELINE_ROOT.glob("*curve.png")):
        copy_file(path, backtest_dir / "figures" / path.name)
    for path in sorted((TRADING_BASELINE_ROOT / "variants").glob("*/*.csv")):
        copy_file(path, backtest_dir / "variants" / path.parent.name / path.name)
    for path in sorted((TRADING_BASELINE_ROOT / "variants").glob("*/*.yaml")):
        copy_file(path, backtest_dir / "variants" / path.parent.name / path.name)

    # Add missing P1 diagnostic placeholders as explicit incomplete artifacts.
    write_yaml(
        eval_dir / "time_bucket_metrics.yaml",
        {"status": "not_computed", "priority": "P1", "reason": "time bucket attribution is planned after P0 freeze"},
    )
    write_yaml(
        eval_dir / "extreme_value_metrics.yaml",
        {"status": "not_computed", "priority": "P1", "reason": "extreme value diagnostics are planned after P0 freeze"},
    )
    write_yaml(
        eval_dir / "normalization_metrics.yaml",
        {"status": "not_computed", "priority": "P1", "reason": "normalization sensitivity diagnostics are planned after P0 freeze"},
    )
    return {
        "top_tail_metrics": "evaluation/top_tail_metrics.csv",
        "liquidity_signal_metrics": "evaluation/liquidity_signal_metrics.csv",
        "alpha_per_turnover": "evaluation/alpha_per_turnover.csv",
        "trading_baseline_summary": "backtest/baseline_summary.csv",
    }


def build_benchmark_metrics() -> dict[str, object]:
    """Extract headline metrics for benchmark.yaml and the benchmark card."""
    # Read existing structured outputs and compute headline fields.
    decile = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "top_tail_metrics.csv")
    alpha = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "alpha_per_turnover.csv")
    baseline = pd.read_csv(BENCHMARK_ROOT / "backtest" / "baseline_summary.csv")
    checkpoint = pd.read_csv(BENCHMARK_ROOT / "train" / "checkpoint_metrics.csv")
    top = decile.loc[decile["signal_decile"].astype(int) == 10].iloc[0]
    bottom = decile.loc[decile["signal_decile"].astype(int) == 1].iloc[0]
    best_alpha = alpha.iloc[0]
    q95_q80 = baseline.loc[baseline["strategy_name"].astype(str) == "q95_q80"].iloc[0]
    selected = checkpoint.loc[checkpoint["selected"].astype(bool)].iloc[0]
    return {
        "val_mse": float(selected["val/objective/mse"]),
        "val_rank_ic": float(selected["val/quality/rank_ic"]),
        "val_global_ic": float(selected["val/quality/global_ic"]),
        "top_decile_return_bps": float(top["daily_return_bps"]),
        "bottom_decile_return_bps": float(bottom["daily_return_bps"]),
        "top_minus_bottom_bps": float(top["daily_return_bps"] - bottom["daily_return_bps"]),
        "best_alpha_strategy": str(best_alpha["strategy_name"]),
        "best_gross_bps_per_turnover": float(best_alpha["gross_bps_per_turnover"]),
        "best_net_bps_per_turnover": float(best_alpha["net_bps_per_turnover"]),
        "q95_q80_gross_daily_bps": float(q95_q80["gross_daily_return"] * 1e4),
        "q95_q80_net_daily_bps": float(q95_q80["net_daily_return"] * 1e4),
        "q95_q80_daily_turnover": float(q95_q80["daily_turnover"]),
    }


def write_benchmark_card(metrics: dict[str, object], benchmark_payload: dict[str, object]) -> dict[str, str]:
    """Write markdown and HTML benchmark cards."""
    # Build a compact markdown benchmark card for repo review.
    reports_dir = BENCHMARK_ROOT / "reports"
    ensure_clean_dir(reports_dir)
    lines = [
        "# Prediction-NN-2 Current Baseline Benchmark Card",
        "",
        "## 定义",
        "",
        "这是当前 prediction-NN-2 research baseline 的 frozen benchmark。它固定现有模型、诊断和 10min percentile hysteresis trading baseline, 用作后续假设检验实验的 parent benchmark。",
        "",
        "## 核心指标",
        "",
        f"- benchmark id: `{BENCHMARK_ID}`.",
        f"- val MSE: `{float(metrics['val_mse']):.8g}`.",
        f"- val rank IC: `{float(metrics['val_rank_ic']):.6f}`.",
        f"- top decile return: `{float(metrics['top_decile_return_bps']):.3f}` bps.",
        f"- top-minus-bottom: `{float(metrics['top_minus_bottom_bps']):.3f}` bps.",
        f"- best alpha per turnover strategy: `{metrics['best_alpha_strategy']}`.",
        f"- best gross bps/turnover: `{float(metrics['best_gross_bps_per_turnover']):.3f}`.",
        f"- best net bps/turnover: `{float(metrics['best_net_bps_per_turnover']):.3f}`.",
        f"- q95_q80 net daily return: `{float(metrics['q95_q80_net_daily_bps']):.2f}` bps.",
        "",
        "## 已固化",
        "",
        "- code snapshot 和 dirty diff.",
        "- data / normalization / join contracts.",
        "- best checkpoint.",
        "- TensorBoard scalar parquet.",
        "- train/test/inference prediction manifests.",
        "- enriched test prediction feature chunks, hardlink/copy 固化.",
        "- top-tail / liquidity / alpha-per-turnover diagnostics.",
        "- 10min percentile hysteresis trading baseline.",
        "",
        "## 已知缺口",
        "",
        "- time bucket attribution 尚未计算, 标记为 P1.",
        "- extreme value diagnostics 尚未计算, 标记为 P1.",
        "- normalization sensitivity diagnostics 尚未计算, 标记为 P1.",
        "- validation prediction manifest 未物化, 当前仅保存 eval_val metrics.",
        "",
        "## 使用规则",
        "",
        "后续实验必须声明 parent benchmark 为本 benchmark, 并输出统一 schema 的 comparison_against_baseline。晋升新 benchmark 前必须通过 replay test。",
    ]
    markdown = "\n".join(lines)
    md_path = reports_dir / "benchmark_card.md"
    md_path.write_text(markdown, encoding="utf-8")

    # Render a single-file HTML card with the benchmark YAML appendix.
    rows = [(str(key), str(value)) for key, value in dict(metrics).items()]
    sections = [
        render_section("Headline Metrics", render_value_rows(rows)),
        render_section("Benchmark YAML", render_code_block(yaml.safe_dump(benchmark_payload, sort_keys=False, allow_unicode=True))),
        render_section("Markdown Appendix", render_code_block(markdown)),
    ]
    html = build_page("Prediction-NN-2 Current Baseline", "Frozen benchmark card for future hypothesis-driven experiments.", sections)
    html_path = reports_dir / "benchmark_card.html"
    html_path.write_text(html, encoding="utf-8")

    # Copy lightweight cards into the repo report directory.
    REPO_BENCHMARK_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    copy_file(md_path, REPO_BENCHMARK_REPORT_DIR / "current_benchmark.md")
    copy_file(html_path, REPO_BENCHMARK_REPORT_DIR / "current_benchmark.html")
    return {"markdown": "reports/benchmark_card.md", "html": "reports/benchmark_card.html"}


def write_replay_manifest() -> dict[str, object]:
    """Run a minimal replay check over frozen benchmark artifacts."""
    # Check that required artifacts are present and readable.
    checks = {
        "benchmark_yaml_exists": (BENCHMARK_ROOT / "benchmark.yaml").exists(),
        "train_scalars_readable": readable_parquet(BENCHMARK_ROOT / "train" / "train_scalars.parquet"),
        "best_checkpoint_exists": (BENCHMARK_ROOT / "model" / "best_checkpoint.pt").exists(),
        "enriched_predictions_manifest_readable": readable_yaml(BENCHMARK_ROOT / "predictions" / "enriched_test_predictions_manifest.yaml"),
        "top_tail_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "top_tail_metrics.csv"),
        "liquidity_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_signal_metrics.csv"),
        "backtest_summary_readable": readable_csv(BENCHMARK_ROOT / "backtest" / "baseline_summary.csv"),
        "benchmark_card_exists": (BENCHMARK_ROOT / "reports" / "benchmark_card.md").exists(),
    }
    status = "passed" if all(bool(value) for value in checks.values()) else "failed"
    payload = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "status": status,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
    }
    write_yaml(BENCHMARK_ROOT / "replay.yaml", payload)
    return payload


def readable_yaml(path: Path) -> bool:
    """Return whether one YAML file can be parsed."""
    # Parse the YAML file and let any parse error mark the check false.
    try:
        load_yaml(path)
        return True
    except Exception:
        return False


def readable_csv(path: Path) -> bool:
    """Return whether one CSV file can be read with at least one row or header."""
    # Read a small CSV preview to validate the file.
    try:
        pd.read_csv(path, nrows=5)
        return True
    except Exception:
        return False


def readable_parquet(path: Path) -> bool:
    """Return whether one parquet file can be read with a small preview."""
    # Read a tiny parquet slice to validate the file.
    try:
        pd.read_parquet(path, columns=None).head(1)
        return True
    except Exception:
        return False


def write_registry(benchmark_payload: dict[str, object], replay_payload: dict[str, object]) -> None:
    """Update benchmark current pointer and lightweight registry files."""
    # Write data-cache current pointer.
    BENCHMARK_PARENT.mkdir(parents=True, exist_ok=True)
    current = {
        "current_benchmark_id": BENCHMARK_ID,
        "current_benchmark_root": BENCHMARK_ROOT.as_posix(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "replay_status": str(replay_payload["status"]),
    }
    write_yaml(BENCHMARK_PARENT / "current.yaml", current)

    # Write repo registry with the latest benchmark row.
    REPO_BENCHMARK_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    registry = {
        "benchmarks": [
            {
                "benchmark_id": BENCHMARK_ID,
                "benchmark_root": BENCHMARK_ROOT.as_posix(),
                "created_at": benchmark_payload["created_at"],
                "replay_status": replay_payload["status"],
                "benchmark_card": "current_benchmark.md",
            }
        ]
    }
    write_yaml(REPO_BENCHMARK_REPORT_DIR / "benchmark_registry.yaml", registry)


def build_benchmark_payload(
    code: dict[str, object],
    data: dict[str, object],
    train: dict[str, object],
    predictions: dict[str, object],
    reports: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    """Assemble the top-level benchmark manifest."""
    # Build a single manifest that points to every frozen contract and artifact.
    return {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "frozen",
        "description": "Current prediction-NN-2 research baseline frozen for future hypothesis-driven experiments.",
        "source": code,
        "data": data,
        "model": {
            "effective_model_summary": "model/effective_model_summary.yaml",
            "state_dict_summary": "model/state_dict_summary.yaml",
            "best_checkpoint": "model/best_checkpoint.pt",
        },
        "training": train,
        "predictions": predictions,
        "evaluation": reports,
        "reports": {
            "benchmark_card_md": "reports/benchmark_card.md",
            "benchmark_card_html": "reports/benchmark_card.html",
            "model_signal_diagnostics": "reports/model_signal_diagnostics.md",
            "trading_baseline_report": "reports/trading_baseline_report.md",
        },
        "primary_metrics": metrics,
        "known_gaps": [
            "time bucket attribution is not computed in P0",
            "extreme value diagnostics are not computed in P0",
            "normalization sensitivity diagnostics are not computed in P0",
            "val prediction manifest is not materialized in the current run",
        ],
    }


def freeze_current_baseline() -> Path:
    """Freeze the current baseline and return benchmark.yaml path."""
    # Create the benchmark root and freeze each artifact family.
    ensure_clean_dir(BENCHMARK_ROOT)
    code = freeze_code_snapshot()
    data = freeze_data_contracts()
    train = freeze_model_and_train()
    predictions = freeze_predictions()
    reports = freeze_existing_reports_and_metrics()
    metrics = build_benchmark_metrics()

    # Write benchmark.yaml before replay, then benchmark card and registry.
    benchmark_payload = build_benchmark_payload(code, data, train, predictions, reports, metrics)
    benchmark_path = write_yaml(BENCHMARK_ROOT / "benchmark.yaml", benchmark_payload)
    write_benchmark_card(metrics, benchmark_payload)
    replay_payload = write_replay_manifest()
    benchmark_payload["replay"] = {"manifest": "replay.yaml", "status": replay_payload["status"]}
    write_yaml(BENCHMARK_ROOT / "benchmark.yaml", benchmark_payload)
    write_registry(benchmark_payload, replay_payload)
    return benchmark_path
