"""Run the end-to-end pipeline: data prep, training, evaluation, and report generation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import pandas as pd

from qmodel.config import LRSchedulerConfig
from qmodel.metrics import builtin

from prediction_nn2.data_prep import DataPrepConfig, list_trade_dates, prepare_npz_splits
from prediction_nn2.dataset import NpzDatasetSpec, Stock1mNpzDataset
from prediction_nn2.clean_report import render_clean_report_from_meta
from prediction_nn2.eval_ic import (
    EvalConfig,
    attach_labels,
    compute_predict_report_from_manifest,
    ic_time_series_summary,
    intraday_time_series_ic,
    load_eval_predictions,
    pooled_ic,
    price_rolling_ic,
    volatility_rolling_ic,
)
from prediction_nn2.model import MlpConfig, MlpRegressor


@dataclass(frozen=True)
class PipelineConfig:
    """Define top-level pipeline knobs and output paths."""

    pipeline_mode: str
    root_dir: Path
    stock1m_dir: Path
    start_trade_date: int
    end_trade_date: int
    split_policy: str
    train_end_trade_date: int
    val_end_trade_date: int
    train_days: int
    val_days: int
    test_days: int
    rolling_step_days: int
    seed: int
    horizon_minutes: int
    sample_stocks_per_minute: int
    use_cross_sectional_gaussianize: bool
    data_prep_include_predict_split: bool
    data_prep_norm_fit_scope: str
    data_prep_days_per_call: int
    data_prep_workers: int
    batch_size: int
    num_workers: int
    num_iters: int
    save_every: int
    eval_every: int
    eval_during: bool
    eval_during_num_iters: int
    eval_batch_size: int
    learning_rate: float
    hidden_dims: list[int]
    dropout: float
    input_window_size: int
    rolling_window: int
    rolling_step: int


def _redirect_to_data_cache(path: Path) -> Path:
    """Redirect any non-/data-cache path into /data-cache to reduce container disk pressure."""
    # Preserve the original path shape under /data-cache to keep runs easy to locate.
    p = Path(path)
    if p.is_absolute() and str(p).startswith("/data-cache/"):
        return p
    return Path("/data-cache") / p.as_posix().lstrip("/")


def _eval_mse_metric(namespace: str, it: int, pred: np.ndarray, target: np.ndarray, **kwargs: object) -> dict[str, float]:
    """Compute mean squared error as a scalar evaluation metric."""
    # Compute MSE in float64 for numerical stability on large arrays.
    diff = pred.astype(np.float64, copy=False) - target.astype(np.float64, copy=False)
    mse = float(np.mean(diff * diff, dtype=np.float64))
    return {f"{str(namespace)}/objective/mse": mse}


def _export_train_loss_curve(tb_dir: Path, out_png: Path) -> None:
    """Export a training loss curve PNG from TensorBoard event files."""
    # Load scalar events using TensorBoard's event accumulator.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Read events and require the standard qmodel scalar tag to exist.
    acc = EventAccumulator(tb_dir.as_posix(), size_guidance={"scalars": 0})
    acc.Reload()
    tag = "train/objective/loss"
    scalars = acc.Scalars(tag)
    steps = np.asarray([s.step for s in scalars], dtype=int)
    values = np.asarray([s.value for s in scalars], dtype=float)

    # Plot the raw loss curve with a log-y axis to emphasize early training dynamics.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, values, linewidth=1.6, color="#4c72b0")
    ax.set_title("Training loss curve (TensorBoard scalar: train/objective/loss)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _format_seconds(seconds: float) -> str:
    """Format a float second count into a short human-readable string."""
    # Convert seconds into hour/minute/second integers for stable log output.
    total = int(round(float(seconds)))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = int(total % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _stage_manifest_path(out_root: Path, stage: str) -> Path:
    """Resolve the YAML manifest path for one pipeline stage."""
    # Keep all stage markers in a single folder so skip logic is easy to inspect.
    p = Path(out_root) / "manifests" / f"{str(stage)}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _stage_fingerprint(payload: dict[str, object]) -> str:
    """Compute a stable SHA256 fingerprint for stage skip decisions."""
    # Hash YAML text so fingerprint remains stable across Python versions.
    import yaml

    txt = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def _load_stage_manifest(path: Path) -> dict[str, object]:
    """Load a stage YAML manifest into a dict."""
    # Parse YAML without fallback so corrupt manifests fail loudly.
    import yaml

    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _write_stage_manifest(path: Path, payload: dict[str, object]) -> None:
    """Persist one stage manifest as YAML."""
    # Write the manifest with stable key ordering disabled so humans can diff it.
    import yaml

    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _move_eval_dir(run_dir: Path, *, src_name: str, dst_name: str, it: int) -> Path:
    """Move one qmodel eval iter directory to a new parent to avoid overwrites."""
    # Resolve source and destination paths under the run directory.
    src = Path(run_dir) / str(src_name) / f"iter_{int(it)}"
    dst = Path(run_dir) / str(dst_name) / f"iter_{int(it)}"
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Remove any prior destination so repeated pipeline invocations can resume cleanly.
    if dst.exists():
        import shutil

        shutil.rmtree(dst)

    # Move the directory as an atomic rename when possible.
    src.rename(dst)
    return dst


def _build_qmodel_config(cfg: PipelineConfig, feature_dim: int, run_root: Path) -> SimpleNamespace:
    """Build a qmodel-compatible flat config namespace for single-GPU training."""
    # Select training device; prefer CUDA when available.
    use_cuda = bool(torch.cuda.is_available())
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")

    # Define dataset callable with explicit spec object to avoid hidden globals.
    npz_dir = Path(run_root) / "artifacts" / "npz"
    dataset_spec = NpzDatasetSpec(data_dir=npz_dir, pin_memory=use_cuda, window_size=int(cfg.input_window_size))

    def dataset_class(group: str, dtype: torch.dtype) -> Stock1mNpzDataset:
        """Build a dataset instance for the requested split."""
        # Create the dataset with an explicit NPZ spec.
        return Stock1mNpzDataset(group, dtype, dataset_spec)

    # Build model config and core training components.
    model_cfg = MlpConfig(
        in_dim=int(feature_dim) * int(cfg.input_window_size),
        hidden_dims=list(cfg.hidden_dims),
        dropout=float(cfg.dropout),
        dtype=torch.float32,
    )

    # Build evaluator config namespace to match qmodel evaluator expectations.
    evaluator = SimpleNamespace(
        eval_checkpoint_iter=[int(cfg.num_iters) - 1],
        eval_all_num_iters=0,
        eval_batch_size=int(cfg.eval_batch_size),
        predict_chunk_row_count=2_000_000,
    )

    # Assemble a flat config object matching qmodel CpuTrainer/CpuEvaluator field access.
    conf = SimpleNamespace(
        device=device,
        dist_backend=None,
        window_size=int(cfg.input_window_size),
        ret_col_name="label_ret",
        dataset_class=dataset_class,
        model_class=MlpRegressor,
        model=model_cfg,
        seed=int(cfg.seed),
        amp_dtype=torch.float32,
        eval_dtype=torch.float32,
        train_dtype=torch.float32,
        criterion=torch.nn.MSELoss(),
        optimizer_class=torch.optim.AdamW,
        learning_rate=float(cfg.learning_rate),
        use_amp="none",
        use_lr_sched="custom",
        grad_clip_norm=None,
        expr_name="prediction-nn-2",
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        num_iters=int(cfg.num_iters),
        save_every=int(cfg.save_every),
        eval_every=int(cfg.eval_every),
        eval_during=bool(cfg.eval_during),
        eval_during_num_iters=int(cfg.eval_during_num_iters),
        load_from_iter=None,
        log_every=50,
        mean_loss_length=200,
        root_dir=str(Path(run_root) / "run"),
        tensorboard_dir=str(Path(run_root) / "run" / "tb"),
        lr_scheduler=LRSchedulerConfig(
            start_warmup_factor=0.001,
            end_warmup_factor=1.0,
            warmup_iters=200,
            finish_decay_iter=int(cfg.num_iters),
            eta_min=1e-6,
        ),
        profiler=SimpleNamespace(
            profile_section="none",
            profile_dir=str(Path(cfg.root_dir) / "run" / "profile"),
            all_ranks=False,
            wait=0,
            warmup=0,
            active=0,
            repeat=0,
        ),
        evaluator=evaluator,
        eval_metric_fns=[builtin.eval_global_ic, builtin.eval_rank_ic, builtin.eval_distribution_scalars, _eval_mse_metric],
        train_metric_fns=[builtin.train_basic_metrics, builtin.train_timer_metrics],
        console_log_all_ranks=False,
    )
    return conf


def _list_checkpoint_iters(run_dir: Path) -> list[int]:
    """List checkpoint iteration numbers under one qmodel run directory."""
    # Scan ckpt files and parse iter numbers from `iter_<it>.pt` naming.
    ckpt_dir = Path(run_dir) / "run" / "ckpt"
    iters: list[int] = []
    for p in sorted(ckpt_dir.glob("iter_*.pt")):
        stem = p.stem
        parts = stem.split("_")
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        iters.append(int(parts[1]))
    if len(iters) == 0:
        return []
    return sorted(set(iters))

def _select_best_checkpoint_by_val(run_root: Path, qconf: SimpleNamespace, checkpoint_iters: list[int]) -> tuple[int, dict[int, dict[str, float]]]:
    """Evaluate validation metrics for checkpoints and pick the best one."""
    # Evaluate each checkpoint on the validation set and keep scalar metrics per iter.
    from qmodel.core.evaluator import Evaluator
    from qmodel.core.cpu_evaluator import CpuEvaluator

    metrics_by_it: dict[int, dict[str, float]] = {}
    if torch.device(qconf.device).type == "cuda":
        evaluator = Evaluator(qconf, group="val", writer=None, enable_logging=True)
    else:
        evaluator = CpuEvaluator(qconf, group="val", writer=None, enable_logging=True)
    for it in list(checkpoint_iters):
        metrics = evaluator.eval_single(int(it), n_iter=0, namespace="val")
        metrics_by_it[int(it)] = {str(k): float(v) for k, v in metrics.items()}
    evaluator.close()

    # Choose the best checkpoint by minimal validation MSE.
    def _val_mse(it: int) -> float:
        # Read MSE from computed metrics dict and require it to exist.
        m = metrics_by_it[int(it)]
        return float(m["val/objective/mse"])

    best_it = sorted(list(checkpoint_iters), key=_val_mse)[0]
    return int(best_it), metrics_by_it


def _load_val_metrics_from_disk(run_root: Path, checkpoint_iters: list[int]) -> tuple[int, dict[int, dict[str, float]]]:
    """Load persisted validation metrics and pick the best checkpoint."""
    # Read one metrics.json file per checkpoint and coerce all values to float.
    metrics_by_it: dict[int, dict[str, float]] = {}
    for it in list(checkpoint_iters):
        metrics_path = Path(run_root) / "run" / "eval_val" / f"iter_{int(it)}" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_by_it[int(it)] = {str(k): float(v) for k, v in dict(metrics).items()}

    # Choose the best checkpoint by minimal validation MSE.
    def _val_mse(it: int) -> float:
        # Read MSE from the loaded metric dict and require it to exist.
        m = metrics_by_it[int(it)]
        return float(m["val/objective/mse"])

    best_it = sorted(list(checkpoint_iters), key=_val_mse)[0]
    return int(best_it), metrics_by_it


def _prepare_split_inputs(
    cfg: PipelineConfig,
    *,
    out_root: Path,
    start_trade_date: int,
    end_trade_date: int,
    train_days: int,
    val_days: int,
    test_days: int,
) -> tuple[dict[str, object], float]:
    """Prepare split artifacts and return the prep payload plus elapsed seconds."""
    # Prepare output directories early so downstream code can assume they exist.
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "artifacts").mkdir(parents=True, exist_ok=True)

    # Disable minute sampling for sequence inputs so stock histories stay contiguous.
    prep_sample_stocks_per_minute = 0 if int(cfg.input_window_size) > 1 else int(cfg.sample_stocks_per_minute)

    # Run data preparation and persist NPZ splits plus data-clean artifacts.
    print(f"[pipeline] data_prep start out_root={out_root.as_posix()} start={start_trade_date} end={end_trade_date}", flush=True)
    t_prep0 = time.time()
    prep_cfg = DataPrepConfig(
        stock1m_dir=Path(cfg.stock1m_dir),
        out_dir=Path(out_root) / "artifacts",
        start_trade_date=int(start_trade_date),
        end_trade_date=int(end_trade_date),
        train_days=int(train_days),
        val_days=int(val_days),
        test_days=int(test_days),
        seed=int(cfg.seed),
        horizon_minutes=int(cfg.horizon_minutes),
        sample_stocks_per_minute=int(prep_sample_stocks_per_minute),
        use_cross_sectional_gaussianize=bool(cfg.use_cross_sectional_gaussianize),
        include_predict_split=bool(cfg.data_prep_include_predict_split),
        norm_fit_scope=str(cfg.data_prep_norm_fit_scope),
        days_per_call=int(cfg.data_prep_days_per_call),
        workers=int(cfg.data_prep_workers),
    )

    # Run data prep and rely on persisted artifacts to make reruns cheap.
    prep = prepare_npz_splits(prep_cfg)
    t_prep1 = time.time()
    progress = dict(prep["progress"])
    print(
        "[pipeline] data_prep done "
        f"seconds={float(t_prep1 - t_prep0):.2f} "
        f"elapsed_total={_format_seconds(float(progress['elapsed_seconds']))} "
        f"train_rows={int(prep['train_rows'])}",
        flush=True,
    )
    return prep, float(t_prep1 - t_prep0)


def _load_existing_test_eval_dir(qconf: SimpleNamespace, best_it: int) -> Path:
    """Load the existing test eval directory for the selected checkpoint."""
    # Resolve the expected test eval directory and require it to exist.
    test_eval_iter_dir = Path(qconf.root_dir) / "eval_test" / f"iter_{int(best_it)}"
    if not test_eval_iter_dir.exists():
        raise RuntimeError(f"Missing existing test eval dir: {test_eval_iter_dir}")
    return test_eval_iter_dir


def _load_existing_predict_manifest(qconf: SimpleNamespace, best_it: int) -> Path:
    """Load the existing predict manifest for the selected checkpoint."""
    # Resolve the expected predict manifest path and require it to exist.
    manifest_path = Path(qconf.root_dir) / "eval" / f"iter_{int(best_it)}" / "predict_manifest.yaml"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing existing predict manifest: {manifest_path}")
    return manifest_path


def _run_train_report_postprocess(
    cfg: PipelineConfig,
    prep: dict[str, object],
    prep_elapsed_seconds: float,
    out_root: Path,
    qconf: SimpleNamespace,
    best_it: int,
    val_metrics_by_it: dict[int, dict[str, float]],
    require_existing_eval: bool,
) -> None:
    """Build the train report from existing artifacts or freshly generated test eval outputs."""
    # Export a loss-curve plot from TensorBoard events for the training log requirement.
    loss_png = Path(out_root) / "train_loss.png"
    _export_train_loss_curve(Path(qconf.root_dir) / "tb", loss_png)

    # Resolve the test eval directory from disk or by running the evaluator once.
    if bool(require_existing_eval):
        test_eval_iter_dir = _load_existing_test_eval_dir(qconf, int(best_it))
    else:
        if torch.device(qconf.device).type == "cuda":
            from qmodel.core.evaluator import Evaluator

            evaluator = Evaluator(qconf, group="test", writer=None, enable_logging=False)
            evaluator.eval_single(int(best_it), n_iter=0, namespace="eval")
            evaluator.close()
        else:
            from qmodel.core.cpu_evaluator import CpuEvaluator

            evaluator = CpuEvaluator(qconf, group="test", writer=None, enable_logging=False)
            evaluator.eval_single(int(best_it), n_iter=0, namespace="eval")
            evaluator.close()
        test_eval_iter_dir = _move_eval_dir(Path(qconf.root_dir), src_name="eval", dst_name="eval_test", it=int(best_it))

    # Load test predictions and compute all required IC diagnostics.
    test_shard_path = Path(test_eval_iter_dir) / "rank0.feather"
    pred_df = load_eval_predictions(test_shard_path)
    pooled = pooled_ic(pred_df)
    test_ic_summary_yaml = Path(out_root) / "ic_time_series_summary.yaml"
    test_ic_summary = ic_time_series_summary(pred_df, test_ic_summary_yaml)

    # Emit test-side intraday and rolling IC diagnostics.
    intraday_csv = Path(out_root) / "intraday_ic.csv"
    intraday_png = Path(out_root) / "intraday_ic.png"
    intraday_time_series_ic(pred_df, intraday_csv, intraday_png)
    eval_cfg = EvalConfig(
        stock1m_dir=Path(cfg.stock1m_dir),
        window_size=int(cfg.rolling_window),
        step_size=int(cfg.rolling_step),
        horizon_minutes=int(cfg.horizon_minutes),
    )
    vol_csv = Path(out_root) / "vol_rolling_ic.csv"
    vol_png = Path(out_root) / "vol_rolling_ic.png"
    price_csv = Path(out_root) / "price_rolling_ic.csv"
    price_png = Path(out_root) / "price_rolling_ic.png"

    # Attach volatility and price labels once and reuse across test rolling curves.
    t_attach0 = time.time()
    labeled_df = attach_labels(pred_df, eval_cfg)
    t_attach1 = time.time()
    t_roll0 = time.time()
    vol_agg = volatility_rolling_ic(labeled_df, eval_cfg, vol_csv, vol_png)
    price_agg = price_rolling_ic(labeled_df, eval_cfg, price_csv, price_png)
    t_roll1 = time.time()

    # Persist a compact performance audit for the evaluation and report stage.
    device = torch.device(qconf.device)
    pin_memory = bool(device.type == "cuda")
    perf = {
        "data_prep": {
            "elapsed_seconds": float(prep_elapsed_seconds),
            "audit": prep["audit"],
            "audit_rates": prep["audit_rates"],
        },
        "train": {
            "device": str(device),
            "pin_memory": bool(pin_memory),
            "num_workers": int(cfg.num_workers),
        },
        "eval": {
            "attach_labels_seconds": float(t_attach1 - t_attach0),
            "rolling_ic_seconds": float(t_roll1 - t_roll0),
        },
    }
    import yaml

    (Path(out_root) / "perf_audit.yaml").write_text(yaml.safe_dump(perf, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Render and persist the train markdown report.
    report_path = Path(out_root) / "train_report.md"
    report = _render_train_report(
        cfg,
        prep,
        pooled,
        test_ic_summary,
        test_ic_summary_yaml,
        int(best_it),
        val_metrics_by_it,
        intraday_csv,
        vol_csv,
        price_csv,
        intraday_png,
        vol_png,
        price_png,
        loss_png,
        vol_agg,
        price_agg,
        perf,
    )
    report_path.write_text(report, encoding="utf-8")


def _render_train_report(
    cfg: PipelineConfig,
    prep: dict[str, object],
    pooled: dict[str, float],
    test_ic_summary: dict[str, object],
    test_ic_summary_yaml: Path,
    best_it: int,
    val_metrics_by_it: dict[int, dict[str, float]],
    intraday_csv: Path,
    vol_csv: Path,
    price_csv: Path,
    intraday_png: Path,
    vol_png: Path,
    price_png: Path,
    loss_png: Path,
    vol_agg,
    price_agg,
    perf: dict[str, object],
) -> str:
    """Render the train-report markdown content."""
    # Build a compact report body that focuses on training and test diagnostics only.
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    best_metrics = dict(val_metrics_by_it[int(best_it)])
    val_mse = float(best_metrics["val/objective/mse"])
    val_ic = float(best_metrics["val/quality/global_ic"])
    val_rank_ic = float(best_metrics["val/quality/rank_ic"])
    approx_epoch = _approx_epoch(int(best_it), int(prep["train_rows"]), int(cfg.batch_size))

    # Summarize test pooled and time-series IC stats for quick scanning.
    pooled_lines = (
        f"- Pearson IC: {pooled['pearson_ic']:.6f}\n"
        f"- Rank IC (Spearman): {pooled['rank_ic']:.6f}\n"
        f"- Count: {int(pooled['count'])}"
    )
    test_ic_lines = (
        f"- Pearson IC t-stat: {float(test_ic_summary['pearson_ic']['t_stat']):.4f}\n"
        f"- Pearson IC>0 占比: {float(test_ic_summary['pearson_ic']['positive_ratio']):.2%}\n"
        f"- Rank IC t-stat: {float(test_ic_summary['rank_ic']['t_stat']):.4f}\n"
        f"- Rank IC>0 占比: {float(test_ic_summary['rank_ic']['positive_ratio']):.2%}\n"
        f"- Timestamp Count: {int(test_ic_summary['timestamp_count'])}"
    )

    # Build a short val-sweep table sorted by MSE to document checkpoint selection.
    def _mse_row(it: int) -> tuple[int, float, float, float]:
        m = dict(val_metrics_by_it[int(it)])
        return int(it), float(m["val/objective/mse"]), float(m["val/quality/global_ic"]), float(m["val/quality/rank_ic"])

    sweep = sorted([_mse_row(it) for it in list(val_metrics_by_it.keys())], key=lambda r: float(r[1]))[:10]
    sweep_lines = ["| iter | val_mse | val_ic | val_rank_ic |", "| ---: | ---: | ---: | ---: |"]
    for it, mse, ic, ric in sweep:
        sweep_lines.append(f"| {int(it)} | {float(mse):.6e} | {float(ic):.6f} | {float(ric):.6f} |")

    # Render the final report body in Chinese with explicit artifact links.
    md = f"""# NN Train Report

- generated_at: `{now}`
- root_dir: `{Path(cfg.root_dir).as_posix()}`
- out_root: `{Path(prep['meta_path']).parent.parent.parent.as_posix()}`

## 数据清洗 (Data Clean)

- meta: `{Path(prep['meta_path']).as_posix()}`
- rows: train={int(prep['train_rows'])}, val={int(prep['val_rows'])}, test={int(prep['test_rows'])}
- audit_rates: train.kept_rate={float(prep['audit_rates']['train']['kept_rate']):.2%}, train.sampled_rate_vs_raw={float(prep['audit_rates']['train']['sampled_rate_vs_raw']):.2%}

## 模型训练 (Training)

- best_checkpoint: iter={int(best_it)}, approx_epoch={float(approx_epoch):.2f}
- val: MSE={float(val_mse):.6e}, IC={float(val_ic):.6f}, RankIC={float(val_rank_ic):.6f}

### Val Checkpoint Sweep (Top 10 by MSE)

{chr(10).join(sweep_lines)}

### Train Loss Curve

![]({loss_png.name})

## 测试集诊断 (Test Diagnostics)

### Test Pooled IC

{pooled_lines}

### Test IC Time Series Summary

{test_ic_lines}

- summary_yaml: `{Path(test_ic_summary_yaml).as_posix()}`

### Intraday IC

- csv: `{Path(intraday_csv).as_posix()}`
- png: `{Path(intraday_png).as_posix()}`

![]({intraday_png.name})

### Rolling IC

- volatility: `{Path(vol_csv).as_posix()}`, `{Path(vol_png).as_posix()}`
- price: `{Path(price_csv).as_posix()}`, `{Path(price_png).as_posix()}`
- volatility_desc: {vol_agg.desc}
- price_desc: {price_agg.desc}

![]({vol_png.name})

![]({price_png.name})

## 性能审计 (Performance Audit)

- data_prep_seconds={float(perf['data_prep']['elapsed_seconds']):.4f}
- attach_labels_seconds={float(perf['eval']['attach_labels_seconds']):.4f}
- rolling_ic_seconds={float(perf['eval']['rolling_ic_seconds']):.4f}
"""
    return md

def _approx_epoch(step: int, train_rows: int, batch_size: int) -> float:
    """Convert qmodel iteration step into an approximate epoch count."""
    # Map iterations to epochs by dividing by batches_per_epoch.
    batches_per_epoch = float(train_rows) / float(batch_size)
    return float(step) / float(batches_per_epoch)


def _describe_group_inflection(df: np.ndarray, ranks: np.ndarray) -> str:
    """Summarize the most salient group-curve inflections for report text."""
    # Identify extrema and the first zero-crossing to keep the output compact.
    if int(df.shape[0]) == 0:
        return "Empty curve (n < window_size)."

    i_max = int(np.nanargmax(df))
    i_min = int(np.nanargmin(df))
    max_rank = float(ranks[i_max])
    min_rank = float(ranks[i_min])
    max_val = float(df[i_max])
    min_val = float(df[i_min])

    s = np.sign(df)
    zc = np.where((s[:-1] <= 0) & (s[1:] > 0) | (s[:-1] >= 0) & (s[1:] < 0))[0]
    if int(zc.shape[0]) > 0:
        cross_rank = float(ranks[int(zc[0]) + 1])
        return f"max@{max_rank:.3f}({max_val:.4f}), min@{min_rank:.3f}({min_val:.4f}), first_sign_flip@{cross_rank:.3f}"
    return f"max@{max_rank:.3f}({max_val:.4f}), min@{min_rank:.3f}({min_val:.4f})"


def _load_prep_summary_from_meta(meta_path: Path) -> dict[str, object]:
    """Load a minimal prep summary dict from an existing `artifacts/npz/meta.yaml`."""
    # Parse meta.yaml and map it onto the keys expected by report stages.
    import yaml

    meta_path = Path(meta_path)
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    groups = dict(meta["storage"]["groups"])
    out_dir = meta_path.parent.parent
    stats_dir = Path(out_dir) / "data_clean"
    moment_path = stats_dir / "feature_moments.csv"
    moment_table = pd.read_csv(moment_path) if moment_path.exists() else pd.DataFrame([])
    out = {
        "done": True,
        "feature_names": list(meta["feature_names"]),
        "train_rows": int(groups["train"]["rows"]),
        "val_rows": int(groups["val"]["rows"]),
        "test_rows": int(groups["test"]["rows"]),
        "moment_table": moment_table,
        "meta_path": meta_path,
        "audit": dict(meta["audit"]),
        "audit_rates": dict(meta["audit_rates"]),
        "groups": sorted(list(groups.keys())),
    }
    if "predict" in groups:
        out["predict_rows"] = int(groups["predict"]["rows"])
    return out


def run_data_clean_stage(
    cfg: PipelineConfig,
    *,
    out_root: Path,
    start_trade_date: int,
    end_trade_date: int,
    train_days: int,
    val_days: int,
    test_days: int,
) -> tuple[dict[str, object], float]:
    """Run the data-clean stage and return (prep_summary, elapsed_seconds)."""
    # Build a stage fingerprint so repeated runs can skip the heavy prep work.
    stage_cfg = {
        "stage": "data_clean",
        "start_trade_date": int(start_trade_date),
        "end_trade_date": int(end_trade_date),
        "train_days": int(train_days),
        "val_days": int(val_days),
        "test_days": int(test_days),
        "use_cross_sectional_gaussianize": bool(cfg.use_cross_sectional_gaussianize),
        "include_predict_split": bool(cfg.data_prep_include_predict_split),
        "norm_fit_scope": str(cfg.data_prep_norm_fit_scope),
        "days_per_call": int(cfg.data_prep_days_per_call),
        "workers": int(cfg.data_prep_workers),
        "horizon_minutes": int(cfg.horizon_minutes),
        "sample_stocks_per_minute": int(cfg.sample_stocks_per_minute),
        "input_window_size": int(cfg.input_window_size),
        "seed": int(cfg.seed),
    }
    manifest_path = _stage_manifest_path(Path(out_root), "data_clean")
    fp = _stage_fingerprint(stage_cfg)

    # Skip the stage when the manifest and meta.yaml already match.
    meta_path = Path(out_root) / "artifacts" / "npz" / "meta.yaml"
    if manifest_path.exists() and meta_path.exists():
        m = _load_stage_manifest(manifest_path)
        if str(m.get("fingerprint")) == str(fp):
            return _load_prep_summary_from_meta(meta_path), float(m.get("elapsed_seconds", 0.0))

    # Run split data prep and persist the stage manifest after completion.
    prep, elapsed = _prepare_split_inputs(
        cfg,
        out_root=Path(out_root),
        start_trade_date=int(start_trade_date),
        end_trade_date=int(end_trade_date),
        train_days=int(train_days),
        val_days=int(val_days),
        test_days=int(test_days),
    )
    _write_stage_manifest(
        manifest_path,
        {
            "stage": "data_clean",
            "fingerprint": str(fp),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": float(elapsed),
            "meta_path": str(meta_path.as_posix()),
        },
    )
    return prep, float(elapsed)


def run_clean_report_stage(cfg: PipelineConfig, *, out_root: Path) -> Path:
    """Render the clean report from existing data-clean artifacts."""
    # Build a stage fingerprint so report rebuild can skip when config stays unchanged.
    stage_cfg = {
        "stage": "clean_report",
        "use_cross_sectional_gaussianize": bool(cfg.use_cross_sectional_gaussianize),
        "norm_fit_scope": str(cfg.data_prep_norm_fit_scope),
    }
    manifest_path = _stage_manifest_path(Path(out_root), "clean_report")
    fp = _stage_fingerprint(stage_cfg)

    # Skip the stage when the manifest and report path already match.
    meta_path = Path(out_root) / "artifacts" / "npz" / "meta.yaml"
    report_path = Path(out_root) / "artifacts" / "data_clean" / "report.md"
    if manifest_path.exists() and report_path.exists():
        m = _load_stage_manifest(manifest_path)
        if str(m.get("fingerprint")) == str(fp):
            return report_path

    # Render the clean report purely from persisted artifacts.
    out_path = render_clean_report_from_meta(meta_path)
    _write_stage_manifest(
        manifest_path,
        {
            "stage": "clean_report",
            "fingerprint": str(fp),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "report_path": str(out_path.as_posix()),
        },
    )
    return out_path


def run_train_stage(
    cfg: PipelineConfig,
    *,
    out_root: Path,
    feature_dim: int,
) -> tuple[SimpleNamespace, int, dict[int, dict[str, float]]]:
    """Run NN training (skip if complete) and return (qconf, best_it, val_metrics_by_it)."""
    # Build a stage fingerprint so training can skip cleanly across repeated invocations.
    stage_cfg = {
        "stage": "train",
        "seed": int(cfg.seed),
        "num_iters": int(cfg.num_iters),
        "save_every": int(cfg.save_every),
        "batch_size": int(cfg.batch_size),
        "learning_rate": float(cfg.learning_rate),
        "hidden_dims": list(cfg.hidden_dims),
        "dropout": float(cfg.dropout),
        "input_window_size": int(cfg.input_window_size),
        "feature_dim": int(feature_dim),
    }
    manifest_path = _stage_manifest_path(Path(out_root), "train")
    fp = _stage_fingerprint(stage_cfg)

    # Always build qmodel config so checkpoint paths resolve consistently.
    qconf = _build_qmodel_config(cfg, feature_dim=int(feature_dim), run_root=Path(out_root))
    final_iter = int(cfg.num_iters) - 1

    # Skip training when the final checkpoint already exists and the stage manifest matches.
    ckpt_iters_before = _list_checkpoint_iters(Path(out_root))
    last_ckpt = int(max(ckpt_iters_before)) if len(ckpt_iters_before) else -1
    if manifest_path.exists() and int(last_ckpt) >= int(final_iter):
        m = _load_stage_manifest(manifest_path)
        if str(m.get("fingerprint")) == str(fp):
            best_it = int(m["best_it"])
            val_metrics_by_it = _load_val_metrics_from_disk(Path(out_root), _list_checkpoint_iters(Path(out_root)))[1]
            return qconf, int(best_it), dict(val_metrics_by_it)

    # Advance training chunk-by-chunk until the final checkpoint is materialized.
    if int(last_ckpt) >= int(final_iter):
        print(f"[pipeline] train skip already_complete last_ckpt={last_ckpt} final_iter={final_iter}", flush=True)
    while int(last_ckpt) < int(final_iter):
        # Decide the next target iteration as the next save boundary, capped at the final iteration.
        step = int(cfg.save_every)
        next_target = int(min(int(final_iter), int(last_ckpt + step) if int(last_ckpt) >= 0 else int(step)))
        chunk_save_every = int(cfg.save_every)
        if int(next_target) == int(final_iter) and int(final_iter) % int(cfg.save_every) != 0:
            chunk_save_every = int(max(int(final_iter), 1))
        qconf.num_iters = int(next_target + 1)
        qconf.save_every = int(chunk_save_every)
        qconf.load_from_iter = -1 if int(last_ckpt) >= 0 else None
        print(
            f"[pipeline] train chunk start last_ckpt={last_ckpt} target_iter={next_target} "
            f"save_every={int(chunk_save_every)} batch_size={int(cfg.batch_size)}",
            flush=True,
        )

        # Run one trainer chunk on the selected device.
        if torch.device(qconf.device).type == "cuda":
            from qmodel.core.trainer import Trainer

            trainer = Trainer(qconf)
            trainer.train()
        else:
            from qmodel.core.cpu_trainer import CpuTrainer

            trainer = CpuTrainer(qconf)
            trainer.train()

        # Refresh checkpoint progress and continue until the final iteration is saved.
        ckpt_iters_after = _list_checkpoint_iters(Path(out_root))
        last_ckpt = int(max(ckpt_iters_after)) if len(ckpt_iters_after) else -1
        print(f"[pipeline] train chunk done last_ckpt={last_ckpt} final_iter={final_iter}", flush=True)

    # Evaluate validation metrics for all checkpoints and pick the best one.
    ckpt_iters = _list_checkpoint_iters(Path(out_root))
    best_it, val_metrics_by_it = _select_best_checkpoint_by_val(Path(out_root), qconf, ckpt_iters)
    _write_stage_manifest(
        manifest_path,
        {
            "stage": "train",
            "fingerprint": str(fp),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_iter": int(final_iter),
            "best_it": int(best_it),
        },
    )
    return qconf, int(best_it), dict(val_metrics_by_it)


def run_train_report_stage(
    cfg: PipelineConfig,
    *,
    out_root: Path,
    prep: dict[str, object],
    prep_elapsed_seconds: float,
    qconf: SimpleNamespace,
    best_it: int,
    val_metrics_by_it: dict[int, dict[str, float]],
    require_existing_eval: bool,
) -> Path:
    """Build the train report and return its markdown path."""
    # Build a stage fingerprint so report rebuild can skip when inputs are unchanged.
    stage_cfg = {"stage": "train_report", "best_it": int(best_it)}
    manifest_path = _stage_manifest_path(Path(out_root), "train_report")
    fp = _stage_fingerprint(stage_cfg)
    report_path = Path(out_root) / "train_report.md"

    # Skip the stage when the manifest and report already match.
    if manifest_path.exists() and report_path.exists():
        m = _load_stage_manifest(manifest_path)
        if str(m.get("fingerprint")) == str(fp):
            return report_path

    # Run the postprocess that generates test diagnostics and writes train_report.md.
    _run_train_report_postprocess(
        cfg,
        prep,
        float(prep_elapsed_seconds),
        Path(out_root),
        qconf,
        int(best_it),
        dict(val_metrics_by_it),
        bool(require_existing_eval),
    )
    _write_stage_manifest(
        manifest_path,
        {"stage": "train_report", "fingerprint": str(fp), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "report_path": str(report_path.as_posix())},
    )
    return report_path


def run_predict_eval_stage(cfg: PipelineConfig, *, out_root: Path, qconf: SimpleNamespace, best_it: int, require_existing_eval: bool) -> Path:
    """Run predict evaluator (or reuse existing) and return the predict manifest path."""
    # Build a stage fingerprint so predict evaluator can be rerun deterministically.
    stage_cfg = {"stage": "predict_eval", "best_it": int(best_it)}
    manifest_path = _stage_manifest_path(Path(out_root), "predict_eval")
    fp = _stage_fingerprint(stage_cfg)

    # Resolve the expected predict manifest output path.
    predict_manifest_path = Path(qconf.root_dir) / "eval" / f"iter_{int(best_it)}" / "predict_manifest.yaml"
    if manifest_path.exists() and predict_manifest_path.exists():
        m = _load_stage_manifest(manifest_path)
        if str(m.get("fingerprint")) == str(fp):
            return predict_manifest_path

    # Require an existing manifest for report-only modes.
    if bool(require_existing_eval):
        return _load_existing_predict_manifest(qconf, int(best_it))

    # Run the predict evaluator to produce the manifest and chunk artifacts.
    if torch.device(qconf.device).type == "cuda":
        from qmodel.core.evaluator import Evaluator

        predictor = Evaluator(qconf, group="predict", writer=None, enable_logging=False)
        predictor.eval_single(int(best_it), n_iter=0, namespace="predict")
        predictor.close()
    else:
        from qmodel.core.cpu_evaluator import CpuEvaluator

        predictor = CpuEvaluator(qconf, group="predict", writer=None, enable_logging=False)
        predictor.eval_single(int(best_it), n_iter=0, namespace="predict")
        predictor.close()
    _write_stage_manifest(
        manifest_path,
        {
            "stage": "predict_eval",
            "fingerprint": str(fp),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "predict_manifest_path": str(predict_manifest_path.as_posix()),
        },
    )
    return predict_manifest_path


def run_predict_report_from_manifest(manifest_path: Path, cfg: PipelineConfig, out_root: Path) -> Path:
    """Compute predict report artifacts from an existing predict manifest."""
    # Compute the predict report in a dedicated folder so heavy artifacts stay isolated.
    out_root = Path(out_root)
    manifest_path = Path(manifest_path)
    report_dir = Path(out_root) / "predict_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build evaluator config and compute the full predict report from the manifest.
    eval_cfg = EvalConfig(
        stock1m_dir=Path(cfg.stock1m_dir),
        window_size=int(cfg.rolling_window),
        step_size=int(cfg.rolling_step),
        horizon_minutes=int(cfg.horizon_minutes),
    )
    artifacts = compute_predict_report_from_manifest(Path(manifest_path), eval_cfg, Path(report_dir))

    # Render a minimal markdown wrapper that links to the produced artifacts.
    md_path = Path(out_root) / "predict_report.md"
    md = _render_predict_report(cfg, manifest_path, artifacts, report_dir)
    md_path.write_text(md, encoding="utf-8")
    return md_path


def _render_predict_report(cfg: PipelineConfig, manifest_path: Path, artifacts, report_dir: Path) -> str:
    """Render the predict-report markdown content."""
    # Summarize the heavy predict diagnostics into a compact markdown report.
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    pooled = dict(artifacts.pooled)
    ic_summary = dict(artifacts.ic_summary)
    turnover_summary = dict(artifacts.turnover_summary)
    residual_summary = dict(artifacts.residual_summary)
    annual_tbl = artifacts.annual_tbl
    annual_years = annual_tbl["year"].astype(int).tolist()
    year_range = f"{int(annual_years[0])}-{int(annual_years[-1])}" if int(annual_years[0]) != int(annual_years[-1]) else f"{int(annual_years[0])}"
    md = f"""# Predict Report

- generated_at: `{now}`
- manifest: `{Path(manifest_path).as_posix()}`
- report_dir: `{Path(report_dir).as_posix()}`

## Pooled

- Pearson IC: {float(pooled['pearson_ic']):.6f}
- Rank IC (Spearman): {float(pooled['rank_ic']):.6f}
- Count: {int(pooled['count'])}

## IC Time Series Summary

- ic_summary_yaml: `{Path(artifacts.ic_summary_yaml).as_posix()}`
- pearson_t_stat: {float(ic_summary['pearson_ic']['t_stat']):.4f}
- rank_t_stat: {float(ic_summary['rank_ic']['t_stat']):.4f}

## Annual IC ({year_range})

- annual_csv: `{Path(artifacts.annual_csv).as_posix()}`
- annual_png: `{Path(artifacts.annual_png).as_posix()}`

![]({Path(artifacts.annual_png).name})

## Intraday IC

- intraday_csv: `{Path(artifacts.intraday_csv).as_posix()}`
- intraday_png: `{Path(artifacts.intraday_png).as_posix()}`

![]({Path(artifacts.intraday_png).name})

## Rolling IC

- vol_csv: `{Path(artifacts.vol_csv).as_posix()}`
- vol_png: `{Path(artifacts.vol_png).as_posix()}`
- price_csv: `{Path(artifacts.price_csv).as_posix()}`
- price_png: `{Path(artifacts.price_png).as_posix()}`

![]({Path(artifacts.vol_png).name})

![]({Path(artifacts.price_png).name})

## Rank Diagnostics

- rank_png: `{Path(artifacts.rank_png).as_posix()}`

![]({Path(artifacts.rank_png).name})

## Turnover

- turnover_csv: `{Path(artifacts.turnover_csv).as_posix()}`
- turnover_png: `{Path(artifacts.turnover_png).as_posix()}`
- summary: {turnover_summary}

![]({Path(artifacts.turnover_png).name})

## Residual

- residual_yaml: `{Path(artifacts.residual_yaml).as_posix()}`
- residual_png: `{Path(artifacts.residual_png).as_posix()}`
- summary: {residual_summary}

![]({Path(artifacts.residual_png).name})
"""
    return md


def run_predict_report_stage(cfg: PipelineConfig, *, out_root: Path, predict_manifest_path: Path) -> Path:
    """Build the predict report from an existing predict manifest and return its path."""
    # Build a stage fingerprint so report rebuild can skip when manifest stays the same.
    stage_cfg = {"stage": "predict_report", "manifest": str(Path(predict_manifest_path).as_posix())}
    manifest_path = _stage_manifest_path(Path(out_root), "predict_report")
    fp = _stage_fingerprint(stage_cfg)
    report_md = Path(out_root) / "predict_report.md"

    # Skip the stage when the manifest and report already match.
    if manifest_path.exists() and report_md.exists():
        m = _load_stage_manifest(manifest_path)
        if str(m.get("fingerprint")) == str(fp):
            return report_md

    # Compute predict report purely from the predict manifest.
    out_path = run_predict_report_from_manifest(Path(predict_manifest_path), cfg, Path(out_root))
    _write_stage_manifest(
        manifest_path,
        {"stage": "predict_report", "fingerprint": str(fp), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "report_path": str(out_path.as_posix())},
    )
    return out_path


def _run_single_split_staged(cfg: PipelineConfig, *, out_root: Path, start_trade_date: int, end_trade_date: int, train_days: int, val_days: int, test_days: int) -> None:
    """Run one split in the configured pipeline mode using explicit stages."""
    # Dispatch stage execution based on the configured pipeline_mode string.
    mode = str(cfg.pipeline_mode)
    out_root = Path(out_root)

    if mode == "train_full":
        # Run data clean, clean report, train, and train report without predict heavy stages.
        prep, prep_elapsed = run_data_clean_stage(
            cfg,
            out_root=out_root,
            start_trade_date=int(start_trade_date),
            end_trade_date=int(end_trade_date),
            train_days=int(train_days),
            val_days=int(val_days),
            test_days=int(test_days),
        )
        run_clean_report_stage(cfg, out_root=out_root)
        qconf, best_it, val_metrics_by_it = run_train_stage(cfg, out_root=out_root, feature_dim=len(prep["feature_names"]))
        run_train_report_stage(
            cfg,
            out_root=out_root,
            prep=prep,
            prep_elapsed_seconds=float(prep_elapsed),
            qconf=qconf,
            best_it=int(best_it),
            val_metrics_by_it=dict(val_metrics_by_it),
            require_existing_eval=False,
        )
        return

    if mode == "train_report_only":
        # Rebuild train report from disk artifacts without rerunning data prep or training.
        meta_path = Path(out_root) / "artifacts" / "npz" / "meta.yaml"
        prep = _load_prep_summary_from_meta(meta_path)
        qconf = _build_qmodel_config(cfg, feature_dim=len(prep["feature_names"]), run_root=Path(out_root))
        ckpt_iters = _list_checkpoint_iters(Path(out_root))
        best_it, val_metrics_by_it = _load_val_metrics_from_disk(Path(out_root), ckpt_iters)
        run_train_report_stage(
            cfg,
            out_root=out_root,
            prep=prep,
            prep_elapsed_seconds=float(prep["audit"]["train"]["elapsed_seconds"]) + float(prep["audit"]["val"]["elapsed_seconds"]) + float(prep["audit"]["test"]["elapsed_seconds"]),
            qconf=qconf,
            best_it=int(best_it),
            val_metrics_by_it=dict(val_metrics_by_it),
            require_existing_eval=True,
        )
        return

    if mode == "predict_full":
        # Run the full heavy pipeline including predict evaluator and predict report.
        prep, prep_elapsed = run_data_clean_stage(
            cfg,
            out_root=out_root,
            start_trade_date=int(start_trade_date),
            end_trade_date=int(end_trade_date),
            train_days=int(train_days),
            val_days=int(val_days),
            test_days=int(test_days),
        )
        run_clean_report_stage(cfg, out_root=out_root)
        qconf, best_it, val_metrics_by_it = run_train_stage(cfg, out_root=out_root, feature_dim=len(prep["feature_names"]))
        run_train_report_stage(
            cfg,
            out_root=out_root,
            prep=prep,
            prep_elapsed_seconds=float(prep_elapsed),
            qconf=qconf,
            best_it=int(best_it),
            val_metrics_by_it=dict(val_metrics_by_it),
            require_existing_eval=False,
        )
        predict_manifest_path = run_predict_eval_stage(cfg, out_root=out_root, qconf=qconf, best_it=int(best_it), require_existing_eval=False)
        run_predict_report_stage(cfg, out_root=out_root, predict_manifest_path=predict_manifest_path)
        return

    if mode == "predict_report_only":
        # Rebuild predict report from an existing manifest without rerunning training or data clean.
        meta_path = Path(out_root) / "artifacts" / "npz" / "meta.yaml"
        prep = _load_prep_summary_from_meta(meta_path)
        qconf = _build_qmodel_config(cfg, feature_dim=len(prep["feature_names"]), run_root=Path(out_root))
        ckpt_iters = _list_checkpoint_iters(Path(out_root))
        best_it, _val_metrics_by_it = _load_val_metrics_from_disk(Path(out_root), ckpt_iters)
        predict_manifest_path = run_predict_eval_stage(cfg, out_root=out_root, qconf=qconf, best_it=int(best_it), require_existing_eval=True)
        run_predict_report_stage(cfg, out_root=out_root, predict_manifest_path=predict_manifest_path)
        return

    raise RuntimeError(f"Unknown pipeline_mode: {mode}")


def _render_report(
    cfg: PipelineConfig,
    prep: dict[str, object],
    pooled: dict[str, float],
    test_ic_summary: dict[str, object],
    test_ic_summary_yaml: Path,
    best_it: int,
    val_metrics_by_it: dict[int, dict[str, float]],
    intraday_csv: Path,
    vol_csv: Path,
    price_csv: Path,
    intraday_png: Path,
    vol_png: Path,
    price_png: Path,
    rank_png: Path,
    loss_png: Path,
    vol_agg,
    price_agg,
    predict_pooled: dict[str, float],
    annual_tbl,
    annual_csv: Path,
    annual_png: Path,
    predict_ic_summary: dict[str, object],
    predict_ic_summary_yaml: Path,
    predict_intraday_csv: Path,
    predict_intraday_png: Path,
    predict_vol_csv: Path,
    predict_vol_png: Path,
    predict_price_csv: Path,
    predict_price_png: Path,
    turnover_csv: Path,
    turnover_png: Path,
    turnover_yaml: Path,
    residual_yaml: Path,
    residual_png: Path,
    turnover_summary: dict[str, object],
    residual_summary: dict[str, object],
    perf: dict[str, object],
) -> str:
    """Render the final markdown report content."""
    # Build a compact report body with conclusion-driven summaries.
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    best_metrics = dict(val_metrics_by_it[int(best_it)])
    val_mse = float(best_metrics["val/objective/mse"])
    val_ic = float(best_metrics["val/quality/global_ic"])
    val_rank_ic = float(best_metrics["val/quality/rank_ic"])
    approx_epoch = _approx_epoch(int(best_it), int(prep["train_rows"]), int(cfg.batch_size))
    pooled_lines = (
        f"- Pearson IC: {pooled['pearson_ic']:.6f}\n"
        f"- Rank IC (Spearman): {pooled['rank_ic']:.6f}\n"
        f"- Count: {int(pooled['count'])}"
    )
    test_ic_lines = (
        f"- Pearson IC t-stat: {float(test_ic_summary['pearson_ic']['t_stat']):.4f}\n"
        f"- Pearson IC>0 占比: {float(test_ic_summary['pearson_ic']['positive_ratio']):.2%}\n"
        f"- Rank IC t-stat: {float(test_ic_summary['rank_ic']['t_stat']):.4f}\n"
        f"- Rank IC>0 占比: {float(test_ic_summary['rank_ic']['positive_ratio']):.2%}\n"
        f"- Timestamp Count: {int(test_ic_summary['timestamp_count'])}"
    )
    predict_lines = (
        f"- Pearson IC: {predict_pooled['pearson_ic']:.6f}\n"
        f"- Rank IC (Spearman): {predict_pooled['rank_ic']:.6f}\n"
        f"- Count: {int(predict_pooled['count'])}"
    )
    predict_ic_lines = (
        f"- Pearson IC t-stat: {float(predict_ic_summary['pearson_ic']['t_stat']):.4f}\n"
        f"- Pearson IC>0 占比: {float(predict_ic_summary['pearson_ic']['positive_ratio']):.2%}\n"
        f"- Rank IC t-stat: {float(predict_ic_summary['rank_ic']['t_stat']):.4f}\n"
        f"- Rank IC>0 占比: {float(predict_ic_summary['rank_ic']['positive_ratio']):.2%}\n"
        f"- Timestamp Count: {int(predict_ic_summary['timestamp_count'])}"
    )
    split_meta_path = Path(prep["meta_path"])
    import yaml

    split_meta = yaml.safe_load(split_meta_path.read_text(encoding="utf-8"))
    audit = dict(split_meta["audit"])
    audit_rates = dict(split_meta["audit_rates"])
    dates = dict(split_meta["dates"])
    invalid_values = dict(split_meta["invalid_values"])
    data_clean_feature_moments_path = split_meta_path.parent.parent / "data_clean" / "feature_moments.csv"
    data_clean_report_rel = Path("artifacts") / "data_clean" / "report.md"
    data_clean_feature_moments_rel = Path("artifacts") / "data_clean" / "feature_moments.csv"
    data_clean_pooled_png_rel = Path("artifacts") / "data_clean" / "pooled_feature_grid.png"
    tb_dir = split_meta_path.parent.parent.parent / "run" / "tb"
    invalid_stats_path = split_meta_path.parent.parent / str(invalid_values["stats_path"])
    invalid_report_path = split_meta_path.parent.parent / str(invalid_values["report_path"])
    invalid_tbl = pd.read_csv(invalid_stats_path)
    moment_tbl = pd.read_csv(data_clean_feature_moments_path)

    def _range_str(xs: list[int]) -> str:
        # Format a compact inclusive date range string.
        if len(xs) == 0:
            return "[]"
        return f"[{int(xs[0])}, {int(xs[-1])}] ({len(xs)} days)"

    def _missing_str(a: dict[str, object], r: dict[str, object]) -> str:
        # Convert raw/kept/sampled counters into simple missing rates.
        raw = int(a["raw_rows"])
        kept = int(a["kept_rows"])
        sampled = int(a["sampled_rows"])
        feat_drop = 1.0 - float(r["kept_rate"])
        samp_drop = 1.0 - float(r["sampled_rate_vs_kept"])
        return f"raw={raw}, kept={kept} (drop={feat_drop:.2%}), sampled={sampled} (drop={samp_drop:.2%})"

    def _invalid_row_str(row: dict[str, object]) -> str:
        # Format one invalid-stat row into a compact report line.
        return (
            f"{str(row['field'])} ({str(row['field_type'])}): "
            f"nan_ratio={float(row['nan_ratio']):.4%}, "
            f"inf_ratio={float(row['inf_ratio']):.4%}, "
            f"invalid_ratio={float(row['invalid_ratio']):.4%}, "
            f"skew={float(row['skew_finite']):.4f}, "
            f"kurtosis={float(row['kurtosis_finite']):.4f}"
        )

    # Summarize group curve inflections for volatility and price buckets.
    vol_desc = "Empty curve (n < window_size)."
    if hasattr(vol_agg, "shape") and int(vol_agg.shape[0]) > 0:
        vol_desc = _describe_group_inflection(
            vol_agg["mean_ic"].to_numpy(dtype=float),
            vol_agg["group_center_rank"].to_numpy(dtype=float),
        )
    price_desc = "Empty curve (n < window_size)."
    if hasattr(price_agg, "shape") and int(price_agg.shape[0]) > 0:
        price_desc = _describe_group_inflection(
            price_agg["mean_ic"].to_numpy(dtype=float),
            price_agg["group_center_rank"].to_numpy(dtype=float),
        )

    # Extract the first and last available annual IC rows for a regime comparison summary.
    annual_idx = annual_tbl.set_index("year")
    annual_years = annual_tbl["year"].astype(int).tolist()
    first_year = int(annual_years[0])
    last_year = int(annual_years[-1])
    first_year_ic = float(annual_idx.loc[int(first_year), "pearson_ic"])
    last_year_ic = float(annual_idx.loc[int(last_year), "pearson_ic"])
    first_year_rank_ic = float(annual_idx.loc[int(first_year), "rank_ic"])
    last_year_rank_ic = float(annual_idx.loc[int(last_year), "rank_ic"])
    predict_year_range = f"{int(first_year)}-{int(last_year)}" if int(first_year) != int(last_year) else f"{int(first_year)}"

    # Summarize the most important invalid-value rows for the data-clean section.
    invalid_top = invalid_tbl.sort_values(["invalid_ratio", "field"], ascending=[False, True], kind="stable").reset_index(drop=True)
    invalid_lines = "\n".join([f"- {_invalid_row_str(row)}" for row in invalid_top.head(5).to_dict(orient="records")])

    # Summarize the post-clean pooled distribution rows with the largest residual tails.
    moment_top = moment_tbl.sort_values(["kurtosis", "feature"], ascending=[False, True], key=lambda s: s.abs() if s.name == "kurtosis" else s, kind="stable").reset_index(drop=True)
    moment_lines = "\n".join(
        [
            f"- {str(row['feature'])}: mean={float(row['mean']):.4f}, std={float(row['std']):.4f}, "
            f"skew={float(row['skew']):.4f}, kurtosis={float(row['kurtosis']):.4f}"
            for row in moment_top.head(5).to_dict(orient="records")
        ]
    )

    # Summarize the predict prediction-rank turnover in one compact paragraph.
    turnover_lines = (
        f"- Mean turnover: {float(turnover_summary['mean_rank_turnover']):.6f}\n"
        f"- Mean rank corr: {float(turnover_summary['mean_rank_corr']):.6f}\n"
        f"- Positive rank corr 占比: {float(turnover_summary['positive_rank_corr_ratio']):.2%}\n"
        f"- Lowest turnover time: {int(turnover_summary['lowest_turnover_time'])}, value={float(turnover_summary['lowest_turnover_value']):.6f}\n"
        f"- Highest turnover time: {int(turnover_summary['highest_turnover_time'])}, value={float(turnover_summary['highest_turnover_value']):.6f}"
    )

    # Summarize the predict residual distribution in one compact paragraph.
    residual_lines = (
        f"- Residual mean: {float(residual_summary['residual_mean']):.6e}\n"
        f"- Residual std: {float(residual_summary['residual_std']):.6e}\n"
        f"- Residual skew/kurtosis: {float(residual_summary['residual_skew']):.4f} / {float(residual_summary['residual_kurtosis']):.4f}\n"
        f"- MAE / RMSE: {float(residual_summary['mae']):.6e} / {float(residual_summary['rmse']):.6e}\n"
        f"- Corr(prediction, residual): {float(residual_summary['corr_prediction_residual']):.6f}"
    )

    # Render a structured markdown report matching the required sections.
    md = f"""# 神经网络预测与多维度 IC 评估报告

生成时间: {now}

## 结论摘要 (Conclusion)

- 最佳 Checkpoint: iter={best_it} (approx_epoch={approx_epoch:.2f}), Val MSE={val_mse:.6e}, Val IC={val_ic:.6f}, Val RankIC={val_rank_ic:.6f}.
- Test Pooled IC: Pearson={pooled['pearson_ic']:.6f}, RankIC={pooled['rank_ic']:.6f}, Count={int(pooled['count'])}.
- Predict Pooled IC ({predict_year_range} 全周期): Pearson={predict_pooled['pearson_ic']:.6f}, RankIC={predict_pooled['rank_ic']:.6f}, Count={int(predict_pooled['count'])}.
- Annual IC 对比: {first_year} Pearson={first_year_ic:.6f}, RankIC={first_year_rank_ic:.6f}; {last_year} Pearson={last_year_ic:.6f}, RankIC={last_year_rank_ic:.6f}.
- Volatility 分组拐点: {vol_desc}.
- Price 分组拐点: {price_desc}.

## 数据集元数据 (Dataset Metadata)

- Split 配置文件: `{split_meta_path.as_posix()}`
- 日期范围: train={_range_str(list(dates['train']))}, val={_range_str(list(dates['val']))}, test={_range_str(list(dates['test']))}, predict={_range_str(list(dates['predict']))}.
- 样本量: train={int(prep['train_rows'])}, val={int(prep['val_rows'])}, test={int(prep['test_rows'])}, predict={int(prep['predict_rows'])}.
- 缺失/采样审计: train={_missing_str(dict(audit['train']), dict(audit_rates['train']))}, val={_missing_str(dict(audit['val']), dict(audit_rates['val']))}, test={_missing_str(dict(audit['test']), dict(audit_rates['test']))}, predict={_missing_str(dict(audit['predict']), dict(audit_rates['predict']))}.

## Data Clean Invalid Audit

- Invalid 数值表: `{invalid_stats_path.as_posix()}`
- Invalid 报告: `{invalid_report_path.as_posix()}`
{invalid_lines}

## Data Clean Pooled Distribution

- Data clean 报告: `{data_clean_report_rel.as_posix()}`
- Feature moments: `{data_clean_feature_moments_rel.as_posix()}`
- Pooled 分布图: `{data_clean_pooled_png_rel.as_posix()}`
{moment_lines}

![]({data_clean_pooled_png_rel.as_posix()})

## 模型与训练协议 (Experiment Protocol)

- 训练设备: `{str(perf['train']['device'])}`, DataLoader `pin_memory={bool(perf['train']['pin_memory'])}`, `num_workers={int(perf['train']['num_workers'])}`.
- 模型结构: `MLP`, hidden layers 使用 `ReLU(inplace=True)`, 输出层为线性回归头, 不额外添加 activation.
- Optimizer: `AdamW`, base learning rate=`{float(cfg.learning_rate):.6g}`.
- LR Scheduler: 已启用, `use_lr_sched="custom"`, 采用 `Linear Warmup + Cosine Annealing`; `warmup_iters=200`, `start_factor=0.001`, `end_factor=1.0`, `finish_decay_iter={int(cfg.num_iters)}`, `eta_min=1e-6`.
- Checkpoint 选择: 仅使用 Validation (`val/objective/mse`) 选取最佳 iter, Test 仅做一次性最终评估.
- TensorBoard: `{tb_dir.as_posix()}` (loss 标量: `train/objective/loss`).

## 模型性能 (Model Performance)

- 最佳 Checkpoint: iter={best_it}, approx_epoch={approx_epoch:.2f}.
- Val: MSE={val_mse:.6e}, IC={val_ic:.6f}, RankIC={val_rank_ic:.6f}.
- Test Pooled:\n{pooled_lines}
- Test IC 时序诊断:\n{test_ic_lines}
- Predict Pooled:\n{predict_lines}
- Predict IC 时序诊断:\n{predict_ic_lines}

## 分组结论 (Group Findings)

- Volatility rolling IC: {vol_desc}.
- Price rolling IC: {price_desc}.

## 诊断补充 (Diagnostics)

- Test IC 时序摘要: `{test_ic_summary_yaml.as_posix()}`
- Predict IC 时序摘要: `{predict_ic_summary_yaml.as_posix()}`
- Prediction rank turnover 数值表: `{turnover_csv.as_posix()}`
- Prediction rank turnover 摘要: `{turnover_yaml.as_posix()}`
- Residual diagnostics 摘要: `{residual_yaml.as_posix()}`
{turnover_lines}
{residual_lines}

### 公式补充

- IC t-stat: `mean(IC_t) / (std(IC_t) / sqrt(T))`
- IC>0 占比: `mean(1[IC_t > 0])`
- 排序换手: `1 - corr(rank_t, rank_t-1)`
- 残差: `residual = target - prediction`

### 图表补充

#### Prediction Rank Turnover

![]({turnover_png.name})

#### Residual Diagnostics

![]({residual_png.name})

## 性能审计 (Performance Audit)

- data_prep_seconds={float(perf['data_prep']['elapsed_seconds']):.4f}, test_attach_labels_seconds={float(perf['eval']['attach_labels_seconds']):.4f}, test_rolling_ic_seconds={float(perf['eval']['rolling_ic_seconds']):.4f}.
- predict_row_count={int(perf['eval']['predict_row_count'])}, predict_chunk_count={int(perf['eval']['predict_chunk_count'])}, predict_stream_write_seconds={float(perf['eval']['predict_stream_write_seconds']):.4f}, predict_stream_report_seconds={float(perf['eval']['predict_stream_report_seconds']):.4f}.

## 文件输出 (Artifacts)

- `{intraday_csv.as_posix()}`
- `{vol_csv.as_posix()}`
- `{price_csv.as_posix()}`
- `{annual_csv.as_posix()}`

### 四张核心图表

1. Intraday IC Curve: `{intraday_png.as_posix()}`
2. Volatility Rolling IC: `{vol_png.as_posix()}`
3. Price Rolling IC: `{price_png.as_posix()}`
4. Predict 预测收益率 vs 实际收益率 Rank 曲线: `{rank_png.as_posix()}`

### 核心图表预览

#### Prediction Rank Curve

![]({rank_png.name})

### 训练与长周期诊断

1. Train Loss Curve: `{loss_png.as_posix()}`
2. Annual Pooled IC: `{annual_png.as_posix()}`
3. Predict Intraday IC: `{predict_intraday_png.as_posix()}`
4. Predict Volatility Rolling IC: `{predict_vol_png.as_posix()}`
5. Predict Price Rolling IC: `{predict_price_png.as_posix()}`
"""
    return md


def _default_config() -> PipelineConfig:
    """Build the module-level default pipeline config."""
    # Define a small default experiment that is GPU-feasible while remaining fully general.
    return PipelineConfig(
        pipeline_mode="train_full",
        root_dir=Path("outputs") / "upgrade_20260320_seq60",
        stock1m_dir=Path("/data/ashare/market/stock1m"),
        start_trade_date=20210101,
        end_trade_date=20231231,
        split_policy="date_ranges",
        train_end_trade_date=20221231,
        val_end_trade_date=20230228,
        train_days=6,
        val_days=1,
        test_days=1,
        rolling_step_days=1,
        seed=7,
        horizon_minutes=30,
        sample_stocks_per_minute=800,
        use_cross_sectional_gaussianize=False,
        data_prep_include_predict_split=False,
        data_prep_norm_fit_scope="train_only",
        data_prep_days_per_call=100,
        data_prep_workers=32,
        batch_size=8192,
        num_workers=4,
        num_iters=120001,
        save_every=10000,
        eval_every=20000,
        eval_during=False,
        eval_during_num_iters=0,
        eval_batch_size=8192,
        learning_rate=2e-3,
        hidden_dims=[512, 512],
        dropout=0.1,
        input_window_size=60,
        rolling_window=1000,
        rolling_step=10,
    )


def main() -> None:
    """Run the pipeline with module-level configuration constants."""
    # Reuse the shared default config so full and postprocess-only entrypoints stay aligned.
    cfg = _default_config()

    # Execute the full pipeline.
    run_pipeline(cfg)


def _run_pipeline_with_split_runner(cfg: PipelineConfig, split_runner) -> None:
    """Resolve split policy and dispatch one runner per resolved split."""
    # Redirect the configured root_dir into /data-cache for all intermediate artifacts.
    root_dir = _redirect_to_data_cache(Path(cfg.root_dir))

    # Disable minute sampling for sequence inputs so date probing and prep stay aligned.
    probe_sample_stocks_per_minute = 0 if int(cfg.input_window_size) > 1 else int(cfg.sample_stocks_per_minute)

    # List all available trade dates in the configured range.
    probe = DataPrepConfig(
        stock1m_dir=Path(cfg.stock1m_dir),
        out_dir=root_dir / "probe",
        start_trade_date=int(cfg.start_trade_date),
        end_trade_date=int(cfg.end_trade_date),
        train_days=int(cfg.train_days),
        val_days=int(cfg.val_days),
        test_days=int(cfg.test_days),
        seed=int(cfg.seed),
        horizon_minutes=int(cfg.horizon_minutes),
        sample_stocks_per_minute=int(probe_sample_stocks_per_minute),
        use_cross_sectional_gaussianize=bool(cfg.use_cross_sectional_gaussianize),
        include_predict_split=False,
        norm_fit_scope="train_only",
        days_per_call=100,
        workers=32,
    )
    all_dates = list_trade_dates(probe)
    if len(all_dates) == 0:
        raise RuntimeError("No trade dates found in the configured range.")

    # Run the requested split policy without silently truncating dates.
    policy = str(cfg.split_policy)
    if policy == "tail_holdout":
        # Use all leading dates for train, then hold out tail val/test windows.
        train_days = int(len(all_dates) - int(cfg.val_days) - int(cfg.test_days))
        if train_days < int(cfg.train_days):
            raise RuntimeError(f"tail_holdout requires train_days >= cfg.train_days, got={train_days} cfg={int(cfg.train_days)}")
        out_root = root_dir / "tail_holdout"
        split_runner(
            cfg,
            out_root=out_root,
            start_trade_date=int(cfg.start_trade_date),
            end_trade_date=int(cfg.end_trade_date),
            train_days=int(train_days),
            val_days=int(cfg.val_days),
            test_days=int(cfg.test_days),
        )
        return

    if policy == "date_ranges":
        # Validate configured split boundaries before deriving split sizes.
        train_end = int(cfg.train_end_trade_date)
        val_end = int(cfg.val_end_trade_date)
        if not (int(cfg.start_trade_date) <= int(train_end) < int(val_end) < int(cfg.end_trade_date)):
            raise RuntimeError("date_ranges requires start_trade_date <= train_end_trade_date < val_end_trade_date < end_trade_date")

        # Convert calendar boundaries into split counts on the available trade-date list.
        train_dates = [int(d) for d in list(all_dates) if int(cfg.start_trade_date) <= int(d) <= int(train_end)]
        val_dates = [int(d) for d in list(all_dates) if int(train_end) < int(d) <= int(val_end)]
        test_dates = [int(d) for d in list(all_dates) if int(val_end) < int(d) <= int(cfg.end_trade_date)]
        if len(train_dates) == 0 or len(val_dates) == 0 or len(test_dates) == 0:
            raise RuntimeError(f"date_ranges produced empty split: train={len(train_dates)} val={len(val_dates)} test={len(test_dates)}")

        # Run one split that exactly follows the requested calendar ranges.
        out_root = root_dir / "date_ranges"
        split_runner(
            cfg,
            out_root=out_root,
            start_trade_date=int(cfg.start_trade_date),
            end_trade_date=int(cfg.end_trade_date),
            train_days=int(len(train_dates)),
            val_days=int(len(val_dates)),
            test_days=int(len(test_dates)),
        )
        return

    if policy == "rolling_backtest":
        # Roll fixed-size windows across the full date list.
        total = int(cfg.train_days) + int(cfg.val_days) + int(cfg.test_days)
        step = int(cfg.rolling_step_days)
        if step <= 0:
            raise RuntimeError(f"rolling_step_days must be positive, got: {step}")
        out_base = root_dir / "rolling_backtest"
        fold = 0
        for st in range(0, int(len(all_dates) - total + 1), int(step)):
            # Resolve the date window and map it into a fold output directory.
            window = list(all_dates[st : st + total])
            start_date = int(window[0])
            end_date = int(window[-1])
            out_root = out_base / f"fold_{fold:03d}_{start_date}_{end_date}"
            split_runner(
                cfg,
                out_root=out_root,
                start_trade_date=int(start_date),
                end_trade_date=int(end_date),
                train_days=int(cfg.train_days),
                val_days=int(cfg.val_days),
                test_days=int(cfg.test_days),
            )
            fold += 1
        return

    raise RuntimeError(f"Unknown split_policy: {policy}")


def run_pipeline(cfg: PipelineConfig) -> None:
    """Execute the staged pipeline under /data-cache based on cfg.pipeline_mode."""
    # Dispatch split execution through the staged runner so predict work is optional.
    _run_pipeline_with_split_runner(cfg, _run_single_split_staged)


def run_pipeline_postprocess_only(cfg: PipelineConfig) -> None:
    """Backward-compatible alias for rebuilding the train report only."""
    # Reuse the staged runner with train_report_only semantics.
    from dataclasses import replace

    _run_pipeline_with_split_runner(replace(cfg, pipeline_mode="train_report_only"), _run_single_split_staged)


def run_train_report_postprocess_only(cfg: PipelineConfig) -> None:
    """Rebuild train_report.md from existing artifacts without rerunning data prep or training."""
    # Force pipeline_mode to the report-only mode and dispatch through the staged runner.
    from dataclasses import replace

    _run_pipeline_with_split_runner(replace(cfg, pipeline_mode="train_report_only"), _run_single_split_staged)


def run_predict_report_postprocess_only(cfg: PipelineConfig) -> None:
    """Rebuild predict_report.md from an existing predict manifest without rerunning training."""
    # Force pipeline_mode to the predict-report-only mode and dispatch through the staged runner.
    from dataclasses import replace

    _run_pipeline_with_split_runner(replace(cfg, pipeline_mode="predict_report_only"), _run_single_split_staged)


if __name__ == "__main__":
    main()
