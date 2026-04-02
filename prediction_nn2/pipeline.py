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
    compute_predict_report_from_manifest,
    intraday_time_series_ic_train_test_from_manifest,
    pearson_ic_time_series_summary_from_manifest,
    pooled_pearson_ic_from_manifest,
    rolling_group_ic_from_manifest,
)
from prediction_nn2.html_report import build_page, render_embedded_figure, render_figure, render_section, render_table, render_value_rows, render_yaml_block
from prediction_nn2.model import GruMlpConfig, GruMlpRegressor


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


def _effective_sample_stocks_per_minute(cfg: PipelineConfig) -> int:
    """Return the effective per-minute sampling count used by data prep."""
    # Disable minute sampling for sequence inputs so stock histories stay contiguous.
    if int(cfg.input_window_size) > 1:
        return 0
    return int(cfg.sample_stocks_per_minute)


def _lr_scheduler_contract(cfg: PipelineConfig) -> dict[str, object]:
    """Return the fixed LR-scheduler contract used by qmodel training."""
    # Keep scheduler settings in one place so config building and fingerprinting stay aligned.
    return {
        "start_warmup_factor": 0.001,
        "end_warmup_factor": 1.0,
        "warmup_iters": 200,
        "finish_decay_iter": int(cfg.num_iters),
        "eta_min": 1e-6,
    }


def _train_stage_contract(cfg: PipelineConfig, feature_dim: int) -> dict[str, object]:
    """Return the effective train-stage contract that must invalidate stale checkpoints."""
    # Record the fixed architecture and optimizer choices that are not fully encoded in PipelineConfig.
    return {
        "model_class": "GruMlpRegressor",
        "model_config": {
            "input_size": int(feature_dim),
            "hidden_size": 256,
            "num_layers": 2,
            "bidirectional": False,
            "rnn_dropout": 0.0,
            "mlp_hidden_dims": list(cfg.hidden_dims),
            "mlp_dropout": float(cfg.dropout),
        },
        "optimizer_class": "AdamW",
        "criterion": "MSELoss",
        "use_lr_sched": "custom",
        "lr_scheduler": _lr_scheduler_contract(cfg),
    }


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

    # Read events and require the smoother train loss scalar to exist.
    acc = EventAccumulator(tb_dir.as_posix(), size_guidance={"scalars": 0})
    acc.Reload()
    tag = "train/objective/loss_mean"
    scalars = acc.Scalars(tag)
    steps = np.asarray([s.step for s in scalars], dtype=int)
    values = np.asarray([s.value for s in scalars], dtype=float)

    # Drop the step=0 point to make the exported curve easier to read.
    m = steps != 0
    steps = steps[m]
    values = values[m]

    # Keep only the last point for each step so resumed writers do not create vertical spikes.
    keep = steps.size - 1 - np.unique(steps[::-1], return_index=True)[1]
    keep.sort()
    steps = steps[keep]
    values = values[keep]

    # Plot the deduplicated mean loss curve for report readability.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, values, linewidth=1.6, color="#4c72b0")
    ax.set_title("Training loss curve (TensorBoard scalar: train/objective/loss_mean)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
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


def _format_date_range(dates: list[int]) -> str:
    """Format one split date list into a compact inclusive range string."""
    # Return a stable empty marker when the split list is empty.
    if len(dates) == 0:
        return "[]"

    # Format the first and last trade date together with the split length.
    return f"{int(dates[0])} -> {int(dates[-1])} ({int(len(dates))} trade days)"


def _model_summary(cfg: PipelineConfig, feature_dim: int, train_rows: int) -> tuple[list[tuple[str, str]], GruMlpConfig]:
    """Build scalar rows describing the effective NN model and training setup."""
    # Materialize the exact model config used by qmodel training.
    model_cfg = GruMlpConfig(
        input_size=int(feature_dim),
        hidden_size=256,
        num_layers=2,
        bidirectional=False,
        rnn_dropout=0.0,
        mlp_hidden_dims=list(cfg.hidden_dims),
        mlp_dropout=float(cfg.dropout),
        dtype=torch.float32,
    )

    # Instantiate one model to count trainable parameters exactly.
    model = GruMlpRegressor(model_cfg)
    param_count = int(sum(int(p.numel()) for p in model.parameters()))

    # Convert row counts into approximate epoch counts for human review.
    batches_per_epoch = float(train_rows) / float(cfg.batch_size)
    total_epochs = float(cfg.num_iters) / float(batches_per_epoch)

    # Assemble the single-column model summary rows.
    rows = [
        ("model_class", "GruMlpRegressor"),
        ("input_tensor", f"(B, T={int(cfg.input_window_size)}, F={int(feature_dim)})"),
        ("prediction_target", f"{int(cfg.horizon_minutes)}-minute forward log return"),
        ("rnn_type", "GRU"),
        ("gru_hidden_size", str(int(model_cfg.hidden_size))),
        ("gru_num_layers", str(int(model_cfg.num_layers))),
        ("gru_bidirectional", str(bool(model_cfg.bidirectional))),
        ("gru_dropout", f"{float(model_cfg.rnn_dropout):.1f}"),
        ("representation", "last_hidden of final GRU layer"),
        ("mlp_hidden_dims", " -> ".join([str(int(v)) for v in list(cfg.hidden_dims)]) + " -> 1"),
        ("mlp_dropout", f"{float(model_cfg.mlp_dropout):.1f}"),
        ("trainable_parameters", f"{int(param_count):,}"),
        ("optimizer", "AdamW"),
        ("learning_rate", f"{float(cfg.learning_rate):.6g}"),
        ("batch_size", str(int(cfg.batch_size))),
        ("eval_batch_size", str(int(cfg.eval_batch_size))),
        ("num_iters", str(int(cfg.num_iters))),
        ("approx_total_epochs", f"{float(total_epochs):.2f}"),
        ("save_every", str(int(cfg.save_every))),
        ("eval_every", str(int(cfg.eval_every))),
        ("num_workers", str(int(cfg.num_workers))),
    ]
    return rows, model_cfg


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

    # Select AMP settings from the resolved device so CUDA runs use fp16 autocast.
    amp_dtype = torch.float16 if bool(use_cuda) else torch.float32
    use_amp = "torch" if bool(use_cuda) else "none"
    eval_dtype = torch.float16 if bool(use_cuda) else torch.float32

    # Define dataset callable with explicit spec object to avoid hidden globals.
    npz_dir = Path(run_root) / "artifacts" / "npz"
    dataset_spec = NpzDatasetSpec(data_dir=npz_dir, pin_memory=use_cuda, window_size=int(cfg.input_window_size))

    def dataset_class(group: str, dtype: torch.dtype) -> Stock1mNpzDataset:
        """Build a dataset instance for the requested split."""
        # Create the dataset with an explicit NPZ spec.
        return Stock1mNpzDataset(group, dtype, dataset_spec)

    # Build model config and core training components.
    model_cfg = GruMlpConfig(
        input_size=int(feature_dim),
        hidden_size=256,
        num_layers=2,
        bidirectional=False,
        rnn_dropout=0.0,
        mlp_hidden_dims=list(cfg.hidden_dims),
        mlp_dropout=float(cfg.dropout),
        dtype=torch.float32,
    )
    lr_scheduler_cfg = _lr_scheduler_contract(cfg)

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
        model_class=GruMlpRegressor,
        model=model_cfg,
        seed=int(cfg.seed),
        amp_dtype=amp_dtype,
        eval_dtype=eval_dtype,
        train_dtype=torch.float32,
        criterion=torch.nn.MSELoss(),
        optimizer_class=torch.optim.AdamW,
        learning_rate=float(cfg.learning_rate),
        use_amp=use_amp,
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
        lr_scheduler=LRSchedulerConfig(**lr_scheduler_cfg),
        profiler=SimpleNamespace(
            profile_section="none",
            profile_dir=str(Path(run_root) / "run" / "profile"),
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
        sample_stocks_per_minute=int(_effective_sample_stocks_per_minute(cfg)),
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


def _load_existing_train_eval_dir(qconf: SimpleNamespace, best_it: int) -> Path:
    """Load the existing train eval directory for the selected checkpoint."""
    # Resolve the expected train eval directory and require it to exist.
    train_eval_iter_dir = Path(qconf.root_dir) / "eval_train" / f"iter_{int(best_it)}"
    if not train_eval_iter_dir.exists():
        raise RuntimeError(f"Missing existing train eval dir: {train_eval_iter_dir}")
    return train_eval_iter_dir


def _load_existing_predict_manifest(qconf: SimpleNamespace, best_it: int) -> Path:
    """Load the existing predict manifest for the selected checkpoint."""
    # Resolve the expected predict manifest path and require it to exist.
    manifest_path = Path(qconf.root_dir) / "eval" / f"iter_{int(best_it)}" / "predict_manifest.yaml"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing existing predict manifest: {manifest_path}")
    return manifest_path


def _load_existing_eval_manifest(qconf: SimpleNamespace, eval_dir_name: str, best_it: int) -> Path:
    """Load an existing streamed eval manifest for one split."""
    # Resolve the expected streamed manifest path and require it to exist.
    manifest_path = Path(qconf.root_dir) / str(eval_dir_name) / f"iter_{int(best_it)}" / "predict_manifest.yaml"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing existing eval manifest: {manifest_path}")
    return manifest_path


def _run_eval_manifest_once(qconf: SimpleNamespace, group: str, best_it: int, dst_name: str) -> Path:
    """Run one streamed evaluator pass and move it into a stable split-specific directory."""
    # Remove any stale temporary eval directory left by an interrupted prior run.
    import shutil

    run_dir = Path(qconf.root_dir)
    tmp_iter_dir = run_dir / "eval" / f"iter_{int(best_it)}"
    if tmp_iter_dir.exists():
        shutil.rmtree(tmp_iter_dir)

    # Run the evaluator in chunked-manifest mode for the requested split.
    if torch.device(qconf.device).type == "cuda":
        from qmodel.core.evaluator import Evaluator

        evaluator = Evaluator(qconf, group=str(group), writer=None, enable_logging=False)
    else:
        from qmodel.core.cpu_evaluator import CpuEvaluator

        evaluator = CpuEvaluator(qconf, group=str(group), writer=None, enable_logging=False)

    # Stream predictions into parquet chunks and close evaluator resources.
    evaluator._run_predict_inference_to_manifest(it=int(best_it), n_iter=0, iter_dir=tmp_iter_dir)
    evaluator.close()

    # Move the finished streamed eval into its stable destination and return the manifest path.
    dst_iter_dir = _move_eval_dir(run_dir, src_name="eval", dst_name=str(dst_name), it=int(best_it))
    manifest_path = Path(dst_iter_dir) / "predict_manifest.yaml"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing streamed eval manifest after move: {manifest_path}")
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

    # Reuse an existing test manifest when it is already on disk.
    test_manifest_path: Path | None = None
    if bool(require_existing_eval):
        candidate = Path(qconf.root_dir) / "eval_test" / f"iter_{int(best_it)}" / "predict_manifest.yaml"
        if candidate.exists():
            test_manifest_path = candidate

    # Rebuild the test manifest when it is missing.
    if test_manifest_path is None:
        test_manifest_path = _run_eval_manifest_once(qconf, "test", int(best_it), "eval_test")

    # Reuse an existing train manifest when it is already on disk.
    train_manifest_path: Path | None = None
    if bool(require_existing_eval):
        candidate = Path(qconf.root_dir) / "eval_train" / f"iter_{int(best_it)}" / "predict_manifest.yaml"
        if candidate.exists():
            train_manifest_path = candidate

    # Rebuild the train manifest when it is missing.
    if train_manifest_path is None:
        train_manifest_path = _run_eval_manifest_once(qconf, "train", int(best_it), "eval_train")

    # Compute train/test pooled Pearson IC and time-series summaries from manifests.
    test_pooled = pooled_pearson_ic_from_manifest(Path(test_manifest_path))
    test_ic_summary_yaml = Path(out_root) / "ic_time_series_summary_test.yaml"
    t_summary0 = time.time()
    test_ic_summary = pearson_ic_time_series_summary_from_manifest(Path(test_manifest_path), test_ic_summary_yaml)

    train_pooled = pooled_pearson_ic_from_manifest(Path(train_manifest_path))
    train_ic_summary_yaml = Path(out_root) / "ic_time_series_summary_train.yaml"
    train_ic_summary = pearson_ic_time_series_summary_from_manifest(Path(train_manifest_path), train_ic_summary_yaml)
    t_summary1 = time.time()

    # Emit train/test intraday Pearson IC diagnostics from streamed manifests.
    intraday_csv = Path(out_root) / "intraday_ic.csv"
    intraday_png = Path(out_root) / "intraday_ic.png"
    intraday_time_series_ic_train_test_from_manifest(Path(train_manifest_path), Path(test_manifest_path), intraday_csv, intraday_png)
    eval_cfg = EvalConfig(
        stock1m_dir=Path(cfg.stock1m_dir),
        window_size=int(cfg.rolling_window),
        step_size=int(cfg.rolling_step),
        horizon_minutes=int(cfg.horizon_minutes),
    )
    vol_csv = Path(out_root) / "vol_rolling_ic.csv"
    vol_png = Path(out_root) / "vol_rolling_ic.png"
    vol_yaml = Path(out_root) / "vol_rolling_ic.yaml"
    price_csv = Path(out_root) / "price_rolling_ic.csv"
    price_png = Path(out_root) / "price_rolling_ic.png"
    price_yaml = Path(out_root) / "price_rolling_ic.yaml"

    # Compute the test rolling IC curves from the streamed manifest without a giant test dataframe.
    t_roll0 = time.time()
    vol_agg = rolling_group_ic_from_manifest(Path(test_manifest_path), eval_cfg, "volatility_label", vol_csv, vol_png, vol_yaml)
    price_agg = rolling_group_ic_from_manifest(Path(test_manifest_path), eval_cfg, "price_label", price_csv, price_png, price_yaml)
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
            "summary_stream_seconds": float(t_summary1 - t_summary0),
            "rolling_ic_seconds": float(t_roll1 - t_roll0),
        },
    }
    import yaml

    (Path(out_root) / "perf_audit.yaml").write_text(yaml.safe_dump(perf, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Render and persist the self-contained HTML train report.
    report_path = Path(out_root) / "train_report.html"
    report = _render_train_report_html(
        cfg,
        prep,
        train_pooled,
        test_pooled,
        train_ic_summary,
        test_ic_summary,
        train_ic_summary_yaml,
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


def _render_train_report_html(
    cfg: PipelineConfig,
    prep: dict[str, object],
    train_pooled: dict[str, float],
    test_pooled: dict[str, float],
    train_ic_summary: dict[str, object],
    test_ic_summary: dict[str, object],
    train_ic_summary_yaml: Path,
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
    """Render the train-report self-contained HTML content."""
    # Load all persisted split metadata and data-clean artifacts needed by the report.
    import yaml

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_path = Path(prep["meta_path"])
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    stats_dir = meta_path.parent.parent / "data_clean"
    invalid_stats = pd.read_csv(stats_dir / "invalid_feature_stats.csv")
    invalid_stats = invalid_stats.sort_values(["invalid_ratio", "invalid_count"], ascending=False, kind="stable").reset_index(drop=True)
    moment_tbl = pd.read_csv(stats_dir / "feature_moments.csv")

    # Derive the main checkpoint-selection metrics shown near the top of the report.
    best_metrics = dict(val_metrics_by_it[int(best_it)])
    val_mse = float(best_metrics["val/objective/mse"])
    val_ic = float(best_metrics["val/quality/global_ic"])
    val_rank_ic = float(best_metrics["val/quality/rank_ic"])
    approx_epoch = _approx_epoch(int(best_it), int(prep["train_rows"]), int(cfg.batch_size))
    model_rows, _model_cfg = _model_summary(cfg, len(prep["feature_names"]), int(prep["train_rows"]))

    # Summarize rolling-curve shape so the report has scalar takeaways next to the plots.
    vol_desc = _describe_group_inflection(
        vol_agg["mean_ic"].to_numpy(dtype=float),
        vol_agg["group_center_rank"].to_numpy(dtype=float),
    )
    price_desc = _describe_group_inflection(
        price_agg["mean_ic"].to_numpy(dtype=float),
        price_agg["group_center_rank"].to_numpy(dtype=float),
    )

    # Build the validation sweep table used to document checkpoint selection.
    def _mse_row(it: int) -> tuple[int, float, float, float]:
        """Convert one validation metric dict into a compact table row."""
        # Read the three core validation metrics for one checkpoint.
        metrics = dict(val_metrics_by_it[int(it)])
        return (
            int(it),
            float(metrics["val/objective/mse"]),
            float(metrics["val/quality/global_ic"]),
            float(metrics["val/quality/rank_ic"]),
        )

    sweep = sorted([_mse_row(it) for it in list(val_metrics_by_it.keys())], key=lambda r: float(r[1]))[:10]
    sweep_table = render_table(
        ["iter", "val_mse", "val_ic", "val_rank_ic"],
        [[str(int(it)), f"{float(mse):.6e}", f"{float(ic):.6f}", f"{float(ric):.6f}"] for it, mse, ic, ric in list(sweep)],
    )

    # Build the stacked summary sections that replace the old markdown body.
    dates = dict(meta["dates"])
    groups = dict(meta["storage"]["groups"])
    overview_rows = [
        ("generated_at", now),
        ("out_root", meta_path.parent.parent.parent.as_posix()),
        ("meta_yaml", meta_path.as_posix()),
        ("data_clean_report", (stats_dir / "report.html").as_posix()),
        ("train_rows", str(int(groups["train"]["rows"]))),
        ("val_rows", str(int(groups["val"]["rows"]))),
        ("test_rows", str(int(groups["test"]["rows"]))),
        ("train_range", _format_date_range(list(dates["train"]))),
        ("val_range", _format_date_range(list(dates["val"]))),
        ("test_range", _format_date_range(list(dates["test"]))),
    ]
    clean_rows = [
        ("stock_norm", f"{str(meta['feature_transform']['stock_norm']['type'])} / scope={str(meta['feature_transform']['stock_norm'].get('scope', 'n/a'))}"),
        ("feature_dim", str(int(len(prep["feature_names"])))),
        ("invalid_report_html", (stats_dir / "invalid_feature_report.html").as_posix()),
        ("feature_moments_csv", (stats_dir / "feature_moments.csv").as_posix()),
        ("pooled_distribution_png", (stats_dir / "pooled_feature_grid.png").as_posix()),
        ("train_kept_rate", f"{float(prep['audit_rates']['train']['kept_rate']):.2%}"),
        ("val_kept_rate", f"{float(prep['audit_rates']['val']['kept_rate']):.2%}"),
        ("test_kept_rate", f"{float(prep['audit_rates']['test']['kept_rate']):.2%}"),
        ("max_invalid_ratio", f"{float(invalid_stats.iloc[0]['invalid_ratio']):.4%} ({str(invalid_stats.iloc[0]['field'])})"),
        ("max_abs_mean", f"{float(moment_tbl['mean'].abs().max()):.6f}"),
        ("max_abs_std_shift", f"{float((moment_tbl['std'] - 1.0).abs().max()):.6f}"),
        ("max_abs_skew", f"{float(moment_tbl['skew'].abs().max()):.6f}"),
        ("max_abs_kurtosis", f"{float(moment_tbl['kurtosis'].abs().max()):.6f}"),
    ]
    invalid_table = render_table(
        ["field", "type", "invalid_ratio", "invalid_count", "total_count"],
        [
            [
                str(row["field"]),
                str(row["field_type"]),
                f"{float(row['invalid_ratio']):.4%}",
                str(int(row["invalid_count"])),
                str(int(row["total_count"])),
            ]
            for row in invalid_stats.head(10).to_dict(orient="records")
        ],
    )
    checkpoint_rows = [
        ("best_checkpoint_iter", str(int(best_it))),
        ("best_checkpoint_epoch", f"{float(approx_epoch):.2f}"),
        ("best_val_mse", f"{float(val_mse):.6e}"),
        ("best_val_ic", f"{float(val_ic):.6f}"),
        ("best_val_rank_ic", f"{float(val_rank_ic):.6f}"),
        ("train_loss_png", loss_png.as_posix()),
    ]
    ic_rows = [
        ("pooled_ic_train", f"{float(train_pooled['pearson_ic']):.6f}"),
        ("pooled_ic_test", f"{float(test_pooled['pearson_ic']):.6f}"),
        ("pooled_count_train", str(int(train_pooled["count"]))),
        ("pooled_count_test", str(int(test_pooled["count"]))),
        ("train_t_stat", f"{float(train_ic_summary['pearson_ic']['t_stat']):.4f}"),
        ("test_t_stat", f"{float(test_ic_summary['pearson_ic']['t_stat']):.4f}"),
        ("train_positive_ratio", f"{float(train_ic_summary['pearson_ic']['positive_ratio']):.2%}"),
        ("test_positive_ratio", f"{float(test_ic_summary['pearson_ic']['positive_ratio']):.2%}"),
        ("train_timestamps", str(int(train_ic_summary["timestamp_count"]))),
        ("test_timestamps", str(int(test_ic_summary["timestamp_count"]))),
    ]
    rolling_rows = [
        ("intraday_csv", intraday_csv.as_posix()),
        ("volatility_csv", vol_csv.as_posix()),
        ("price_csv", price_csv.as_posix()),
        ("volatility_curve", str(vol_desc)),
        ("price_curve", str(price_desc)),
    ]
    perf_rows = [
        ("perf_audit_yaml", (Path(meta_path).parent.parent.parent / "perf_audit.yaml").as_posix()),
        ("data_prep_seconds", f"{float(perf['data_prep']['elapsed_seconds']):.4f}"),
        ("summary_stream_seconds", f"{float(perf['eval']['summary_stream_seconds']):.4f}"),
        ("rolling_ic_seconds", f"{float(perf['eval']['rolling_ic_seconds']):.4f}"),
        ("device", str(perf["train"]["device"])),
        ("num_workers", str(int(perf["train"]["num_workers"]))),
    ]

    # Assemble the final self-contained single-column HTML report.
    sections = [
        render_section("Run Overview", render_value_rows(overview_rows)),
        render_section("NN Model", render_value_rows(model_rows)),
        render_section("Data Clean Summary", render_value_rows(clean_rows) + invalid_table),
        render_figure("Data Clean Distribution Overview", stats_dir / "pooled_feature_grid.png", "Pooled standardized feature distributions from the data clean stage."),
        render_section("Validation And Checkpoint", render_value_rows(checkpoint_rows) + sweep_table),
        render_figure("Train Loss Curve", loss_png, "Training loss exported from TensorBoard `train/objective/loss_mean`, with duplicated resume steps deduplicated and step 0 removed."),
        render_section(
            "Train Vs Test IC Summary",
            render_value_rows(ic_rows)
            + render_value_rows([("train_summary_yaml", train_ic_summary_yaml.as_posix()), ("test_summary_yaml", test_ic_summary_yaml.as_posix())])
            + render_yaml_block({"train": train_ic_summary, "test": test_ic_summary}),
        ),
        render_section(
            "Intraday IC",
            render_value_rows([("intraday_ic_csv", intraday_csv.as_posix())]) + render_embedded_figure("Intraday IC", intraday_png, "Train and test Pearson IC are plotted on the same intraday axis."),
        ),
        render_section(
            "Volatility Rolling IC",
            render_value_rows([("volatility_ic_csv", vol_csv.as_posix()), ("curve_summary", str(vol_desc))])
            + render_embedded_figure("Volatility Rolling IC", vol_png, "Rolling Pearson IC on test data grouped by volatility label."),
        ),
        render_section(
            "Price Rolling IC",
            render_value_rows([("price_ic_csv", price_csv.as_posix()), ("curve_summary", str(price_desc))])
            + render_embedded_figure("Price Rolling IC", price_png, "Rolling Pearson IC on test data grouped by price label."),
        ),
        render_section("Performance Audit", render_value_rows(perf_rows) + render_yaml_block(perf)),
    ]
    return build_page("NN Train Report", "Self-contained HTML report with vertically stacked sections.", sections)

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
    effective_sample = int(_effective_sample_stocks_per_minute(cfg))
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
        "sample_stocks_per_minute": int(effective_sample),
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
    report_path = Path(out_root) / "artifacts" / "data_clean" / "report.html"
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
        "train_contract": _train_stage_contract(cfg, int(feature_dim)),
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

        # Release trainer state between chunks to keep CUDA memory pressure stable.
        del trainer
        if torch.device(qconf.device).type == "cuda":
            torch.cuda.empty_cache()

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
    """Build the train report and return its HTML path."""
    # Build a stage fingerprint so report rebuild can skip when inputs are unchanged.
    stage_cfg = {"stage": "train_report", "best_it": int(best_it)}
    manifest_path = _stage_manifest_path(Path(out_root), "train_report")
    fp = _stage_fingerprint(stage_cfg)
    report_path = Path(out_root) / "train_report.html"

    # Skip the stage when the manifest and report already match.
    if manifest_path.exists() and report_path.exists():
        m = _load_stage_manifest(manifest_path)
        if str(m.get("fingerprint")) == str(fp):
            return report_path

    # Run the postprocess that generates test diagnostics and writes train_report.html.
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

    # Render one self-contained HTML wrapper around the produced artifacts.
    html_path = Path(out_root) / "predict_report.html"
    html = _render_predict_report_html(cfg, manifest_path, artifacts, report_dir)
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _render_predict_report_html(cfg: PipelineConfig, manifest_path: Path, artifacts, report_dir: Path) -> str:
    """Render the predict-report self-contained HTML content."""
    # Summarize the heavy predict diagnostics into stacked scalar sections.
    import yaml

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    pooled = dict(artifacts.pooled)
    ic_summary = dict(artifacts.ic_summary)
    turnover_summary = dict(artifacts.turnover_summary)
    residual_summary = dict(artifacts.residual_summary)
    annual_tbl = artifacts.annual_tbl
    annual_years = annual_tbl["year"].astype(int).tolist()
    year_range = f"{int(annual_years[0])}-{int(annual_years[-1])}" if int(annual_years[0]) != int(annual_years[-1]) else f"{int(annual_years[0])}"
    meta_path = Path(report_dir).parent / "artifacts" / "npz" / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    model_rows, _model_cfg = _model_summary(cfg, feature_dim=len(meta["feature_names"]), train_rows=int(meta["storage"]["groups"]["train"]["rows"]))

    # Build the table used for annual pooled IC review.
    annual_table = render_table(
        ["year", "pearson_ic", "rank_ic", "count"],
        [
            [
                str(int(row["year"])),
                f"{float(row['pearson_ic']):.6f}",
                f"{float(row['rank_ic']):.6f}" if np.isfinite(float(row["rank_ic"])) else "nan",
                str(int(row["count"])),
            ]
            for row in annual_tbl.to_dict(orient="records")
        ],
    )

    # Assemble the final self-contained single-column HTML report.
    sections = [
        render_section(
            "Predict Overview",
            render_value_rows(
                [
                    ("generated_at", now),
                    ("predict_manifest", Path(manifest_path).as_posix()),
                    ("report_dir", Path(report_dir).as_posix()),
                    ("pooled_pearson_ic", f"{float(pooled['pearson_ic']):.6f}"),
                    ("pooled_rank_ic", f"{float(pooled['rank_ic']):.6f}"),
                    ("count", str(int(pooled["count"]))),
                ]
            ),
        ),
        render_section("NN Model", render_value_rows(model_rows)),
        render_section(
            "IC Time Series Summary",
            render_value_rows(
                [
                    ("ic_summary_yaml", Path(artifacts.ic_summary_yaml).as_posix()),
                    ("pearson_t_stat", f"{float(ic_summary['pearson_ic']['t_stat']):.4f}"),
                    ("rank_t_stat", f"{float(ic_summary['rank_ic']['t_stat']):.4f}"),
                    ("timestamp_count", str(int(ic_summary["timestamp_count"]))),
                ]
            )
            + render_yaml_block(ic_summary),
        ),
        render_section(
            f"Annual IC ({year_range})",
            render_value_rows([("annual_csv", Path(artifacts.annual_csv).as_posix())])
            + annual_table
            + render_embedded_figure("Annual IC", Path(artifacts.annual_png), "Annual pooled IC curve from the predict manifest."),
        ),
        render_section(
            "Intraday IC",
            render_value_rows([("intraday_csv", Path(artifacts.intraday_csv).as_posix())])
            + render_embedded_figure("Predict Intraday IC", Path(artifacts.intraday_png), "Predict-side intraday IC curve."),
        ),
        render_section(
            "Volatility Rolling IC",
            render_value_rows([("volatility_csv", Path(artifacts.vol_csv).as_posix()), ("volatility_yaml", Path(artifacts.vol_yaml).as_posix())])
            + render_embedded_figure("Predict Volatility Rolling IC", Path(artifacts.vol_png), "Rolling IC grouped by volatility label.")
            + render_yaml_block(yaml.safe_load(Path(artifacts.vol_yaml).read_text(encoding="utf-8"))),
        ),
        render_section(
            "Price Rolling IC",
            render_value_rows([("price_csv", Path(artifacts.price_csv).as_posix()), ("price_yaml", Path(artifacts.price_yaml).as_posix())])
            + render_embedded_figure("Predict Price Rolling IC", Path(artifacts.price_png), "Rolling IC grouped by price label.")
            + render_yaml_block(yaml.safe_load(Path(artifacts.price_yaml).read_text(encoding="utf-8"))),
        ),
        render_section(
            "Rank Diagnostics",
            render_value_rows([("rank_png", Path(artifacts.rank_png).as_posix())])
            + render_embedded_figure("Predict Rank Diagnostics", Path(artifacts.rank_png), "Prediction rank versus target rank diagnostics."),
        ),
        render_section(
            "Turnover",
            render_value_rows([("turnover_csv", Path(artifacts.turnover_csv).as_posix()), ("turnover_yaml", Path(artifacts.turnover_yaml).as_posix())])
            + render_embedded_figure("Predict Turnover", Path(artifacts.turnover_png), "Adjacent-timestamp prediction rank turnover.")
            + render_yaml_block(turnover_summary),
        ),
        render_section(
            "Residual Diagnostics",
            render_value_rows([("residual_yaml", Path(artifacts.residual_yaml).as_posix())])
            + render_embedded_figure("Predict Residual Diagnostics", Path(artifacts.residual_png), "Residual histogram and scatter diagnostics.")
            + render_yaml_block(residual_summary),
        ),
    ]
    return build_page("Predict Report", "Self-contained HTML report generated from the predict manifest.", sections)


def run_predict_report_stage(cfg: PipelineConfig, *, out_root: Path, predict_manifest_path: Path) -> Path:
    """Build the predict report from an existing predict manifest and return its path."""
    # Build a stage fingerprint so report rebuild can skip when manifest stays the same.
    stage_cfg = {"stage": "predict_report", "manifest": str(Path(predict_manifest_path).as_posix())}
    manifest_path = _stage_manifest_path(Path(out_root), "predict_report")
    fp = _stage_fingerprint(stage_cfg)
    report_html = Path(out_root) / "predict_report.html"

    # Skip the stage when the manifest and report already match.
    if manifest_path.exists() and report_html.exists():
        m = _load_stage_manifest(manifest_path)
        if str(m.get("fingerprint")) == str(fp):
            return report_html

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
    """Render the final combined report content."""
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
    data_clean_report_rel = Path("artifacts") / "data_clean" / "report.html"
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

    # Render a structured report body matching the required sections.
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
        root_dir=Path("outputs") / "upgrade_20260328_gru_seq60_h10",
        stock1m_dir=Path("/data/ashare/market/stock1m"),
        start_trade_date=20210104,
        end_trade_date=20241231,
        split_policy="date_ranges",
        train_end_trade_date=20231229,
        val_end_trade_date=20240229,
        train_days=6,
        val_days=1,
        test_days=1,
        rolling_step_days=1,
        seed=7,
        horizon_minutes=10,
        sample_stocks_per_minute=800,
        use_cross_sectional_gaussianize=False,
        data_prep_include_predict_split=False,
        data_prep_norm_fit_scope="train_only",
        data_prep_days_per_call=100,
        data_prep_workers=32,
        batch_size=4096,
        num_workers=4,
        num_iters=80000,
        save_every=5000,
        eval_every=10000,
        eval_during=False,
        eval_during_num_iters=0,
        eval_batch_size=4096,
        learning_rate=1e-3,
        hidden_dims=[512, 512],
        dropout=0.0,
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
        sample_stocks_per_minute=int(_effective_sample_stocks_per_minute(cfg)),
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
    """Rebuild train_report.html from existing artifacts without rerunning data prep or training."""
    # Force pipeline_mode to the report-only mode and dispatch through the staged runner.
    from dataclasses import replace

    _run_pipeline_with_split_runner(replace(cfg, pipeline_mode="train_report_only"), _run_single_split_staged)


def run_predict_report_postprocess_only(cfg: PipelineConfig) -> None:
    """Rebuild predict_report.html from an existing predict manifest without rerunning training."""
    # Force pipeline_mode to the predict-report-only mode and dispatch through the staged runner.
    from dataclasses import replace

    _run_pipeline_with_split_runner(replace(cfg, pipeline_mode="predict_report_only"), _run_single_split_staged)


if __name__ == "__main__":
    main()
