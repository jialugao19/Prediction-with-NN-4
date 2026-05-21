"""Run the end-to-end pipeline: data prep, training, evaluation, and report generation."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

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
    compute_test_evaluation_report_from_manifest,
    daily_pearson_ic_summary_from_manifest,
    intraday_time_series_ic_train_test_from_manifest,
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
    label_entry_lag_minutes: int
    sample_stocks_per_minute: int
    use_cross_sectional_gaussianize: bool
    data_prep_use_ret_signed_log1p: bool
    data_prep_use_heavy_tanh: bool
    data_prep_heavy_tanh_c: float
    data_prep_add_is_zero_features: bool
    data_prep_include_session_id: bool
    data_prep_include_inference_splits: bool
    data_prep_norm_fit_scope: str
    data_prep_label_norm: str
    data_prep_label_norm_fit_scope: str
    data_prep_days_per_call: int
    data_prep_workers: int
    batch_size: int
    num_workers: int
    dataloader_pin_memory: bool
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: bool
    num_iters: int
    save_every: int
    eval_every: int
    eval_during: bool
    eval_during_num_iters: int
    eval_batch_size: int
    train_profile_section: str
    train_profile_wait: int
    train_profile_warmup: int
    train_profile_active: int
    train_profile_repeat: int
    learning_rate: float
    hidden_dims: list[int]
    dropout: float
    input_window_size: int
    rolling_window: int
    rolling_step: int


def _resolve_data_cache_nn_root_dir(configured_root_dir: Path, pipeline_mode: str) -> Path:
    """Resolve one run root directory under /data-cache/nn using a date tag."""
    # Use Asia/Shanghai explicitly so directory naming is stable across machines.
    now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))

    # Treat /data-cache/nn as a parent hint and allocate a dated child directory.
    nn_parent = Path("/data-cache/nn")
    cfg = Path(configured_root_dir)
    if cfg.resolve() == nn_parent.resolve():
        base_name = now.strftime("%m%d")
        if str(pipeline_mode) in {"train_report_only", "test_report_only"}:
            return _select_latest_date_dir(nn_parent, base_name)
        return _allocate_unique_date_dir(nn_parent, base_name)

    # Allow users to explicitly pass a directory under /data-cache/nn.
    if cfg.is_absolute() and nn_parent in cfg.parents:
        return cfg

    # Default behavior: ignore legacy outputs/ paths and place results under /data-cache/nn/<MMDD>.
    base_name = now.strftime("%m%d")
    return _allocate_unique_date_dir(nn_parent, base_name)


def _allocate_unique_date_dir(parent_dir: Path, base_name: str) -> Path:
    """Allocate a unique directory name like 0416, 0416-02, 0416-03 under parent_dir."""
    # Ensure the parent directory exists so we can probe for collisions.
    parent = Path(parent_dir)
    parent.mkdir(parents=True, exist_ok=True)

    # Use the base date name if free; otherwise append -02/-03 style suffixes.
    candidate = parent / str(base_name)
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    # Scan suffixes deterministically so repeated same-day runs do not overwrite.
    idx = 2
    while True:
        candidate = parent / f"{str(base_name)}-{idx:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        idx += 1


def _select_latest_date_dir(parent_dir: Path, base_name: str) -> Path:
    """Select the latest existing date directory like 0416-03 under parent_dir."""
    # Ensure the parent directory exists so scans behave predictably.
    parent = Path(parent_dir)
    parent.mkdir(parents=True, exist_ok=True)

    # Collect all existing directories that match the base date name and suffix policy.
    candidates: list[tuple[int, Path]] = []
    plain = parent / str(base_name)
    if plain.is_dir():
        candidates.append((1, plain))
    for p in parent.glob(f"{str(base_name)}-[0-9][0-9]"):
        if not p.is_dir():
            continue
        suffix = p.name.split("-")[-1]
        candidates.append((int(suffix), p))

    # Require at least one candidate so report-only does not silently allocate an empty dir.
    if len(candidates) == 0:
        raise RuntimeError(f"No existing output directory found under {parent} for base_name={base_name}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _effective_sample_stocks_per_minute(cfg: PipelineConfig) -> int:
    """Return the effective per-minute sampling count used by data prep."""
    # Disable minute sampling for sequence inputs so stock histories stay contiguous.
    if int(cfg.input_window_size) > 1:
        return 0
    return int(cfg.sample_stocks_per_minute)


def _resolved_num_iters(cfg: PipelineConfig, train_rows: int) -> int:
    """Resolve an effective num_iters that runs for more than one epoch."""
    # Convert train rows into the minimum final iteration that exceeds one epoch.
    batches_per_epoch = int(np.ceil(float(train_rows) / float(cfg.batch_size)))
    min_final_iter = int(batches_per_epoch)

    # Respect the configured lower bound before aligning to checkpoint cadence.
    configured_final_iter = int(max(int(cfg.num_iters) - 1, int(min_final_iter)))

    # Align the final iteration to save_every so the last checkpoint is materialized.
    save_every = int(cfg.save_every)
    resolved_final_iter = int(((configured_final_iter + save_every - 1) // save_every) * save_every)
    return int(resolved_final_iter + 1)


def _lr_scheduler_contract(cfg: PipelineConfig, train_rows: int) -> dict[str, object]:
    """Return the fixed LR-scheduler contract used by qmodel training."""
    # Resolve the effective training length before wiring the decay schedule.
    effective_num_iters = int(_resolved_num_iters(cfg, int(train_rows)))

    # Keep scheduler settings in one place so config building and fingerprinting stay aligned.
    return {
        "start_warmup_factor": 0.001,
        "end_warmup_factor": 1.0,
        "warmup_iters": 200,
        "finish_decay_iter": int(effective_num_iters),
        "eta_min": 1e-6,
    }


def _train_stage_contract(cfg: PipelineConfig, feature_dim: int, train_rows: int) -> dict[str, object]:
    """Return the effective train-stage contract that must invalidate stale checkpoints."""
    # Resolve the effective iteration count before recording scheduler choices.
    effective_num_iters = int(_resolved_num_iters(cfg, int(train_rows)))

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
        "num_iters": int(effective_num_iters),
        "lr_scheduler": _lr_scheduler_contract(cfg, int(train_rows)),
        "dataloader": {
            "batch_size": int(cfg.batch_size),
            "num_workers": int(cfg.num_workers),
            "pin_memory": bool(cfg.dataloader_pin_memory),
            "prefetch_factor": int(cfg.dataloader_prefetch_factor),
            "persistent_workers": bool(cfg.dataloader_persistent_workers),
        },
        "profiler": {
            "profile_section": str(cfg.train_profile_section),
            "wait": int(cfg.train_profile_wait),
            "warmup": int(cfg.train_profile_warmup),
            "active": int(cfg.train_profile_active),
            "repeat": int(cfg.train_profile_repeat),
        },
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

    # Compute the 1000-step sliding-window mean for the requested smoother overlay.
    smooth_window = 1000
    smooth_values = pd.Series(values).rolling(window=int(smooth_window), min_periods=int(smooth_window)).mean().to_numpy()

    # Plot the raw deduplicated loss curve and the smoother overlay for report readability.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, values, linewidth=1.0, color="#4c72b0", alpha=0.35, label="raw")
    ax.plot(steps, smooth_values, linewidth=1.8, color="#dd8452", label="sliding mean (1000)")
    ax.set_title("Training loss curve (TensorBoard scalar: train/objective/loss_mean)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(loc="best")
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
    # Resolve the effective training length before computing epoch counts.
    effective_num_iters = int(_resolved_num_iters(cfg, int(train_rows)))

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
    total_epochs = float(effective_num_iters) / float(batches_per_epoch)

    # Assemble the single-column model summary rows.
    label_definition = f"log_close[t+{int(cfg.horizon_minutes)}] - log_close[t+{int(cfg.label_entry_lag_minutes)}]"
    rows = [
        ("model_class", "GruMlpRegressor"),
        ("input_tensor", f"(B, T={int(cfg.input_window_size)}, F={int(feature_dim)})"),
        ("prediction_target", label_definition),
        ("label_horizon_minutes", str(int(cfg.horizon_minutes))),
        ("label_entry_lag_minutes", str(int(cfg.label_entry_lag_minutes))),
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
        ("num_iters", str(int(effective_num_iters))),
        ("approx_total_epochs", f"{float(total_epochs):.2f}"),
        ("save_every", str(int(cfg.save_every))),
        ("eval_every", str(int(cfg.eval_every))),
        ("num_workers", str(int(cfg.num_workers))),
        ("dataloader_pin_memory", str(bool(cfg.dataloader_pin_memory))),
        ("dataloader_prefetch_factor", str(int(cfg.dataloader_prefetch_factor))),
        ("dataloader_persistent_workers", str(bool(cfg.dataloader_persistent_workers))),
        ("train_profile_section", str(cfg.train_profile_section)),
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


def _load_data_contract_from_meta(meta_path: Path) -> dict[str, object]:
    """Load the preprocessing contract that should invalidate downstream stages."""
    # Parse meta.yaml and keep only fields that change training/evaluation semantics.
    import yaml

    meta = yaml.safe_load(Path(meta_path).read_text(encoding="utf-8"))
    return {
        "prep_config": dict(meta["prep_config"]),
        "feature_transform": dict(meta["feature_transform"]),
        "label": dict(meta["label"]),
        "label_transform": dict(meta.get("label_transform", {"type": "none"})),
        "sampling": dict(meta["sampling"]),
        "dates": dict(meta["dates"]),
    }


def _load_train_rows_from_meta(meta_path: Path) -> int:
    """Load the train row count from meta.yaml."""
    # Parse meta.yaml and read the stored train row count directly.
    import yaml

    meta = yaml.safe_load(Path(meta_path).read_text(encoding="utf-8"))
    return int(meta["storage"]["groups"]["train"]["rows"])


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


def _build_qmodel_config(cfg: PipelineConfig, feature_dim: int, run_root: Path, train_rows: int) -> SimpleNamespace:
    """Build a qmodel-compatible flat config namespace for single-GPU training."""
    # Resolve the effective training length before building trainer config.
    effective_num_iters = int(_resolved_num_iters(cfg, int(train_rows)))

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
    lr_scheduler_cfg = _lr_scheduler_contract(cfg, int(train_rows))

    # Build evaluator config namespace to match qmodel evaluator expectations.
    evaluator = SimpleNamespace(
        eval_checkpoint_iter=[int(effective_num_iters) - 1],
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
        dataloader_pin_memory=bool(cfg.dataloader_pin_memory),
        dataloader_prefetch_factor=int(cfg.dataloader_prefetch_factor),
        dataloader_persistent_workers=bool(cfg.dataloader_persistent_workers),
        num_iters=int(effective_num_iters),
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
            profile_section=str(cfg.train_profile_section),
            profile_dir=str(Path(run_root) / "run" / "profile"),
            all_ranks=False,
            wait=int(cfg.train_profile_wait),
            warmup=int(cfg.train_profile_warmup),
            active=int(cfg.train_profile_active),
            repeat=int(cfg.train_profile_repeat),
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
    checkpoint_iters = sorted(set(int(it) for it in list(checkpoint_iters)))[-5:]
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
        label_entry_lag_minutes=int(cfg.label_entry_lag_minutes),
        sample_stocks_per_minute=int(_effective_sample_stocks_per_minute(cfg)),
        use_cross_sectional_gaussianize=bool(cfg.use_cross_sectional_gaussianize),
        use_ret_signed_log1p=bool(cfg.data_prep_use_ret_signed_log1p),
        use_heavy_tanh=bool(cfg.data_prep_use_heavy_tanh),
        heavy_tanh_c=float(cfg.data_prep_heavy_tanh_c),
        add_is_zero_features=bool(cfg.data_prep_add_is_zero_features),
        include_session_id=bool(cfg.data_prep_include_session_id),
        include_inference_splits=bool(cfg.data_prep_include_inference_splits),
        norm_fit_scope=str(cfg.data_prep_norm_fit_scope),
        label_norm=str(cfg.data_prep_label_norm),
        label_norm_fit_scope=str(cfg.data_prep_label_norm_fit_scope),
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


def _load_existing_test_evaluation_manifest(qconf: SimpleNamespace, best_it: int) -> Path:
    """Load the existing test-evaluation manifest for the selected checkpoint."""
    # Resolve the expected streamed test manifest path and require it to exist.
    manifest_path = Path(qconf.root_dir) / "eval_test" / f"iter_{int(best_it)}" / "predict_manifest.yaml"
    if not Path(manifest_path).exists():
        raise RuntimeError(f"Missing existing test-evaluation manifest: {manifest_path}")
    return manifest_path


def _load_existing_eval_manifest(qconf: SimpleNamespace, eval_dir_name: str, best_it: int) -> Path:
    """Load an existing streamed eval manifest for one split."""
    # Resolve the expected streamed manifest path and require it to exist.
    manifest_path = Path(qconf.root_dir) / str(eval_dir_name) / f"iter_{int(best_it)}" / "predict_manifest.yaml"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing existing eval manifest: {manifest_path}")
    return manifest_path


def _load_existing_inference_manifest(qconf: SimpleNamespace, inference_dir_name: str, best_it: int) -> Path:
    """Load an existing streamed inference manifest for one split."""
    # Resolve the expected streamed inference manifest path and require it to exist.
    manifest_path = Path(qconf.root_dir) / str(inference_dir_name) / f"iter_{int(best_it)}" / "inference_manifest.yaml"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing existing inference manifest: {manifest_path}")
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


def _run_inference_manifest_once(qconf: SimpleNamespace, group: str, best_it: int, dst_name: str) -> Path:
    """Run one streamed inference evaluator pass and move it into a stable split-specific directory."""
    # Remove any stale temporary inference directory left by an interrupted prior run.
    import shutil

    run_dir = Path(qconf.root_dir)
    tmp_iter_dir = run_dir / "inference" / f"iter_{int(best_it)}"
    if tmp_iter_dir.exists():
        shutil.rmtree(tmp_iter_dir)

    # Run the evaluator in inference chunked-manifest mode for the requested split.
    if torch.device(qconf.device).type == "cuda":
        from qmodel.core.evaluator import Evaluator

        evaluator = Evaluator(qconf, group=str(group), writer=None, enable_logging=False)
    else:
        from qmodel.core.cpu_evaluator import CpuEvaluator

        evaluator = CpuEvaluator(qconf, group=str(group), writer=None, enable_logging=False)

    # Stream predictions into inference parquet chunks and close evaluator resources.
    evaluator._run_inference_to_manifest(it=int(best_it), n_iter=0, iter_dir=tmp_iter_dir)
    evaluator.close()

    # Move the finished streamed inference output into its stable destination and return the manifest path.
    dst_iter_dir = _move_eval_dir(run_dir, src_name="inference", dst_name=str(dst_name), it=int(best_it))
    manifest_path = Path(dst_iter_dir) / "inference_manifest.yaml"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing streamed inference manifest after move: {manifest_path}")
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

    # Reuse an existing inference-test manifest when it is already on disk.
    inference_test_manifest_path: Path | None = None
    if bool(require_existing_eval):
        candidate = Path(qconf.root_dir) / "inference_test" / f"iter_{int(best_it)}" / "inference_manifest.yaml"
        if candidate.exists():
            inference_test_manifest_path = candidate

    # Rebuild the inference-test manifest when it is missing.
    if inference_test_manifest_path is None:
        inference_test_manifest_path = _run_inference_manifest_once(qconf, "inference_test", int(best_it), "inference_test")

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
    test_ic_summary_yaml = Path(out_root) / "daily_ic_summary_test.yaml"
    t_summary0 = time.time()
    test_ic_summary = daily_pearson_ic_summary_from_manifest(Path(test_manifest_path), test_ic_summary_yaml)

    train_pooled = pooled_pearson_ic_from_manifest(Path(train_manifest_path))
    train_ic_summary_yaml = Path(out_root) / "daily_ic_summary_train.yaml"
    train_ic_summary = daily_pearson_ic_summary_from_manifest(Path(train_manifest_path), train_ic_summary_yaml)
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
            "dataloader_pin_memory": bool(cfg.dataloader_pin_memory),
            "dataloader_prefetch_factor": int(cfg.dataloader_prefetch_factor),
            "dataloader_persistent_workers": bool(cfg.dataloader_persistent_workers),
            "batch_size": int(cfg.batch_size),
            "eval_batch_size": int(cfg.eval_batch_size),
            "num_workers": int(cfg.num_workers),
            "train_profile_section": str(cfg.train_profile_section),
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
    label_meta = dict(meta["label"])
    clean_rows = [
        ("stock_norm", f"{str(meta['feature_transform']['stock_norm']['type'])} / scope={str(meta['feature_transform']['stock_norm'].get('scope', 'n/a'))}"),
        ("label_definition", str(label_meta["definition"])),
        ("label_horizon_minutes", str(int(label_meta["horizon_minutes"]))),
        ("label_entry_lag_minutes", str(int(label_meta["entry_lag_minutes"]))),
        ("label_norm", f"{str(meta.get('label_transform', {'type': 'none'})['type'])} / scope={str(meta.get('label_transform', {'scope': 'n/a'}).get('scope', 'n/a'))}"),
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
        ("train_loss_space", "normalized_label" if str(meta.get("label_transform", {"type": "none"})["type"]) == "pooled_zscore" else "raw_label"),
        ("eval_metric_space", "raw_label / " + str(label_meta["definition"])),
        ("train_loss_png", loss_png.as_posix()),
    ]
    # Summarize the training-time LR schedule and initialization policy for reproducibility.
    lr_sched = _lr_scheduler_contract(cfg, int(prep["train_rows"]))
    lr_init = float(cfg.learning_rate) * float(lr_sched["start_warmup_factor"])
    lr_peak = float(cfg.learning_rate) * float(lr_sched["end_warmup_factor"])
    lr_rows = [
        ("optim/optimizer", "AdamW"),
        ("optim/base_lr", f"{float(cfg.learning_rate):.6g}"),
        ("optim/criterion", "MSELoss"),
        ("lr_scheduler/type", "LinearWarmup + CosineAnnealing (GraphLRScheduler)"),
        ("lr_scheduler/use_lr_sched", "custom"),
        ("lr_scheduler/warmup_iters", str(int(lr_sched["warmup_iters"]))),
        ("lr_scheduler/start_factor", f"{float(lr_sched['start_warmup_factor']):.6g}"),
        ("lr_scheduler/end_factor", f"{float(lr_sched['end_warmup_factor']):.6g}"),
        ("lr_scheduler/finish_decay_iter", str(int(lr_sched["finish_decay_iter"]))),
        ("lr_scheduler/eta_min", f"{float(lr_sched['eta_min']):.6g}"),
        ("lr_scheduler/lr_at_iter0", f"{float(lr_init):.6g}"),
        ("lr_scheduler/lr_after_warmup", f"{float(lr_peak):.6g}"),
    ]
    init_rows = [
        ("param_init/policy", "PyTorch default init (no explicit init in prediction_nn2/model.py)"),
        ("param_init/torch_seed", "not set (init is not guaranteed reproducible)"),
        ("param_init/nn.Linear", "kaiming_uniform_(weight), uniform_(bias) (PyTorch default)"),
        ("param_init/nn.GRU", "uniform_(-1/sqrt(hidden_size), +1/sqrt(hidden_size)) (PyTorch default)"),
    ]
    ic_rows = [
        ("pooled_ic_train", f"{float(train_pooled['pearson_ic']):.6f}"),
        ("pooled_ic_test", f"{float(test_pooled['pearson_ic']):.6f}"),
        ("pooled_count_train", str(int(train_pooled["count"]))),
        ("pooled_count_test", str(int(test_pooled["count"]))),
        ("train_daily_ic_mean", f"{float(train_ic_summary['pearson_ic']['mean']):.6f}"),
        ("test_daily_ic_mean", f"{float(test_ic_summary['pearson_ic']['mean']):.6f}"),
        ("train_daily_ic_std", f"{float(train_ic_summary['pearson_ic']['std']):.6f}"),
        ("test_daily_ic_std", f"{float(test_ic_summary['pearson_ic']['std']):.6f}"),
        ("train_daily_icir", f"{float(train_ic_summary['pearson_ic']['icir']):.6f}"),
        ("test_daily_icir", f"{float(test_ic_summary['pearson_ic']['icir']):.6f}"),
        ("train_daily_t_stat", f"{float(train_ic_summary['pearson_ic']['t_stat']):.4f}"),
        ("test_daily_t_stat", f"{float(test_ic_summary['pearson_ic']['t_stat']):.4f}"),
        ("train_day_count", str(int(train_ic_summary["day_count"]))),
        ("test_day_count", str(int(test_ic_summary["day_count"]))),
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
        ("batch_size", str(int(perf["train"]["batch_size"]))),
        ("num_workers", str(int(perf["train"]["num_workers"]))),
        ("dataloader_pin_memory", str(bool(perf["train"]["dataloader_pin_memory"]))),
        ("dataloader_prefetch_factor", str(int(perf["train"]["dataloader_prefetch_factor"]))),
        ("dataloader_persistent_workers", str(bool(perf["train"]["dataloader_persistent_workers"]))),
        ("train_profile_section", str(perf["train"]["train_profile_section"])),
    ]

    # Assemble the final self-contained single-column HTML report.
    sections = [
        render_section("Run Overview", render_value_rows(overview_rows)),
        render_section("NN Model", render_value_rows(model_rows)),
        render_section("Training Setup", render_value_rows(lr_rows) + render_value_rows(init_rows) + render_yaml_block({"lr_scheduler": lr_sched})),
        render_section("Data Clean Summary", render_value_rows(clean_rows) + invalid_table),
        render_figure("Data Clean Distribution Overview", stats_dir / "pooled_feature_grid.png", "Pooled standardized feature distributions from the data clean stage."),
        render_section("Validation And Checkpoint", render_value_rows(checkpoint_rows) + sweep_table),
        render_figure("Train Loss Curve", loss_png, "Training loss exported from TensorBoard `train/objective/loss_mean`, with raw values and a 1000-step sliding mean overlay."),
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
    for group_name in ["inference_train", "inference_val", "inference_test"]:
        if group_name in groups:
            out[f"{group_name}_rows"] = int(groups[group_name]["rows"])
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
        "include_inference_splits": bool(cfg.data_prep_include_inference_splits),
        "norm_fit_scope": str(cfg.data_prep_norm_fit_scope),
        "label_norm": str(cfg.data_prep_label_norm),
        "label_norm_fit_scope": str(cfg.data_prep_label_norm_fit_scope),
        "label_entry_lag_minutes": int(cfg.label_entry_lag_minutes),
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
    meta_path = Path(out_root) / "artifacts" / "npz" / "meta.yaml"
    stage_cfg = {
        "stage": "clean_report",
        "use_cross_sectional_gaussianize": bool(cfg.use_cross_sectional_gaussianize),
        "norm_fit_scope": str(cfg.data_prep_norm_fit_scope),
        "label_norm": str(cfg.data_prep_label_norm),
        "label_norm_fit_scope": str(cfg.data_prep_label_norm_fit_scope),
        "label_entry_lag_minutes": int(cfg.label_entry_lag_minutes),
        "data_contract": _load_data_contract_from_meta(meta_path),
    }
    manifest_path = _stage_manifest_path(Path(out_root), "clean_report")
    fp = _stage_fingerprint(stage_cfg)

    # Skip the stage when the manifest and report path already match.
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
    meta_path = Path(out_root) / "artifacts" / "npz" / "meta.yaml"
    train_rows = int(_load_train_rows_from_meta(meta_path))
    effective_num_iters = int(_resolved_num_iters(cfg, int(train_rows)))
    stage_cfg = {
        "stage": "train",
        "seed": int(cfg.seed),
        "num_iters": int(effective_num_iters),
        "save_every": int(cfg.save_every),
        "batch_size": int(cfg.batch_size),
        "learning_rate": float(cfg.learning_rate),
        "hidden_dims": list(cfg.hidden_dims),
        "dropout": float(cfg.dropout),
        "input_window_size": int(cfg.input_window_size),
        "feature_dim": int(feature_dim),
        "train_contract": _train_stage_contract(cfg, int(feature_dim), int(train_rows)),
        "data_contract": _load_data_contract_from_meta(meta_path),
    }
    manifest_path = _stage_manifest_path(Path(out_root), "train")
    fp = _stage_fingerprint(stage_cfg)

    # Always build qmodel config so checkpoint paths resolve consistently.
    qconf = _build_qmodel_config(cfg, feature_dim=int(feature_dim), run_root=Path(out_root), train_rows=int(train_rows))
    final_iter = int(effective_num_iters) - 1
    required_last_ckpt = (int(final_iter) // int(cfg.save_every)) * int(cfg.save_every)

    # Skip training when the last required checkpoint already exists and the stage manifest matches.
    ckpt_iters_before = _list_checkpoint_iters(Path(out_root))
    last_ckpt = int(max(ckpt_iters_before)) if len(ckpt_iters_before) else -1
    if manifest_path.exists() and int(last_ckpt) >= int(required_last_ckpt):
        m = _load_stage_manifest(manifest_path)
        if str(m.get("fingerprint")) == str(fp):
            best_it = int(m["best_it"])
            val_metrics_by_it = _load_val_metrics_from_disk(Path(out_root), _list_checkpoint_iters(Path(out_root)))[1]
            return qconf, int(best_it), dict(val_metrics_by_it)

    # Skip training when enough checkpoints exist but a prior run did not write the train manifest.
    if (not manifest_path.exists()) and int(last_ckpt) >= int(required_last_ckpt):
        print(
            f"[pipeline] train skip: last_ckpt={last_ckpt} required_last_ckpt={required_last_ckpt} final_iter={final_iter}",
            flush=True,
        )
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

    # Run training in a single continuous trainer session.
    # Checkpoints are still saved at cfg.save_every, but we avoid chunk restarts to keep GPU utilization stable.
    qconf.num_iters = int(effective_num_iters)
    qconf.save_every = int(cfg.save_every)
    qconf.load_from_iter = -1 if int(last_ckpt) >= 0 else None
    print(
        f"[pipeline] train start last_ckpt={last_ckpt} required_last_ckpt={required_last_ckpt} final_iter={final_iter} "
        f"save_every={int(cfg.save_every)} batch_size={int(cfg.batch_size)} effective_num_iters={int(effective_num_iters)}",
        flush=True,
    )

    # Configure the CUDA allocator to reduce fragmentation when the env var is not already set.
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Run one trainer session on the selected device.
    if torch.device(qconf.device).type == "cuda":
        from qmodel.core.trainer import Trainer

        trainer = Trainer(qconf)
        trainer.train()
    else:
        from qmodel.core.cpu_trainer import CpuTrainer

        trainer = CpuTrainer(qconf)
        trainer.train()

    ckpt_iters_after = _list_checkpoint_iters(Path(out_root))
    last_ckpt = int(max(ckpt_iters_after)) if len(ckpt_iters_after) else -1
    if int(last_ckpt) < int(required_last_ckpt):
        raise RuntimeError(
            f"Training finished but last required checkpoint is missing: last_ckpt={last_ckpt} "
            f"required_last_ckpt={required_last_ckpt} final_iter={final_iter}"
        )
    print(
        f"[pipeline] train done last_ckpt={last_ckpt} required_last_ckpt={required_last_ckpt} final_iter={final_iter}",
        flush=True,
    )

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
    stage_cfg = {
        "stage": "train_report",
        "report_version": 4,
        "best_it": int(best_it),
        "data_contract": _load_data_contract_from_meta(Path(prep["meta_path"])),
        "lr_scheduler": _lr_scheduler_contract(cfg, int(prep["train_rows"])),
        "param_init_policy": "pytorch_default_init_no_manual_seed",
        "ic_summary_mode": "daily_ic_t_stat",
        "effective_num_iters": int(_resolved_num_iters(cfg, int(prep["train_rows"]))),
    }
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


def run_test_evaluation_stage(cfg: PipelineConfig, *, out_root: Path, qconf: SimpleNamespace, best_it: int, require_existing_eval: bool) -> Path:
    """Run test-evaluation inference (or reuse existing) and return the manifest path."""
    # Build a stage fingerprint so test evaluation can be rerun deterministically.
    stage_cfg = {
        "stage": "test_evaluation",
        "best_it": int(best_it),
        "data_contract": _load_data_contract_from_meta(Path(out_root) / "artifacts" / "npz" / "meta.yaml"),
    }
    manifest_path = _stage_manifest_path(Path(out_root), "test_evaluation")
    fp = _stage_fingerprint(stage_cfg)

    # Resolve the expected test manifest output path.
    test_manifest_path = Path(qconf.root_dir) / "eval_test" / f"iter_{int(best_it)}" / "predict_manifest.yaml"
    if manifest_path.exists() and Path(test_manifest_path).exists():
        m = _load_stage_manifest(manifest_path)
        if str(m.get("fingerprint")) == str(fp):
            return test_manifest_path

    # Reuse an existing test manifest for report-only modes.
    if bool(require_existing_eval):
        test_manifest_path = _load_existing_test_evaluation_manifest(qconf, int(best_it))
    else:
        test_manifest_path = Path(test_manifest_path)

    # Reuse or rebuild the streamed test manifest under eval_test.
    if not Path(test_manifest_path).exists():
        test_manifest_path = _run_eval_manifest_once(qconf, "test", int(best_it), "eval_test")

    # Reuse or rebuild the streamed inference-test manifest under inference_test.
    inference_manifest_path = Path(qconf.root_dir) / "inference_test" / f"iter_{int(best_it)}" / "inference_manifest.yaml"
    if bool(require_existing_eval) and Path(inference_manifest_path).exists():
        inference_manifest_path = _load_existing_inference_manifest(qconf, "inference_test", int(best_it))
    elif not Path(inference_manifest_path).exists():
        inference_manifest_path = _run_inference_manifest_once(qconf, "inference_test", int(best_it), "inference_test")

    _write_stage_manifest(
        manifest_path,
        {
            "stage": "test_evaluation",
            "fingerprint": str(fp),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_manifest_path": str(Path(test_manifest_path).as_posix()),
            "inference_manifest_path": str(Path(inference_manifest_path).as_posix()),
        },
    )
    return test_manifest_path


def run_test_evaluation_report_from_manifest(manifest_path: Path, cfg: PipelineConfig, out_root: Path) -> Path:
    """Compute test-evaluation report artifacts from an existing test manifest."""
    # Compute the test-evaluation report in a dedicated folder so heavy artifacts stay isolated.
    out_root = Path(out_root)
    manifest_path = Path(manifest_path)
    report_dir = Path(out_root) / "test_evaluation_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build evaluator config and compute the full test-evaluation report from the manifest.
    eval_cfg = EvalConfig(
        stock1m_dir=Path(cfg.stock1m_dir),
        window_size=int(cfg.rolling_window),
        step_size=int(cfg.rolling_step),
        horizon_minutes=int(cfg.horizon_minutes),
    )
    artifacts = compute_test_evaluation_report_from_manifest(Path(manifest_path), eval_cfg, Path(report_dir))

    # Render one self-contained HTML wrapper around the produced artifacts.
    html_path = Path(out_root) / "test_evaluation_report.html"
    html = _render_test_evaluation_report_html(cfg, manifest_path, artifacts, report_dir)
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _render_test_evaluation_report_html(cfg: PipelineConfig, manifest_path: Path, artifacts, report_dir: Path) -> str:
    """Render the test-evaluation self-contained HTML content."""
    # Summarize the heavy test diagnostics into stacked scalar sections.
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
        ["year", "pearson_ic", "count"],
        [
            [
                str(int(row["year"])),
                f"{float(row['pearson_ic']):.6f}",
                str(int(row["count"])),
            ]
            for row in annual_tbl.to_dict(orient="records")
        ],
    )
    annual_body = render_value_rows([("annual_csv", Path(artifacts.annual_csv).as_posix())]) + annual_table
    if int(annual_tbl.shape[0]) > 1:
        annual_body += render_embedded_figure("Annual IC", Path(artifacts.annual_png), "Annual pooled IC curve from the test manifest.")

    # Assemble the final self-contained single-column HTML report.
    sections = [
        render_section(
            "Test Evaluation Overview",
            render_value_rows(
                [
                    ("generated_at", now),
                    ("test_manifest", Path(manifest_path).as_posix()),
                    ("report_dir", Path(report_dir).as_posix()),
                    ("pooled_pearson_ic", f"{float(pooled['pearson_ic']):.6f}"),
                    ("count", str(int(pooled["count"]))),
                ]
            ),
        ),
        render_section("NN Model", render_value_rows(model_rows)),
        render_section(
            "Daily IC Summary",
            render_value_rows(
                [
                    ("ic_summary_yaml", Path(artifacts.ic_summary_yaml).as_posix()),
                    ("daily_ic_mean", f"{float(ic_summary['pearson_ic']['mean']):.6f}"),
                    ("daily_ic_std", f"{float(ic_summary['pearson_ic']['std']):.6f}"),
                    ("daily_icir", f"{float(ic_summary['pearson_ic']['icir']):.6f}"),
                    ("daily_t_stat", f"{float(ic_summary['pearson_ic']['t_stat']):.4f}"),
                    ("day_count", str(int(ic_summary["day_count"]))),
                ]
            )
            + render_yaml_block(ic_summary),
        ),
        render_section(
            f"Annual IC ({year_range})",
            annual_body,
        ),
        render_section(
            "Intraday IC",
            render_value_rows([("intraday_csv", Path(artifacts.intraday_csv).as_posix())])
            + render_embedded_figure("Test Intraday IC", Path(artifacts.intraday_png), "Test-side intraday IC curve."),
        ),
        render_section(
            "Volatility Rolling IC",
            render_value_rows([("volatility_csv", Path(artifacts.vol_csv).as_posix()), ("volatility_yaml", Path(artifacts.vol_yaml).as_posix())])
            + render_embedded_figure("Test Volatility Rolling IC", Path(artifacts.vol_png), "Rolling IC grouped by volatility label.")
            + render_yaml_block(yaml.safe_load(Path(artifacts.vol_yaml).read_text(encoding="utf-8"))),
        ),
        render_section(
            "Price Rolling IC",
            render_value_rows([("price_csv", Path(artifacts.price_csv).as_posix()), ("price_yaml", Path(artifacts.price_yaml).as_posix())])
            + render_embedded_figure("Test Price Rolling IC", Path(artifacts.price_png), "Rolling IC grouped by price label.")
            + render_yaml_block(yaml.safe_load(Path(artifacts.price_yaml).read_text(encoding="utf-8"))),
        ),
        render_section(
            "Rank Diagnostics",
            render_value_rows([("rank_png", Path(artifacts.rank_png).as_posix())])
            + render_embedded_figure("Test Rank Diagnostics", Path(artifacts.rank_png), "Prediction rank versus target rank diagnostics."),
        ),
        render_section(
            "Turnover",
            render_value_rows([("turnover_csv", Path(artifacts.turnover_csv).as_posix()), ("turnover_yaml", Path(artifacts.turnover_yaml).as_posix())])
            + render_embedded_figure("Test Turnover", Path(artifacts.turnover_png), "Adjacent-timestamp prediction rank turnover.")
            + render_yaml_block(turnover_summary),
        ),
        render_section(
            "Residual Diagnostics",
            render_value_rows([("residual_yaml", Path(artifacts.residual_yaml).as_posix())])
            + render_embedded_figure("Test Residual Diagnostics", Path(artifacts.residual_png), "Residual histogram and scatter diagnostics.")
            + render_yaml_block(residual_summary),
        ),
    ]
    return build_page("Test Evaluation Report", "Self-contained HTML report generated from the test manifest.", sections)


def run_test_evaluation_report_stage(cfg: PipelineConfig, *, out_root: Path, test_manifest_path: Path) -> Path:
    """Build the test-evaluation report from an existing test manifest and return its path."""
    # Build a stage fingerprint so report rebuild can skip when manifest stays the same.
    manifest_stat = Path(test_manifest_path).stat()
    stage_cfg = {
        "stage": "test_evaluation_report",
        "report_version": 3,
        "manifest": str(Path(test_manifest_path).as_posix()),
        "manifest_size_bytes": int(manifest_stat.st_size),
        "manifest_mtime_ns": int(manifest_stat.st_mtime_ns),
    }
    manifest_path = _stage_manifest_path(Path(out_root), "test_evaluation_report")
    fp = _stage_fingerprint(stage_cfg)
    report_html = Path(out_root) / "test_evaluation_report.html"

    # Skip the stage when the manifest and report already match.
    if manifest_path.exists() and report_html.exists():
        m = _load_stage_manifest(manifest_path)
        if str(m.get("fingerprint")) == str(fp):
            return report_html

    # Compute test-evaluation report purely from the test manifest.
    out_path = run_test_evaluation_report_from_manifest(Path(test_manifest_path), cfg, Path(out_root))
    _write_stage_manifest(
        manifest_path,
        {"stage": "test_evaluation_report", "fingerprint": str(fp), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "report_path": str(out_path.as_posix())},
    )
    return out_path


def _run_single_split_staged(cfg: PipelineConfig, *, out_root: Path, start_trade_date: int, end_trade_date: int, train_days: int, val_days: int, test_days: int) -> None:
    """Run one split in the configured pipeline mode using explicit stages."""
    # Dispatch stage execution based on the configured pipeline_mode string.
    mode = str(cfg.pipeline_mode)
    out_root = Path(out_root)

    if mode == "train_full":
        # Run data clean, clean report, train, and train report without inference-heavy stages.
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
        qconf = _build_qmodel_config(cfg, feature_dim=len(prep["feature_names"]), run_root=Path(out_root), train_rows=int(prep["train_rows"]))
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

    if mode == "test_full":
        # Run the full heavy pipeline including test evaluation and the final test report.
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
        test_manifest_path = run_test_evaluation_stage(cfg, out_root=out_root, qconf=qconf, best_it=int(best_it), require_existing_eval=False)
        run_test_evaluation_report_stage(cfg, out_root=out_root, test_manifest_path=test_manifest_path)
        return

    if mode == "test_report_only":
        # Rebuild the test-evaluation report from an existing manifest without rerunning training or data clean.
        meta_path = Path(out_root) / "artifacts" / "npz" / "meta.yaml"
        prep = _load_prep_summary_from_meta(meta_path)
        qconf = _build_qmodel_config(cfg, feature_dim=len(prep["feature_names"]), run_root=Path(out_root), train_rows=int(prep["train_rows"]))
        ckpt_iters = _list_checkpoint_iters(Path(out_root))
        best_it, _val_metrics_by_it = _load_val_metrics_from_disk(Path(out_root), ckpt_iters)
        test_manifest_path = run_test_evaluation_stage(cfg, out_root=out_root, qconf=qconf, best_it=int(best_it), require_existing_eval=True)
        run_test_evaluation_report_stage(cfg, out_root=out_root, test_manifest_path=test_manifest_path)
        return

    raise RuntimeError(f"Unknown pipeline_mode: {mode}")


def _default_config() -> PipelineConfig:
    """Build the module-level default pipeline config."""
    # Define a small default experiment that is GPU-feasible while remaining fully general.
    return PipelineConfig(
        pipeline_mode="train_full",
        root_dir=Path("/data-cache/nn"),
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
        label_entry_lag_minutes=1,
        sample_stocks_per_minute=800,
        use_cross_sectional_gaussianize=False,
        data_prep_use_ret_signed_log1p=True,
        data_prep_use_heavy_tanh=True,
        data_prep_heavy_tanh_c=3.0,
        data_prep_add_is_zero_features=False,
        data_prep_include_session_id=False,
        data_prep_include_inference_splits=True,
        data_prep_norm_fit_scope="train_only",
        data_prep_label_norm="pooled_zscore",
        data_prep_label_norm_fit_scope="train_only",
        data_prep_days_per_call=100,
        data_prep_workers=32,
        batch_size=8192,
        num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        num_iters=140001,
        save_every=35000,
        eval_every=10000,
        eval_during=False,
        eval_during_num_iters=0,
        eval_batch_size=8192,
        train_profile_section="none",
        train_profile_wait=20,
        train_profile_warmup=5,
        train_profile_active=20,
        train_profile_repeat=1,
        learning_rate=1e-3,
        hidden_dims=[512, 512],
        dropout=0.0,
        input_window_size=60,
        rolling_window=2000,
        rolling_step=50,
    )


def main() -> None:
    """Run the pipeline with module-level configuration constants."""
    # Reuse the shared default config so full and postprocess-only entrypoints stay aligned.
    cfg = _default_config()

    # Execute the full pipeline.
    run_pipeline(cfg)


def _run_pipeline_with_split_runner(cfg: PipelineConfig, split_runner) -> None:
    """Resolve split policy and dispatch one runner per resolved split."""
    # Resolve a dated run root under /data-cache/nn so artifacts are grouped by day.
    root_dir = _resolve_data_cache_nn_root_dir(Path(cfg.root_dir), pipeline_mode=str(cfg.pipeline_mode))

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
        label_entry_lag_minutes=int(cfg.label_entry_lag_minutes),
        sample_stocks_per_minute=int(_effective_sample_stocks_per_minute(cfg)),
        use_cross_sectional_gaussianize=bool(cfg.use_cross_sectional_gaussianize),
        use_ret_signed_log1p=bool(cfg.data_prep_use_ret_signed_log1p),
        use_heavy_tanh=bool(cfg.data_prep_use_heavy_tanh),
        heavy_tanh_c=float(cfg.data_prep_heavy_tanh_c),
        add_is_zero_features=bool(cfg.data_prep_add_is_zero_features),
        include_session_id=bool(cfg.data_prep_include_session_id),
        include_inference_splits=False,
        norm_fit_scope="train_only",
        label_norm=str(cfg.data_prep_label_norm),
        label_norm_fit_scope=str(cfg.data_prep_label_norm_fit_scope),
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
    """Execute the staged pipeline under /data-cache/nn based on cfg.pipeline_mode."""
    # Dispatch split execution through the staged runner so inference work is optional.
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


def run_test_evaluation_report_postprocess_only(cfg: PipelineConfig) -> None:
    """Rebuild test_evaluation_report.html from an existing test manifest without rerunning training."""
    # Force pipeline_mode to the test-report-only mode and dispatch through the staged runner.
    from dataclasses import replace

    _run_pipeline_with_split_runner(replace(cfg, pipeline_mode="test_report_only"), _run_single_split_staged)


if __name__ == "__main__":
    main()
