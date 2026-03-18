"""Run the end-to-end pipeline: data prep, training, evaluation, and report generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from qmodel.config import LRSchedulerConfig
from qmodel.metrics import builtin

from prediction_nn2.data_prep import DataPrepConfig, list_trade_dates, prepare_npz_splits
from prediction_nn2.dataset import NpzDatasetSpec, Stock1mNpzDataset
from prediction_nn2.eval_ic import (
    EvalConfig,
    attach_labels,
    annual_pooled_ic,
    intraday_time_series_ic,
    load_eval_predictions,
    pooled_ic,
    price_rolling_ic,
    score_ret_rank_plot,
    volatility_rolling_ic,
)
from prediction_nn2.model import MlpConfig, MlpRegressor


@dataclass(frozen=True)
class PipelineConfig:
    """Define top-level pipeline knobs and output paths."""

    root_dir: Path
    stock1m_dir: Path
    start_trade_date: int
    end_trade_date: int
    split_policy: str
    train_days: int
    val_days: int
    test_days: int
    rolling_step_days: int
    seed: int
    horizon_minutes: int
    sample_stocks_per_minute: int
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
    dataset_spec = NpzDatasetSpec(data_dir=npz_dir, pin_memory=use_cuda)

    def dataset_class(group: str, dtype: torch.dtype) -> Stock1mNpzDataset:
        """Build a dataset instance for the requested split."""
        # Create the dataset with an explicit NPZ spec.
        return Stock1mNpzDataset(group, dtype, dataset_spec)

    # Build model config and core training components.
    model_cfg = MlpConfig(
        in_dim=int(feature_dim),
        hidden_dims=list(cfg.hidden_dims),
        dropout=float(cfg.dropout),
        dtype=torch.float32,
    )

    # Build evaluator config namespace to match qmodel evaluator expectations.
    evaluator = SimpleNamespace(
        eval_checkpoint_iter=[int(cfg.num_iters) - 1],
        eval_all_num_iters=0,
        eval_batch_size=int(cfg.eval_batch_size),
    )

    # Assemble a flat config object matching qmodel CpuTrainer/CpuEvaluator field access.
    conf = SimpleNamespace(
        device=device,
        dist_backend=None,
        window_size=1,
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


def _run_single_split(cfg: PipelineConfig, *, out_root: Path, start_trade_date: int, end_trade_date: int, train_days: int, val_days: int, test_days: int) -> None:
    """Run one end-to-end split: prep, train, val-select, test-eval, IC report."""
    # Prepare output directories early so downstream code can assume they exist.
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "artifacts").mkdir(parents=True, exist_ok=True)

    # Run data preparation and persist NPZ splits and distribution artifacts.
    print(f"[pipeline] data_prep start out_root={out_root.as_posix()} start={start_trade_date} end={end_trade_date}", flush=True)
    t_prep0 = time.time()
    prep = prepare_npz_splits(
        DataPrepConfig(
            stock1m_dir=Path(cfg.stock1m_dir),
            out_dir=Path(out_root) / "artifacts",
            start_trade_date=int(start_trade_date),
            end_trade_date=int(end_trade_date),
            train_days=int(train_days),
            val_days=int(val_days),
            test_days=int(test_days),
            seed=int(cfg.seed),
            horizon_minutes=int(cfg.horizon_minutes),
            sample_stocks_per_minute=int(cfg.sample_stocks_per_minute),
            workers=32,
        )
    )
    t_prep1 = time.time()
    if not bool(prep["done"]):
        print(
            f"[pipeline] data_prep partial seconds={float(t_prep1 - t_prep0):.2f} next_stage={prep['stage']} index={int(prep['index'])}",
            flush=True,
        )
        return
    print(f"[pipeline] data_prep done seconds={float(t_prep1 - t_prep0):.2f} train_rows={int(prep['train_rows'])}", flush=True)

    # Build qmodel config and run training on the selected device.
    qconf = _build_qmodel_config(cfg, feature_dim=len(prep["feature_names"]), run_root=Path(out_root))
    device = torch.device(qconf.device)
    pin_memory = bool(device.type == "cuda")

    # Chunk training by checkpoint interval so repeated invocations can eventually reach 100k+ iters.
    final_iter = int(cfg.num_iters) - 1
    ckpt_iters_before = _list_checkpoint_iters(Path(out_root))
    last_ckpt = int(max(ckpt_iters_before)) if len(ckpt_iters_before) else -1
    if int(last_ckpt) < int(final_iter):
        # Decide the next target iteration as the next save boundary, capped at the final iteration.
        step = int(cfg.save_every)
        next_target = int(min(int(final_iter), int(last_ckpt + step) if int(last_ckpt) >= 0 else int(step)))
        qconf.num_iters = int(next_target + 1)
        qconf.load_from_iter = -1 if int(last_ckpt) >= 0 else None
        print(
            f"[pipeline] train chunk start last_ckpt={last_ckpt} target_iter={next_target} batch_size={int(cfg.batch_size)}",
            flush=True,
        )

        if torch.device(qconf.device).type == "cuda":
            from qmodel.core.trainer import Trainer

            trainer = Trainer(qconf)
            trainer.train()
        else:
            from qmodel.core.cpu_trainer import CpuTrainer

            trainer = CpuTrainer(qconf)
            trainer.train()

        ckpt_iters_after = _list_checkpoint_iters(Path(out_root))
        last_after = int(max(ckpt_iters_after)) if len(ckpt_iters_after) else -1
        if int(last_after) < int(final_iter):
            print(f"[pipeline] train partial last_ckpt={last_after} final_iter={final_iter}", flush=True)
            return
    else:
        print(f"[pipeline] train skip already_complete last_ckpt={last_ckpt} final_iter={final_iter}", flush=True)

    # Evaluate validation metrics for all checkpoints and pick the best one.
    ckpt_iters = _list_checkpoint_iters(Path(out_root))
    best_it, val_metrics_by_it = _select_best_checkpoint_by_val(Path(out_root), qconf, ckpt_iters)
    print(f"[pipeline] train done best_it={int(best_it)} ckpts={len(ckpt_iters)}", flush=True)

    # Export a loss-curve plot from TensorBoard events for the training log requirement.
    loss_png = Path(out_root) / "train_loss.png"
    _export_train_loss_curve(Path(qconf.root_dir) / "tb", loss_png)

    # Run evaluator on test split only once for the selected checkpoint.
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

    # Move test evaluator outputs aside before running predict evaluation to avoid overwrite.
    test_eval_iter_dir = _move_eval_dir(Path(qconf.root_dir), src_name="eval", dst_name="eval_test", it=int(best_it))

    # Load test predictions and compute all required IC diagnostics.
    test_shard_path = Path(test_eval_iter_dir) / "rank0.feather"
    pred_df = load_eval_predictions(test_shard_path)
    pooled = pooled_ic(pred_df)

    # Emit intraday IC CSV and plot.
    intraday_csv = Path(out_root) / "intraday_ic.csv"
    intraday_png = Path(out_root) / "intraday_ic.png"
    intraday_time_series_ic(pred_df, intraday_csv, intraday_png)

    # Emit volatility and price rolling IC CSVs and plots.
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

    # Attach volatility/price labels once and reuse across group IC computations.
    t_attach0 = time.time()
    labeled_df = attach_labels(pred_df, eval_cfg)
    t_attach1 = time.time()

    # Compute rolling curves on the reused labeled dataframe.
    t_roll0 = time.time()
    vol_agg = volatility_rolling_ic(labeled_df, eval_cfg, vol_csv, vol_png)
    price_agg = price_rolling_ic(labeled_df, eval_cfg, price_csv, price_png)
    t_roll1 = time.time()

    # Emit score-vs-target rank curve plot as the 4th required figure.
    rank_png = Path(out_root) / "pred_vs_target_rank.png"
    score_ret_rank_plot(pred_df, rank_png)

    # Run long-horizon predict evaluation and annual IC reporting.
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

    # Compute pooled/annual/intraday/rolling IC on the full-span predict set.
    predict_shard_path = Path(qconf.root_dir) / "eval" / f"iter_{int(best_it)}" / "rank0.feather"
    predict_df = load_eval_predictions(predict_shard_path)
    predict_pooled = pooled_ic(predict_df)
    annual_csv = Path(out_root) / "annual_ic.csv"
    annual_png = Path(out_root) / "annual_ic.png"
    annual_tbl = annual_pooled_ic(predict_df, annual_csv, annual_png)
    predict_intraday_csv = Path(out_root) / "predict_intraday_ic.csv"
    predict_intraday_png = Path(out_root) / "predict_intraday_ic.png"
    intraday_time_series_ic(predict_df, predict_intraday_csv, predict_intraday_png)

    # Attach labels and compute rolling IC curves on the long-horizon predict set.
    t_attach_pred0 = time.time()
    predict_labeled = attach_labels(predict_df, eval_cfg)
    t_attach_pred1 = time.time()
    predict_vol_csv = Path(out_root) / "predict_vol_rolling_ic.csv"
    predict_vol_png = Path(out_root) / "predict_vol_rolling_ic.png"
    predict_price_csv = Path(out_root) / "predict_price_rolling_ic.csv"
    predict_price_png = Path(out_root) / "predict_price_rolling_ic.png"
    t_roll_pred0 = time.time()
    volatility_rolling_ic(predict_labeled, eval_cfg, predict_vol_csv, predict_vol_png)
    price_rolling_ic(predict_labeled, eval_cfg, predict_price_csv, predict_price_png)
    t_roll_pred1 = time.time()

    # Persist a small performance audit record for evaluation-time bottlenecks.
    perf = {
        "data_prep": {
            "elapsed_seconds": float(t_prep1 - t_prep0),
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
            "predict_attach_labels_seconds": float(t_attach_pred1 - t_attach_pred0),
            "predict_rolling_ic_seconds": float(t_roll_pred1 - t_roll_pred0),
        }
    }
    import yaml

    (Path(out_root) / "perf_audit.yaml").write_text(yaml.safe_dump(perf, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Write a conclusion-driven markdown report with required parts.
    report_path = Path(out_root) / "report.md"
    report = _render_report(
        cfg,
        prep,
        pooled,
        best_it,
        val_metrics_by_it,
        intraday_csv,
        vol_csv,
        price_csv,
        intraday_png,
        vol_png,
        price_png,
        rank_png,
        loss_png,
        vol_agg,
        price_agg,
        predict_pooled,
        annual_tbl,
        annual_csv,
        annual_png,
        predict_intraday_csv,
        predict_intraday_png,
        predict_vol_csv,
        predict_vol_png,
        predict_price_csv,
        predict_price_png,
        perf,
    )
    report_path.write_text(report, encoding="utf-8")


def _render_report(
    cfg: PipelineConfig,
    prep: dict[str, object],
    pooled: dict[str, float],
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
    predict_intraday_csv: Path,
    predict_intraday_png: Path,
    predict_vol_csv: Path,
    predict_vol_png: Path,
    predict_price_csv: Path,
    predict_price_png: Path,
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
    predict_lines = (
        f"- Pearson IC: {predict_pooled['pearson_ic']:.6f}\n"
        f"- Rank IC (Spearman): {predict_pooled['rank_ic']:.6f}\n"
        f"- Count: {int(predict_pooled['count'])}"
    )
    split_meta_path = Path(prep["meta_path"])
    import yaml

    split_meta = yaml.safe_load(split_meta_path.read_text(encoding="utf-8"))
    audit = dict(split_meta["audit"])
    audit_rates = dict(split_meta["audit_rates"])
    dates = dict(split_meta["dates"])
    tb_dir = split_meta_path.parent.parent.parent / "run" / "tb"

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

    # Extract annual IC for 2021 vs 2026 for the requested regime comparison.
    annual_idx = annual_tbl.set_index("year")
    ic_2021 = float(annual_idx.loc[2021, "pearson_ic"])
    ic_2026 = float(annual_idx.loc[2026, "pearson_ic"])
    rank_ic_2021 = float(annual_idx.loc[2021, "rank_ic"])
    rank_ic_2026 = float(annual_idx.loc[2026, "rank_ic"])

    # Render a structured markdown report matching the required sections.
    md = f"""# 神经网络预测与多维度 IC 评估报告

生成时间: {now}

## 结论摘要 (Conclusion)

- 最佳 Checkpoint: iter={best_it} (approx_epoch={approx_epoch:.2f}), Val MSE={val_mse:.6e}, Val IC={val_ic:.6f}, Val RankIC={val_rank_ic:.6f}.
- Test Pooled IC: Pearson={pooled['pearson_ic']:.6f}, RankIC={pooled['rank_ic']:.6f}, Count={int(pooled['count'])}.
- Predict Pooled IC (2021-2026 全周期): Pearson={predict_pooled['pearson_ic']:.6f}, RankIC={predict_pooled['rank_ic']:.6f}, Count={int(predict_pooled['count'])}.
- Annual IC 对比: 2021 Pearson={ic_2021:.6f}, RankIC={rank_ic_2021:.6f}; 2026 Pearson={ic_2026:.6f}, RankIC={rank_ic_2026:.6f}.
- Volatility 分组拐点: {vol_desc}.
- Price 分组拐点: {price_desc}.

## 数据集元数据 (Dataset Metadata)

- Split 配置文件: `{split_meta_path.as_posix()}`
- 日期范围: train={_range_str(list(dates['train']))}, val={_range_str(list(dates['val']))}, test={_range_str(list(dates['test']))}, predict={_range_str(list(dates['predict']))}.
- 样本量: train={int(prep['train_rows'])}, val={int(prep['val_rows'])}, test={int(prep['test_rows'])}, predict={int(prep['predict_rows'])}.
- 缺失/采样审计: train={_missing_str(dict(audit['train']), dict(audit_rates['train']))}, val={_missing_str(dict(audit['val']), dict(audit_rates['val']))}, test={_missing_str(dict(audit['test']), dict(audit_rates['test']))}, predict={_missing_str(dict(audit['predict']), dict(audit_rates['predict']))}.

## 模型与训练协议 (Experiment Protocol)

- 训练设备: `{str(perf['train']['device'])}`, DataLoader `pin_memory={bool(perf['train']['pin_memory'])}`, `num_workers={int(perf['train']['num_workers'])}`.
- Checkpoint 选择: 仅使用 Validation (`val/objective/mse`) 选取最佳 iter, Test 仅做一次性最终评估.
- TensorBoard: `{tb_dir.as_posix()}` (loss 标量: `train/objective/loss`).

## 模型性能 (Model Performance)

- 最佳 Checkpoint: iter={best_it}, approx_epoch={approx_epoch:.2f}.
- Val: MSE={val_mse:.6e}, IC={val_ic:.6f}, RankIC={val_rank_ic:.6f}.
- Test Pooled:\n{pooled_lines}
- Predict Pooled:\n{predict_lines}

## 分组结论 (Group Findings)

- Volatility rolling IC: {vol_desc}.
- Price rolling IC: {price_desc}.

## 性能审计 (Performance Audit)

- data_prep_seconds={float(perf['data_prep']['elapsed_seconds']):.4f}, test_attach_labels_seconds={float(perf['eval']['attach_labels_seconds']):.4f}, test_rolling_ic_seconds={float(perf['eval']['rolling_ic_seconds']):.4f}.
- predict_attach_labels_seconds={float(perf['eval']['predict_attach_labels_seconds']):.4f}, predict_rolling_ic_seconds={float(perf['eval']['predict_rolling_ic_seconds']):.4f}.

## 文件输出 (Artifacts)

- `{intraday_csv.as_posix()}`
- `{vol_csv.as_posix()}`
- `{price_csv.as_posix()}`
- `{annual_csv.as_posix()}`

### 四张核心图表

1. Intraday IC Curve: `{intraday_png.as_posix()}`
2. Volatility Rolling IC: `{vol_png.as_posix()}`
3. Price Rolling IC: `{price_png.as_posix()}`
4. 预测收益率 vs 实际收益率 Rank 曲线: `{rank_png.as_posix()}`

### 训练与长周期诊断

1. Train Loss Curve: `{loss_png.as_posix()}`
2. Annual Pooled IC: `{annual_png.as_posix()}`
3. Predict Intraday IC: `{predict_intraday_png.as_posix()}`
4. Predict Volatility Rolling IC: `{predict_vol_png.as_posix()}`
5. Predict Price Rolling IC: `{predict_price_png.as_posix()}`
"""
    return md


def main() -> None:
    """Run the pipeline with module-level configuration constants."""
    # Define a small default experiment that is GPU-feasible while remaining fully general.
    cfg = PipelineConfig(
        root_dir=Path("outputs") / "upgrade_20260318",
        stock1m_dir=Path("/data/ashare/market/stock1m"),
        start_trade_date=20210101,
        end_trade_date=20260317,
        split_policy="tail_holdout",
        train_days=6,
        val_days=1,
        test_days=1,
        rolling_step_days=1,
        seed=7,
        horizon_minutes=30,
        sample_stocks_per_minute=800,
        batch_size=8192,
        num_workers=4,
        num_iters=120001,
        save_every=20000,
        eval_every=20000,
        eval_during=False,
        eval_during_num_iters=0,
        eval_batch_size=8192,
        learning_rate=2e-3,
        hidden_dims=[512, 512],
        dropout=0.1,
        rolling_window=100,
        rolling_step=50,
    )

    # Execute the full pipeline.
    run_pipeline(cfg)

def run_pipeline(cfg: PipelineConfig) -> None:
    """Execute data prep, GPU training, evaluation, and report output under /data-cache."""
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
        sample_stocks_per_minute=int(cfg.sample_stocks_per_minute),
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
        _run_single_split(
            cfg,
            out_root=out_root,
            start_trade_date=int(cfg.start_trade_date),
            end_trade_date=int(cfg.end_trade_date),
            train_days=int(train_days),
            val_days=int(cfg.val_days),
            test_days=int(cfg.test_days),
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
            _run_single_split(
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


if __name__ == "__main__":
    main()
