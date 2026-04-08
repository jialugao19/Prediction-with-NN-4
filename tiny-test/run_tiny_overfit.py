"""Run a tiny-sample overfit experiment and write a self-contained research report."""

from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Bootstrap the repo root so direct script execution can import project modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the shared training helpers and report renderer after bootstrapping sys.path.
from prediction_nn2.pipeline import _build_qmodel_config, _default_config
from tiny_overfit_report import render_tiny_overfit_report


matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Pin the allocator setting before qmodel touches CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


SOURCE_OUT_ROOT = Path("/data-cache/nn/upgrade_20260328_gru_seq60_h10/date_ranges")
SOURCE_NPZ_DIR = SOURCE_OUT_ROOT / "artifacts" / "npz"
SOURCE_DATA_CLEAN_DIR = SOURCE_OUT_ROOT / "artifacts" / "data_clean"
TINY_EXP_ROOT = Path("/data-cache/nn/overfit_20260408_gru_seq60_h10_tiny")
TINY_OUT_ROOT = TINY_EXP_ROOT / "date_ranges"
REPORT_DIR = Path("/home/maomao/prediction-NN-2/report/0408")

SOURCE_TRAIN_GROUP = {
    "rows": 768564829,
    "feature_dim": 18,
    "x": "train_x.f32",
    "y": "train_y.f32",
    "meta": "train_meta.i64",
}
SOURCE_FEATURE_NAMES = [
    "ret_1m",
    "ret_5m",
    "ret_2m",
    "ret_3m",
    "ret_10m",
    "hl",
    "oc",
    "log_vol",
    "log_amount",
    "vol_10m",
    "log_vol_5m_mean",
    "log_amount_5m_mean",
    "ret_1m_is_zero",
    "ret_5m_is_zero",
    "hl_is_zero",
    "oc_is_zero",
    "minute_norm",
    "session_minute_norm",
]
SOURCE_PREP_CONFIG = {
    "start_trade_date": 20210104,
    "end_trade_date": 20241231,
    "train_days": 727,
    "val_days": 37,
    "test_days": 205,
    "seed": 7,
    "horizon_minutes": 10,
    "sample_stocks_per_minute": 0,
    "use_cross_sectional_gaussianize": False,
    "use_ret_signed_log1p": True,
    "use_heavy_tanh": True,
    "heavy_tanh_c": 3.0,
    "add_is_zero_features": False,
    "include_session_id": False,
    "norm_fit_scope": "train_only",
    "label_norm": "pooled_zscore",
    "label_norm_fit_scope": "train_only",
}
SOURCE_FEATURE_TRANSFORM = {
    "stock_features": [
        "ret_1m",
        "ret_5m",
        "ret_2m",
        "ret_3m",
        "ret_10m",
        "hl",
        "oc",
        "log_vol",
        "log_amount",
        "vol_10m",
        "log_vol_5m_mean",
        "log_amount_5m_mean",
    ],
    "is_zero_features": [],
    "value_transforms": {
        "ret_signed_log1p": {
            "enabled": True,
            "features": ["ret_1m", "ret_2m", "ret_3m", "ret_5m", "ret_10m"],
        },
        "heavy_tanh": {
            "enabled": True,
            "c": 3.0,
            "features": ["hl", "oc", "vol_10m"],
        },
        "feature_warmup_minutes": 10,
    },
    "stock_norm": {
        "type": "pooled_zscore",
        "scope": "train_only",
        "params_path": "pooled_zscore.yaml",
    },
    "time_features": ["minute_norm", "session_minute_norm"],
}
SOURCE_LABEL = {"type": "forward_log_return", "horizon_minutes": 10}
SOURCE_LABEL_TRANSFORM = {"type": "pooled_zscore", "scope": "train_only", "params_path": "label_zscore.yaml"}

TRAIN_RUN_COUNT = 16
VAL_RUN_COUNT = 4
TEST_RUN_COUNT = 4
EXPECTED_RUN_LENGTH = 220
WINDOW_SIZE = 60
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
NUM_ITERS = 4001
SAVE_EVERY = 500
LEARNING_RATE = 1e-3


def _feature_keep_indices(feature_names: list[str]) -> tuple[list[int], list[str]]:
    """Keep the current feature set by removing the four *_is_zero columns."""
    # Build the kept feature index list from the existing source feature order.
    keep_indices = [int(i) for i, feature_name in enumerate(list(feature_names)) if not str(feature_name).endswith("_is_zero")]

    # Materialize the kept feature names for the subset meta.
    keep_names = [str(feature_names[i]) for i in list(keep_indices)]
    return keep_indices, keep_names


def _scan_first_stock_day_runs(meta_path: Path, total_rows: int, need_runs: int) -> list[dict[str, int]]:
    """Scan the first trade day and return a fixed number of full stock-day runs."""
    # Memory-map the source meta rows so run detection stays cheap.
    meta_arr = np.memmap(meta_path, mode="r", dtype=np.int64, shape=(int(total_rows), 3))

    # Walk forward until enough same-code same-date runs have been collected.
    runs: list[dict[str, int]] = []
    start = 0
    first_date = int(meta_arr[0, 1])
    for row in range(1, int(total_rows)):
        same_code = int(meta_arr[row, 0]) == int(meta_arr[row - 1, 0])
        same_date = int(meta_arr[row, 1]) == int(meta_arr[row - 1, 1])
        if same_code and same_date:
            continue
        run = {
            "row_start": int(start),
            "row_stop": int(row),
            "row_count": int(row - start),
            "code": int(meta_arr[start, 0]),
            "date": int(meta_arr[start, 1]),
            "time_start": int(meta_arr[start, 2]),
            "time_stop": int(meta_arr[row - 1, 2]),
        }
        if int(run["date"]) != int(first_date):
            break
        runs.append(run)
        if len(runs) >= int(need_runs):
            break
        start = int(row)

    # Require the expected number of same-length runs.
    if len(runs) != int(need_runs):
        raise RuntimeError(f"Expected {need_runs} runs, got {len(runs)}")
    if len({int(run['row_count']) for run in list(runs)}) != 1:
        raise RuntimeError("Tiny overfit subset requires a single run length.")
    if int(runs[0]["row_count"]) != int(EXPECTED_RUN_LENGTH):
        raise RuntimeError(f"Expected run length {EXPECTED_RUN_LENGTH}, got {runs[0]['row_count']}")
    return runs


def _group_runs(all_runs: list[dict[str, int]]) -> dict[str, list[dict[str, int]]]:
    """Split the scanned runs into train, val, and test groups."""
    # Slice the ordered run list into train, val, and test partitions.
    train_runs = list(all_runs[:TRAIN_RUN_COUNT])
    val_runs = list(all_runs[TRAIN_RUN_COUNT : TRAIN_RUN_COUNT + VAL_RUN_COUNT])
    test_runs = list(all_runs[TRAIN_RUN_COUNT + VAL_RUN_COUNT : TRAIN_RUN_COUNT + VAL_RUN_COUNT + TEST_RUN_COUNT])

    # Return the three split run lists under stable group keys.
    return {
        "train": train_runs,
        "val": val_runs,
        "test": test_runs,
    }


def _copy_group_arrays(
    keep_indices: list[int],
    grouped_runs: dict[str, list[dict[str, int]]],
) -> dict[str, dict[str, int]]:
    """Copy the selected raw rows into a dedicated tiny-dataset directory."""
    # Recreate the tiny output root so every run starts from a clean slate.
    if Path(TINY_OUT_ROOT).exists():
        shutil.rmtree(Path(TINY_OUT_ROOT))
    (Path(TINY_OUT_ROOT) / "artifacts" / "npz").mkdir(parents=True, exist_ok=True)
    (Path(TINY_OUT_ROOT) / "artifacts" / "data_clean").mkdir(parents=True, exist_ok=True)

    # Open the source train split memmaps once and reuse them across groups.
    total_rows = int(SOURCE_TRAIN_GROUP["rows"])
    x_source = np.memmap(Path(SOURCE_NPZ_DIR) / str(SOURCE_TRAIN_GROUP["x"]), mode="r", dtype=np.float32, shape=(int(total_rows), int(SOURCE_TRAIN_GROUP["feature_dim"])))
    y_source = np.memmap(Path(SOURCE_NPZ_DIR) / str(SOURCE_TRAIN_GROUP["y"]), mode="r", dtype=np.float32, shape=(int(total_rows), 1))
    meta_source = np.memmap(Path(SOURCE_NPZ_DIR) / str(SOURCE_TRAIN_GROUP["meta"]), mode="r", dtype=np.int64, shape=(int(total_rows), 3))

    # Materialize one raw-bin file per group with the requested feature subset.
    group_rows: dict[str, dict[str, int]] = {}
    for group_name, runs in dict(grouped_runs).items():
        row_spans = [slice(int(run["row_start"]), int(run["row_stop"])) for run in list(runs)]
        x_parts = [np.asarray(x_source[row_span, :][:, keep_indices], dtype=np.float32) for row_span in list(row_spans)]
        y_parts = [np.asarray(y_source[row_span, :], dtype=np.float32) for row_span in list(row_spans)]
        meta_parts = [np.asarray(meta_source[row_span, :], dtype=np.int64) for row_span in list(row_spans)]

        x_group = np.ascontiguousarray(np.concatenate(x_parts, axis=0), dtype=np.float32)
        y_group = np.ascontiguousarray(np.concatenate(y_parts, axis=0), dtype=np.float32)
        meta_group = np.ascontiguousarray(np.concatenate(meta_parts, axis=0), dtype=np.int64)

        x_path = Path(TINY_OUT_ROOT) / "artifacts" / "npz" / f"{group_name}_x.f32"
        y_path = Path(TINY_OUT_ROOT) / "artifacts" / "npz" / f"{group_name}_y.f32"
        meta_path = Path(TINY_OUT_ROOT) / "artifacts" / "npz" / f"{group_name}_meta.i64"
        x_group.tofile(x_path)
        y_group.tofile(y_path)
        meta_group.tofile(meta_path)

        valid_windows = int(len(runs) * (EXPECTED_RUN_LENGTH - WINDOW_SIZE + 1))
        group_rows[group_name] = {
            "rows": int(x_group.shape[0]),
            "feature_dim": int(x_group.shape[1]),
            "valid_windows": int(valid_windows),
        }

    # Copy the persisted label/feature normalization sidecars needed by the evaluator.
    shutil.copyfile(Path(SOURCE_DATA_CLEAN_DIR) / "label_zscore.yaml", Path(TINY_OUT_ROOT) / "artifacts" / "data_clean" / "label_zscore.yaml")
    shutil.copyfile(Path(SOURCE_DATA_CLEAN_DIR) / "pooled_zscore.yaml", Path(TINY_OUT_ROOT) / "artifacts" / "data_clean" / "pooled_zscore.yaml")
    return group_rows


def _write_tiny_meta(
    keep_feature_names: list[str],
    grouped_runs: dict[str, list[dict[str, int]]],
    group_rows: dict[str, dict[str, int]],
) -> Path:
    """Write a compact meta.yaml that points at the tiny raw-bin dataset."""
    # Build a compact prep_config that matches the current feature choice.
    prep_config = dict(SOURCE_PREP_CONFIG)
    prep_config["start_trade_date"] = int(grouped_runs["train"][0]["date"])
    prep_config["end_trade_date"] = int(grouped_runs["test"][-1]["date"])
    prep_config["train_days"] = 1
    prep_config["val_days"] = 1
    prep_config["test_days"] = 1
    prep_config["add_is_zero_features"] = False

    # Build the feature transform section with the dropped *_is_zero features removed.
    feature_transform = dict(SOURCE_FEATURE_TRANSFORM)

    # Build a compact date list because every tiny split comes from the same trade date.
    split_dates = {
        "train": [int(grouped_runs["train"][0]["date"])],
        "val": [int(grouped_runs["val"][0]["date"])],
        "test": [int(grouped_runs["test"][0]["date"])],
    }

    # Build the storage section that points at the tiny raw-bin group files.
    storage = {
        "format": "raw_bin_v1",
        "dtype": {"x": "float32", "y": "float32", "meta": "int64"},
        "groups": {
            group_name: {
                "rows": int(group_rows[group_name]["rows"]),
                "feature_dim": int(group_rows[group_name]["feature_dim"]),
                "x": f"{group_name}_x.f32",
                "y": f"{group_name}_y.f32",
                "meta": f"{group_name}_meta.i64",
            }
            for group_name in ["train", "val", "test"]
        },
    }

    # Record the tiny subset contract for the later report.
    tiny_subset = {
        "source_out_root": str(Path(SOURCE_OUT_ROOT).as_posix()),
        "feature_names": list(keep_feature_names),
        "window_size": int(WINDOW_SIZE),
        "groups": {
            group_name: {
                "run_count": int(len(grouped_runs[group_name])),
                "row_count": int(group_rows[group_name]["rows"]),
                "valid_windows": int(group_rows[group_name]["valid_windows"]),
                "date": int(grouped_runs[group_name][0]["date"]),
                "codes": [int(run["code"]) for run in list(grouped_runs[group_name])],
            }
            for group_name in ["train", "val", "test"]
        },
    }

    # Write the compact meta and subset YAML sidecars.
    meta_payload = {
        "prep_config": prep_config,
        "feature_names": list(keep_feature_names),
        "storage": storage,
        "feature_transform": feature_transform,
        "dates": split_dates,
        "label": dict(SOURCE_LABEL),
        "label_transform": dict(SOURCE_LABEL_TRANSFORM),
        "sampling": {"sample_stocks_per_minute": 0},
        "tiny_subset": tiny_subset,
    }
    meta_path = Path(TINY_OUT_ROOT) / "artifacts" / "npz" / "meta.yaml"
    meta_path.write_text(yaml.safe_dump(meta_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    subset_path = Path(TINY_OUT_ROOT) / "artifacts" / "npz" / "tiny_subset.yaml"
    subset_path.write_text(yaml.safe_dump(tiny_subset, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return meta_path


def _train_tiny_model(feature_dim: int) -> tuple[object, list[int]]:
    """Train the current NN model on the tiny dataset and return qconf plus checkpoints."""
    # Build a dedicated pipeline config for the tiny overfit run.
    cfg = _default_config()
    cfg = replace(
        cfg,
        pipeline_mode="train_full",
        root_dir=Path("outputs") / "overfit_20260408_gru_seq60_h10_tiny",
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_iters=NUM_ITERS,
        save_every=SAVE_EVERY,
        eval_every=SAVE_EVERY,
        learning_rate=LEARNING_RATE,
        num_workers=4,
    )

    # Build qmodel config on top of the tiny out_root and make logging denser.
    qconf = _build_qmodel_config(cfg, feature_dim=int(feature_dim), run_root=Path(TINY_OUT_ROOT))
    qconf.log_every = 10
    qconf.mean_loss_length = 200
    qconf.load_from_iter = None

    # Run one trainer session on the active device.
    if torch.device(qconf.device).type == "cuda":
        from qmodel.core.trainer import Trainer

        trainer = Trainer(qconf)
    else:
        from qmodel.core.cpu_trainer import CpuTrainer

        trainer = CpuTrainer(qconf)
    trainer.train()

    # Return the trained config together with the emitted checkpoint iterations.
    ckpt_dir = Path(qconf.root_dir) / "ckpt"
    ckpt_iters = sorted(int(path.stem.split("_")[1]) for path in ckpt_dir.glob("iter_*.pt"))
    return qconf, ckpt_iters


def _evaluate_checkpoints(qconf: object, ckpt_iters: list[int]) -> list[dict[str, float]]:
    """Evaluate train and val metrics for every saved checkpoint."""
    # Build one evaluator per split so repeated checkpoint loops stay cheap.
    if torch.device(qconf.device).type == "cuda":
        from qmodel.core.evaluator import Evaluator

        train_evaluator = Evaluator(qconf, group="train", writer=None, enable_logging=True)
        val_evaluator = Evaluator(qconf, group="val", writer=None, enable_logging=True)
    else:
        from qmodel.core.cpu_evaluator import CpuEvaluator

        train_evaluator = CpuEvaluator(qconf, group="train", writer=None, enable_logging=True)
        val_evaluator = CpuEvaluator(qconf, group="val", writer=None, enable_logging=True)

    # Evaluate each checkpoint on train and val and keep the scalar summaries.
    rows: list[dict[str, float]] = []
    for ckpt_it in list(ckpt_iters):
        train_metrics = train_evaluator.eval_single(int(ckpt_it), n_iter=0, namespace="train_eval")
        val_metrics = val_evaluator.eval_single(int(ckpt_it), n_iter=0, namespace="val_eval")
        rows.append(
            {
                "iter": int(ckpt_it),
                "train_mse": float(train_metrics["train_eval/objective/mse"]),
                "train_ic": float(train_metrics["train_eval/quality/global_ic"]),
                "val_mse": float(val_metrics["val_eval/objective/mse"]),
                "val_ic": float(val_metrics["val_eval/quality/global_ic"]),
            }
        )

    # Close both evaluators after all checkpoints have been processed.
    train_evaluator.close()
    val_evaluator.close()
    return rows


def _read_raw_train_loss(tb_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the non-rolling train/objective/loss scalar from TensorBoard."""
    # Load TensorBoard scalars and require the raw loss tag to exist.
    acc = EventAccumulator(tb_dir.as_posix(), size_guidance={"scalars": 0})
    acc.Reload()
    scalars = acc.Scalars("train/objective/loss")

    # Keep the last occurrence for each step so resumed writers cannot create duplicates.
    steps = np.asarray([int(s.step) for s in list(scalars)], dtype=np.int64)
    values = np.asarray([float(s.value) for s in list(scalars)], dtype=np.float64)
    keep = steps.size - 1 - np.unique(steps[::-1], return_index=True)[1]
    keep.sort()
    return steps[keep], values[keep]


def _plot_train_loss(steps: np.ndarray, values: np.ndarray, out_path: Path) -> None:
    """Plot the raw train loss curve for the tiny overfit run."""
    # Draw the raw train loss against iteration.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, values, linewidth=1.4, color="#2b6cb0")
    ax.set_title("Tiny-sample overfit test: raw train loss")
    ax.set_xlabel("iteration")
    ax.set_ylabel("train/objective/loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_checkpoint_metrics(metric_rows: list[dict[str, float]], out_path: Path) -> None:
    """Plot train and val checkpoint MSE curves in raw label space."""
    # Convert the metric rows into aligned arrays.
    ckpt_iters = np.asarray([int(row["iter"]) for row in list(metric_rows)], dtype=np.int64)
    train_mse = np.asarray([float(row["train_mse"]) for row in list(metric_rows)], dtype=np.float64)
    val_mse = np.asarray([float(row["val_mse"]) for row in list(metric_rows)], dtype=np.float64)

    # Draw the train and val raw-space MSE across checkpoints.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ckpt_iters, train_mse, marker="o", linewidth=1.4, color="#1f7a1f", label="train MSE")
    ax.plot(ckpt_iters, val_mse, marker="o", linewidth=1.4, color="#c05621", label="val MSE")
    ax.set_title("Tiny-sample overfit test: checkpoint MSE")
    ax.set_xlabel("checkpoint iteration")
    ax.set_ylabel("raw-label MSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _build_summary(
    steps: np.ndarray,
    values: np.ndarray,
    metric_rows: list[dict[str, float]],
    subset_meta: dict[str, object],
) -> dict[str, object]:
    """Build a compact YAML summary for the tiny overfit experiment."""
    # Read the first and last raw train loss points.
    first_loss = float(values[0])
    last_loss = float(values[-1])
    min_loss = float(values.min())
    min_loss_step = int(steps[int(values.argmin())])

    # Read the first and last checkpoint evaluation metrics.
    first_metrics = dict(metric_rows[0])
    last_metrics = dict(metric_rows[-1])

    # Assemble the final summary payload.
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": {
            "out_root": str(Path(TINY_OUT_ROOT).as_posix()),
            "window_size": int(WINDOW_SIZE),
            "batch_size": int(BATCH_SIZE),
            "eval_batch_size": int(EVAL_BATCH_SIZE),
            "num_iters": int(NUM_ITERS),
            "save_every": int(SAVE_EVERY),
            "learning_rate": float(LEARNING_RATE),
        },
        "subset": dict(subset_meta["tiny_subset"]),
        "train_loss": {
            "first_step": int(steps[0]),
            "first_loss": float(first_loss),
            "last_step": int(steps[-1]),
            "last_loss": float(last_loss),
            "min_loss": float(min_loss),
            "min_loss_step": int(min_loss_step),
            "drop_ratio": float((first_loss - last_loss) / first_loss),
        },
        "checkpoint_eval": {
            "first": first_metrics,
            "last": last_metrics,
            "best_train_mse": min(float(row["train_mse"]) for row in list(metric_rows)),
            "best_val_mse": min(float(row["val_mse"]) for row in list(metric_rows)),
        },
        "checkpoint_rows": list(metric_rows),
    }


def main() -> None:
    """Build the tiny subset, train the model, evaluate checkpoints, and write the report."""
    # Prepare local report and source metadata.
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

    # Build the tiny subset with the current feature set.
    source_feature_names = list(SOURCE_FEATURE_NAMES)
    keep_indices, keep_feature_names = _feature_keep_indices(source_feature_names)
    total_runs = TRAIN_RUN_COUNT + VAL_RUN_COUNT + TEST_RUN_COUNT
    all_runs = _scan_first_stock_day_runs(
        Path(SOURCE_NPZ_DIR) / str(SOURCE_TRAIN_GROUP["meta"]),
        int(SOURCE_TRAIN_GROUP["rows"]),
        int(total_runs),
    )
    grouped_runs = _group_runs(all_runs)
    group_rows = _copy_group_arrays(keep_indices, grouped_runs)
    meta_path = _write_tiny_meta(keep_feature_names, grouped_runs, group_rows)
    subset_meta = dict(yaml.safe_load(meta_path.read_text(encoding="utf-8")))

    # Train the tiny overfit model and evaluate saved checkpoints.
    qconf, ckpt_iters = _train_tiny_model(int(len(keep_feature_names)))
    metric_rows = _evaluate_checkpoints(qconf, ckpt_iters)

    # Export the two figures used by the report.
    train_loss_steps, train_loss_values = _read_raw_train_loss(Path(qconf.root_dir) / "tb")
    train_loss_png = Path(REPORT_DIR) / "tiny_overfit_train_loss.png"
    checkpoint_png = Path(REPORT_DIR) / "tiny_overfit_checkpoint_mse.png"
    _plot_train_loss(train_loss_steps, train_loss_values, train_loss_png)
    _plot_checkpoint_metrics(metric_rows, checkpoint_png)

    # Write the machine-readable summary and the self-contained report.
    summary = _build_summary(train_loss_steps, train_loss_values, metric_rows, subset_meta)
    summary_path = Path(REPORT_DIR) / "tiny_overfit_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    report_html = render_tiny_overfit_report(summary, train_loss_png, checkpoint_png)
    report_path = Path(REPORT_DIR) / "research_report_0408_tiny_overfit_self_contained.html"
    report_path.write_text(report_html, encoding="utf-8")

    # Print the key output paths for the caller.
    print(yaml.safe_dump({"summary_yaml": str(summary_path), "report_html": str(report_path)}, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
