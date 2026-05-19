"""Shared helpers for leakage sanity checks that live outside the main package."""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Bootstrap the repo root so direct script execution can import project modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SOURCE_OUT_ROOT = Path("/data-cache/nn/0428/date_ranges")
SOURCE_NPZ_DIR = SOURCE_OUT_ROOT / "artifacts" / "npz"
SOURCE_DATA_CLEAN_DIR = SOURCE_OUT_ROOT / "artifacts" / "data_clean"
EXP_BASE_DIR = Path("/data-cache/nn/tiny-test-3")
WINDOW_SIZE = 60
RUN_ROWS = 220
TRAIN_RUN_COUNT = 24
VAL_RUN_COUNT = 8
TEST_RUN_COUNT = 24
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
NUM_ITERS = 31
SAVE_EVERY = 30
LEARNING_RATE = 1e-3
FEATURE_NAMES = [
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
    "minute_norm",
    "session_minute_norm",
]
SOURCE_GROUPS = {
    "train": {"rows": 768564829, "feature_dim": 14, "x": "train_x.f32", "y": "train_y.f32", "meta": "train_meta.i64"},
    "val": {"rows": 43553840, "feature_dim": 14, "x": "val_x.f32", "y": "val_y.f32", "meta": "val_meta.i64"},
    "test": {"rows": 241788800, "feature_dim": 14, "x": "test_x.f32", "y": "test_y.f32", "meta": "test_meta.i64"},
}
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
        "ret_signed_log1p": {"enabled": True, "features": ["ret_1m", "ret_2m", "ret_3m", "ret_5m", "ret_10m"]},
        "heavy_tanh": {"enabled": True, "c": 3.0, "features": ["hl", "oc", "vol_10m"]},
        "feature_warmup_minutes": 10,
    },
    "stock_norm": {"type": "pooled_zscore", "scope": "train_only", "params_path": "pooled_zscore.yaml"},
    "time_features": ["minute_norm", "session_minute_norm"],
}
SOURCE_LABEL = {"type": "forward_log_return", "horizon_minutes": 10}
SOURCE_LABEL_TRANSFORM = {"type": "pooled_zscore", "scope": "train_only", "params_path": "label_zscore.yaml"}


@dataclass(frozen=True)
class PreparedDataset:
    """Describe one materialized tiny dataset."""

    exp_root: Path
    feature_dim: int
    train_rows: int
    val_rows: int
    test_rows: int
    label_mode: str


@dataclass(frozen=True)
class BaselineSpec:
    """Describe one baseline score formula."""

    name: str
    weights: dict[str, float]


def load_source_meta() -> dict[str, object]:
    """Load the source NPZ metadata."""
    # Return only the small schema needed by tiny-test-3.
    return {
        "prep_config": dict(SOURCE_PREP_CONFIG),
        "feature_transform": dict(SOURCE_FEATURE_TRANSFORM),
        "label": dict(SOURCE_LABEL),
        "label_transform": dict(SOURCE_LABEL_TRANSFORM),
        "storage": {"groups": dict(SOURCE_GROUPS)},
    }


def scan_stock_day_runs(group: str, need_runs: int) -> list[dict[str, int]]:
    """Scan source meta and return complete stock-day runs from one group."""
    # Resolve source group metadata and open its meta array.
    meta = load_source_meta()
    group_meta = dict(dict(meta["storage"])["groups"][str(group)])
    rows = int(group_meta["rows"])
    meta_arr = np.memmap(Path(SOURCE_NPZ_DIR) / str(group_meta["meta"]), mode="r", dtype=np.int64, shape=(int(rows), 3))

    # Read only a small initial block; the tests need the first few stock-day runs, not the full date.
    scan_rows = min(int(rows), 100_000)
    block = np.asarray(meta_arr[: int(scan_rows)], dtype=np.int64)
    first_date = int(block[0, 1])
    first_date_rows = np.where(block[:, 1] == int(first_date))[0]
    if int(first_date_rows.shape[0]) == 0:
        raise RuntimeError(f"Initial scan block did not contain rows for the first date in group={group}")

    # Vectorize code-run boundaries inside the first date.
    date_stop = int(first_date_rows[-1]) + 1
    first_day = block[: int(date_stop)]
    code_change = np.where(first_day[1:, 0] != first_day[:-1, 0])[0] + 1
    starts = np.concatenate([np.asarray([0], dtype=np.int64), code_change.astype(np.int64)])
    stops = np.concatenate([code_change.astype(np.int64), np.asarray([int(date_stop)], dtype=np.int64)])

    # Keep complete stock-day runs in the original order.
    runs: list[dict[str, int]] = []
    for start, stop in zip(starts, stops):
        run = _make_run(first_day, int(start), int(stop))
        if int(run["row_count"]) == int(RUN_ROWS):
            runs.append(run)
        if len(runs) >= int(need_runs):
            break

    # Require enough complete stock-day runs for cross-sectional IC tests.
    if len(runs) != int(need_runs):
        raise RuntimeError(f"Expected {need_runs} complete runs for {group}, got {len(runs)}")
    return runs


def _make_run(meta_arr: np.ndarray, row_start: int, row_stop: int) -> dict[str, int]:
    """Build one run descriptor from row boundaries."""
    # Read run identity and timing from its first and last row.
    return {
        "row_start": int(row_start),
        "row_stop": int(row_stop),
        "row_count": int(row_stop - row_start),
        "code": int(meta_arr[int(row_start), 0]),
        "date": int(meta_arr[int(row_start), 1]),
        "time_start": int(meta_arr[int(row_start), 2]),
        "time_stop": int(meta_arr[int(row_stop) - 1, 2]),
    }


def prepare_tiny_dataset(label_mode: str, exp_name: str) -> PreparedDataset:
    """Materialize one tiny train/val/test dataset with a requested label perturbation."""
    # Recreate the experiment tree so repeated runs are deterministic.
    exp_root = Path(EXP_BASE_DIR) / str(exp_name)
    if exp_root.exists():
        shutil.rmtree(exp_root)
    npz_dir = exp_root / "artifacts" / "npz"
    data_clean_dir = exp_root / "artifacts" / "data_clean"
    npz_dir.mkdir(parents=True, exist_ok=True)
    data_clean_dir.mkdir(parents=True, exist_ok=True)

    # Copy normalization sidecars so evaluator inverse-label logic is unchanged.
    shutil.copyfile(SOURCE_DATA_CLEAN_DIR / "label_zscore.yaml", data_clean_dir / "label_zscore.yaml")
    shutil.copyfile(SOURCE_DATA_CLEAN_DIR / "pooled_zscore.yaml", data_clean_dir / "pooled_zscore.yaml")

    # Build split run plans from source train/val/test groups.
    print(f"[tiny-test-3] scan source runs label_mode={label_mode}", flush=True)
    split_runs = {
        "train": scan_stock_day_runs("train", int(TRAIN_RUN_COUNT)),
        "val": scan_stock_day_runs("val", int(VAL_RUN_COUNT)),
        "test": scan_stock_day_runs("test", int(TEST_RUN_COUNT)),
    }

    # Copy arrays into local raw-bin files and apply the label perturbation.
    print(f"[tiny-test-3] materialize tiny arrays label_mode={label_mode}", flush=True)
    storage_groups: dict[str, dict[str, object]] = {}
    for group_name, runs in dict(split_runs).items():
        group_arrays = _load_group_rows(str(group_name), list(runs))
        y_group = _perturb_labels(str(label_mode), str(group_name), group_arrays["y"], list(runs))
        _write_group_files(npz_dir, str(group_name), group_arrays["x"], y_group, group_arrays["meta"])
        storage_groups[str(group_name)] = {
            "rows": int(group_arrays["x"].shape[0]),
            "feature_dim": int(group_arrays["x"].shape[1]),
            "x": f"{group_name}_x.f32",
            "y": f"{group_name}_y.f32",
            "meta": f"{group_name}_meta.i64",
        }

    # Persist a compact meta.yaml compatible with Stock1mNpzDataset and evaluator.
    meta_payload = _build_meta_payload(str(label_mode), dict(storage_groups), dict(split_runs))
    (npz_dir / "meta.yaml").write_text(yaml.safe_dump(meta_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Return the dataset descriptor used by the training helpers.
    return PreparedDataset(
        exp_root=exp_root,
        feature_dim=int(len(FEATURE_NAMES)),
        train_rows=int(storage_groups["train"]["rows"]),
        val_rows=int(storage_groups["val"]["rows"]),
        test_rows=int(storage_groups["test"]["rows"]),
        label_mode=str(label_mode),
    )


def _load_group_rows(group_name: str, runs: list[dict[str, int]]) -> dict[str, np.ndarray]:
    """Load selected row spans from one source split."""
    # Resolve source binary files for this group.
    meta = load_source_meta()
    group_meta = dict(dict(meta["storage"])["groups"][str(group_name)])
    rows = int(group_meta["rows"])
    feature_dim = int(group_meta["feature_dim"])
    x_src = np.memmap(Path(SOURCE_NPZ_DIR) / str(group_meta["x"]), mode="r", dtype=np.float32, shape=(int(rows), int(feature_dim)))
    y_src = np.memmap(Path(SOURCE_NPZ_DIR) / str(group_meta["y"]), mode="r", dtype=np.float32, shape=(int(rows), 1))
    meta_src = np.memmap(Path(SOURCE_NPZ_DIR) / str(group_meta["meta"]), mode="r", dtype=np.int64, shape=(int(rows), 3))

    # Concatenate requested spans without changing row order.
    spans = [slice(int(run["row_start"]), int(run["row_stop"])) for run in list(runs)]
    x = np.ascontiguousarray(np.concatenate([np.asarray(x_src[span], dtype=np.float32) for span in list(spans)], axis=0))
    y = np.ascontiguousarray(np.concatenate([np.asarray(y_src[span], dtype=np.float32) for span in list(spans)], axis=0))
    m = np.ascontiguousarray(np.concatenate([np.asarray(meta_src[span], dtype=np.int64) for span in list(spans)], axis=0))
    return {"x": x, "y": y, "meta": m}


def _perturb_labels(label_mode: str, group_name: str, y: np.ndarray, runs: list[dict[str, int]]) -> np.ndarray:
    """Return labels after applying one sanity-check perturbation."""
    # Keep labels unchanged for normal and baseline experiments.
    if str(label_mode) in {"normal", "baseline"}:
        return np.ascontiguousarray(y, dtype=np.float32)

    # Shuffle only train labels for the train-label randomization test.
    if str(label_mode) == "shuffle_train":
        out = np.asarray(y, dtype=np.float32).copy()
        if str(group_name) == "train":
            rng = np.random.default_rng(20260519)
            out = out[rng.permutation(int(out.shape[0]))]
        return np.ascontiguousarray(out, dtype=np.float32)

    # Shift every split by one stock-day run to break feature-label alignment.
    if str(label_mode) == "shift_one_stock":
        run_rows = [int(run["row_count"]) for run in list(runs)]
        if len(set(run_rows)) != 1:
            raise RuntimeError(f"shift_one_stock requires equal run lengths, got {sorted(set(run_rows))}")
        out = np.roll(np.asarray(y, dtype=np.float32), shift=int(run_rows[0]), axis=0)
        return np.ascontiguousarray(out, dtype=np.float32)

    # Reject unknown perturbations loudly.
    raise RuntimeError(f"Unknown label_mode: {label_mode}")


def _write_group_files(npz_dir: Path, group_name: str, x: np.ndarray, y: np.ndarray, meta: np.ndarray) -> None:
    """Write one raw-bin group to disk."""
    # Persist arrays with qmodel's expected names and dtypes.
    np.ascontiguousarray(x, dtype=np.float32).tofile(Path(npz_dir) / f"{group_name}_x.f32")
    np.ascontiguousarray(y, dtype=np.float32).tofile(Path(npz_dir) / f"{group_name}_y.f32")
    np.ascontiguousarray(meta, dtype=np.int64).tofile(Path(npz_dir) / f"{group_name}_meta.i64")


def _build_meta_payload(label_mode: str, storage_groups: dict[str, dict[str, object]], split_runs: dict[str, list[dict[str, int]]]) -> dict[str, object]:
    """Build a compact meta.yaml payload for a tiny experiment."""
    # Keep the source preprocessing contract visible for audit.
    source_meta = load_source_meta()
    prep_config = dict(source_meta["prep_config"])
    prep_config["train_days"] = 1
    prep_config["val_days"] = 1
    prep_config["test_days"] = 1

    # Assemble the storage and date sections expected by downstream loaders.
    dates = {name: [int(runs[0]["date"])] for name, runs in dict(split_runs).items()}
    return {
        "prep_config": prep_config,
        "feature_names": list(FEATURE_NAMES),
        "storage": {"format": "raw_bin_v1", "dtype": {"x": "float32", "y": "float32", "meta": "int64"}, "groups": dict(storage_groups)},
        "feature_transform": dict(source_meta["feature_transform"]),
        "dates": dates,
        "label": dict(source_meta["label"]),
        "label_transform": dict(source_meta["label_transform"]),
        "sampling": {"sample_stocks_per_minute": 0},
        "tiny_test_3": {
            "label_mode": str(label_mode),
            "source_out_root": str(SOURCE_OUT_ROOT.as_posix()),
            "window_size": int(WINDOW_SIZE),
            "run_rows": int(RUN_ROWS),
            "runs": {
                name: [{"code": int(run["code"]), "date": int(run["date"])} for run in list(runs)]
                for name, runs in dict(split_runs).items()
            },
        },
    }


def train_and_evaluate(dataset: PreparedDataset) -> dict[str, object]:
    """Train a tiny model and evaluate final train/test IC."""
    # Import the main training helpers only for NN tests.
    import torch
    from prediction_nn2.pipeline import _build_qmodel_config, _default_config

    # Build a local qmodel config that points to the tiny experiment root.
    cfg = replace(
        _default_config(),
        batch_size=int(BATCH_SIZE),
        eval_batch_size=int(EVAL_BATCH_SIZE),
        num_iters=int(NUM_ITERS),
        save_every=int(SAVE_EVERY),
        eval_every=int(SAVE_EVERY),
        learning_rate=float(LEARNING_RATE),
        num_workers=0,
        input_window_size=int(WINDOW_SIZE),
    )
    qconf = _build_qmodel_config(cfg, feature_dim=int(dataset.feature_dim), run_root=Path(dataset.exp_root), train_rows=int(dataset.train_rows))
    qconf.log_every = 10
    qconf.mean_loss_length = 50
    qconf.load_from_iter = None

    # Train one short run from scratch.
    if torch.device(qconf.device).type == "cuda":
        from qmodel.core.trainer import Trainer

        trainer = Trainer(qconf)
    else:
        from qmodel.core.cpu_trainer import CpuTrainer

        trainer = CpuTrainer(qconf)
    trainer.train()

    # Evaluate the final checkpoint on train and test.
    final_it = int(SAVE_EVERY)
    train_metrics = _eval_group(qconf, "train", int(final_it))
    test_metrics = _eval_group(qconf, "test", int(final_it))
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": str(Path(dataset.exp_root).as_posix()),
        "label_mode": str(dataset.label_mode),
        "checkpoint_iter": int(final_it),
        "train": dict(train_metrics),
        "test": dict(test_metrics),
    }
    _write_yaml(Path(dataset.exp_root) / "summary.yaml", summary)
    return summary


def _eval_group(qconf: object, group_name: str, checkpoint_iter: int) -> dict[str, float]:
    """Evaluate one group at one checkpoint."""
    # Import torch only when evaluator dispatch needs device inspection.
    import torch

    # Use the same evaluator implementation as the main pipeline.
    namespace = f"{group_name}_eval"
    if torch.device(qconf.device).type == "cuda":
        from qmodel.core.evaluator import Evaluator

        evaluator = Evaluator(qconf, group=str(group_name), writer=None, enable_logging=True)
    else:
        from qmodel.core.cpu_evaluator import CpuEvaluator

        evaluator = CpuEvaluator(qconf, group=str(group_name), writer=None, enable_logging=True)
    metrics = evaluator.eval_single(int(checkpoint_iter), n_iter=0, namespace=str(namespace))
    evaluator.close()

    # Return a compact scalar schema.
    return {
        "mse": float(metrics[f"{namespace}/objective/mse"]),
        "global_ic": float(metrics[f"{namespace}/quality/global_ic"]),
        "rank_ic": float(metrics[f"{namespace}/quality/rank_ic"]),
    }


def run_manifest_eval(dataset: PreparedDataset, checkpoint_iter: int) -> dict[str, object]:
    """Stream final train/test manifests and compute pooled plus daily IC."""
    # Import the main pipeline-style IC helpers only for manifest tests.
    from prediction_nn2.eval_ic import daily_pearson_ic_summary_from_manifest, pooled_pearson_ic_from_manifest
    from prediction_nn2.pipeline import _build_qmodel_config, _default_config

    # Build a qmodel config for manifest prediction.
    cfg = replace(
        _default_config(),
        batch_size=int(BATCH_SIZE),
        eval_batch_size=int(EVAL_BATCH_SIZE),
        num_iters=int(NUM_ITERS),
        save_every=int(SAVE_EVERY),
        eval_every=int(SAVE_EVERY),
        num_workers=0,
        input_window_size=int(WINDOW_SIZE),
    )
    qconf = _build_qmodel_config(cfg, feature_dim=int(dataset.feature_dim), run_root=Path(dataset.exp_root), train_rows=int(dataset.train_rows))

    # Stream both groups into independent predict manifests.
    manifests = {
        "train": _run_predict_manifest(qconf, "train", int(checkpoint_iter)),
        "test": _run_predict_manifest(qconf, "test", int(checkpoint_iter)),
    }

    # Compute the same IC summaries used by the main train report.
    summary: dict[str, object] = {}
    for group_name, manifest_path in dict(manifests).items():
        daily_yaml = Path(dataset.exp_root) / f"{group_name}_daily_ic.yaml"
        summary[str(group_name)] = {
            "manifest": str(Path(manifest_path).as_posix()),
            "pooled": pooled_pearson_ic_from_manifest(Path(manifest_path)),
            "daily": daily_pearson_ic_summary_from_manifest(Path(manifest_path), Path(daily_yaml)),
        }
    _write_yaml(Path(dataset.exp_root) / "manifest_ic_summary.yaml", summary)
    return summary


def _run_predict_manifest(qconf: object, group_name: str, checkpoint_iter: int) -> Path:
    """Run one evaluator manifest pass for a group."""
    # Import torch only when evaluator dispatch needs device inspection.
    import torch

    # Remove stale destination before streaming predictions.
    iter_dir = Path(qconf.root_dir) / f"eval_{group_name}" / f"iter_{int(checkpoint_iter)}"
    if iter_dir.exists():
        shutil.rmtree(iter_dir)
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Use the evaluator's chunked prediction path directly.
    if torch.device(qconf.device).type == "cuda":
        from qmodel.core.evaluator import Evaluator

        evaluator = Evaluator(qconf, group=str(group_name), writer=None, enable_logging=False)
    else:
        from qmodel.core.cpu_evaluator import CpuEvaluator

        evaluator = CpuEvaluator(qconf, group=str(group_name), writer=None, enable_logging=False)
    manifest_path = evaluator._run_predict_inference_to_manifest(it=int(checkpoint_iter), n_iter=0, iter_dir=Path(iter_dir))
    evaluator.close()
    return Path(manifest_path)


def run_baselines(exp_name: str) -> dict[str, object]:
    """Evaluate simple feature baselines on a normal tiny dataset."""
    # Prepare an unperturbed tiny dataset as the baseline input.
    print("[tiny-test-3] prepare baseline dataset", flush=True)
    dataset = prepare_tiny_dataset("baseline", str(exp_name))
    npz_dir = Path(dataset.exp_root) / "artifacts" / "npz"
    specs = [
        BaselineSpec(name="ret_1m", weights={"ret_1m": 1.0}),
        BaselineSpec(name="ret_5m", weights={"ret_5m": 1.0}),
        BaselineSpec(name="ret_10m", weights={"ret_10m": 1.0}),
        BaselineSpec(name="ret_combo", weights={"ret_1m": 1.0, "ret_5m": 0.5, "ret_10m": 0.25}),
    ]

    # Compute pooled and daily IC for every baseline and split.
    print("[tiny-test-3] compute baseline IC", flush=True)
    rows: list[dict[str, object]] = []
    for spec in list(specs):
        for group_name in ["train", "test"]:
            rows.append(_compute_baseline_row(npz_dir, str(group_name), spec))
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": str(Path(dataset.exp_root).as_posix()),
        "rows": rows,
    }
    _write_yaml(Path(dataset.exp_root) / "baseline_summary.yaml", summary)
    pd.DataFrame(rows).to_csv(Path(dataset.exp_root) / "baseline_summary.csv", index=False)
    return summary


def _compute_baseline_row(npz_dir: Path, group_name: str, spec: BaselineSpec) -> dict[str, object]:
    """Compute one baseline IC row."""
    # Load arrays and construct the weighted score.
    meta_payload = yaml.safe_load((Path(npz_dir) / "meta.yaml").read_text(encoding="utf-8"))
    group_meta = dict(dict(meta_payload["storage"])["groups"][str(group_name)])
    rows = int(group_meta["rows"])
    feature_dim = int(group_meta["feature_dim"])
    x = np.memmap(Path(npz_dir) / str(group_meta["x"]), mode="r", dtype=np.float32, shape=(int(rows), int(feature_dim)))
    y = np.memmap(Path(npz_dir) / str(group_meta["y"]), mode="r", dtype=np.float32, shape=(int(rows), 1))
    meta = np.memmap(Path(npz_dir) / str(group_meta["meta"]), mode="r", dtype=np.int64, shape=(int(rows), 3))
    score = np.zeros((int(rows),), dtype=np.float64)
    for feature_name, weight in dict(spec.weights).items():
        score += float(weight) * x[:, int(FEATURE_NAMES.index(str(feature_name)))].astype(np.float64)

    # Compute pooled and daily cross-sectional IC.
    target = y[:, 0].astype(np.float64)
    pooled = _pearson(score, target)
    daily = _daily_cross_sectional_ic(meta, score, target)
    return {
        "baseline": str(spec.name),
        "group": str(group_name),
        "pooled_ic": float(pooled),
        "daily_ic_mean": float(daily["mean"]),
        "daily_ic_std": float(daily["std"]),
        "daily_count": int(daily["count"]),
    }


def _daily_cross_sectional_ic(meta: np.ndarray, score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute mean daily IC from per-minute cross-sectional IC values."""
    # Build a compact dataframe to group by date and time.
    df = pd.DataFrame({"date": meta[:, 1], "time": meta[:, 2], "score": score, "target": target})
    day_values: list[float] = []
    for _date, day_df in df.groupby("date", sort=False):
        minute_values: list[float] = []
        for (_d, _t), minute_df in day_df.groupby(["date", "time"], sort=False):
            minute_values.append(_pearson(minute_df["score"].to_numpy(dtype=np.float64), minute_df["target"].to_numpy(dtype=np.float64)))
        vals = np.asarray(minute_values, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if int(vals.shape[0]) > 0:
            day_values.append(float(vals.mean(dtype=np.float64)))
    arr = np.asarray(day_values, dtype=np.float64)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "count": int(arr.shape[0])}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Compute finite-pair Pearson correlation."""
    # Filter finite pairs and compute Pearson with numpy.
    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask].astype(np.float64, copy=False)
    y2 = y[mask].astype(np.float64, copy=False)
    if int(x2.shape[0]) < 2:
        return float("nan")
    return float(np.corrcoef(x2, y2)[0, 1])


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    """Write one YAML file."""
    # Ensure the parent exists and write the payload.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
