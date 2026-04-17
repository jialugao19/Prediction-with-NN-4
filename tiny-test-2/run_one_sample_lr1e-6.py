"""Run a one-sample-point training experiment (lr=1e-6) and export the loss curve."""

from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml


# Bootstrap the repo root so direct script execution can import project modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prediction_nn2.pipeline import _build_qmodel_config, _default_config, _export_train_loss_curve


# Pin the allocator setting before qmodel touches CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


SOURCE_OUT_ROOT = Path("/data-cache/nn/upgrade_20260328_gru_seq60_h10/date_ranges")
SOURCE_NPZ_DIR = SOURCE_OUT_ROOT / "artifacts" / "npz"

EXP_ROOT = Path("/data-cache/nn/tiny-test-2/one_sample_lr1e-6")
REPORT_DIR = Path("/home/maomao/prediction-NN-2/report/0409")

WINDOW_SIZE = 60
LEARNING_RATE = 1e-6
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
NUM_ITERS = 20000
SAVE_EVERY = 1000


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


def _feature_keep_indices(feature_names: list[str]) -> tuple[list[int], list[str]]:
    """Keep the current feature set by removing the four *_is_zero columns."""
    # Build kept feature indices from the known feature order.
    keep_indices = [int(i) for i, name in enumerate(list(feature_names)) if not str(name).endswith("_is_zero")]

    # Materialize kept feature names for meta.yaml readability.
    keep_names = [str(feature_names[i]) for i in list(keep_indices)]
    return keep_indices, keep_names


def _prepare_one_sample_npz(*, out_npz_dir: Path, window_size: int) -> dict[str, object]:
    """Materialize a one-window dataset under artifacts/npz for qmodel."""
    # Recreate the output directory so every run is clean and deterministic.
    if Path(out_npz_dir).exists():
        shutil.rmtree(Path(out_npz_dir))
    Path(out_npz_dir).mkdir(parents=True, exist_ok=True)

    # Memory-map the source training arrays from the upstream pipeline output.
    src_meta_path = Path(SOURCE_NPZ_DIR) / "train_meta.i64"
    src_x_path = Path(SOURCE_NPZ_DIR) / "train_x.f32"
    src_y_path = Path(SOURCE_NPZ_DIR) / "train_y.f32"
    src_meta_arr = np.memmap(src_meta_path, mode="r", dtype=np.int64).reshape(-1, 3)
    src_x_arr = np.memmap(src_x_path, mode="r", dtype=np.float32).reshape(int(src_meta_arr.shape[0]), -1)
    src_y_arr = np.memmap(src_y_path, mode="r", dtype=np.float32).reshape(int(src_meta_arr.shape[0]), 1)

    # Select the first valid contiguous window end index from the precomputed cache.
    valid_end_path = Path(SOURCE_NPZ_DIR) / f"train_window{int(window_size)}_valid_end.i32"
    if not valid_end_path.exists():
        raise RuntimeError(f"Missing valid-end cache: {valid_end_path}")
    end = int(np.fromfile(valid_end_path, dtype=np.int32, count=1)[0])
    start = int(end - int(window_size) + 1)

    # Slice one window worth of rows and drop the *_is_zero features.
    keep_indices, keep_feature_names = _feature_keep_indices(list(SOURCE_FEATURE_NAMES))
    x_win = np.ascontiguousarray(src_x_arr[int(start) : int(end) + 1, :][:, keep_indices], dtype=np.float32)
    y_win = np.ascontiguousarray(src_y_arr[int(start) : int(end) + 1, :], dtype=np.float32)
    meta_win = np.ascontiguousarray(src_meta_arr[int(start) : int(end) + 1, :], dtype=np.int64)

    # Write train/val/test binaries so evaluator paths cannot break even if enabled later.
    for group in ["train", "val", "test"]:
        # Write x/y/meta with qmodel's expected naming convention.
        x_win.tofile(Path(out_npz_dir) / f"{group}_x.f32")
        y_win.tofile(Path(out_npz_dir) / f"{group}_y.f32")
        meta_win.tofile(Path(out_npz_dir) / f"{group}_meta.i64")

    # Write a minimal meta.yaml that is sufficient for Stock1mNpzDataset.
    meta_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_names": list(keep_feature_names),
        "storage": {
            "version": 1,
            "dtype": "f32",
            "groups": {
                group: {
                    "rows": int(window_size),
                    "feature_dim": int(x_win.shape[1]),
                    "x": f"{group}_x.f32",
                    "y": f"{group}_y.f32",
                    "meta": f"{group}_meta.i64",
                }
                for group in ["train", "val", "test"]
            },
        },
        "tiny_subset": {
            "source": {
                "npz_dir": str(Path(SOURCE_NPZ_DIR).as_posix()),
                "train_meta": "train_meta.i64",
            },
            "selected_window": {
                "window_size": int(window_size),
                "row_start": int(start),
                "row_end": int(end),
                "code": int(meta_win[0, 0]),
                "date": int(meta_win[0, 1]),
                "time_start": int(meta_win[0, 2]),
                "time_end": int(meta_win[-1, 2]),
            },
        },
    }
    Path(out_npz_dir / "meta.yaml").write_text(yaml.safe_dump(meta_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Return the prepared meta to support reporting.
    return dict(meta_payload)


def _train_one_sample(*, run_root: Path, feature_dim: int) -> object:
    """Train the current NN model on the one-sample dataset and return qconf."""
    # Build a dedicated pipeline config for the one-sample experiment.
    cfg = _default_config()
    cfg = replace(
        cfg,
        pipeline_mode="train_full",
        root_dir=Path("outputs") / "tiny-test-2" / "one_sample_lr1e-6",
        batch_size=int(BATCH_SIZE),
        eval_batch_size=int(EVAL_BATCH_SIZE),
        num_iters=int(NUM_ITERS),
        save_every=int(SAVE_EVERY),
        eval_every=int(SAVE_EVERY),
        learning_rate=float(LEARNING_RATE),
        num_workers=0,
        input_window_size=int(WINDOW_SIZE),
    )

    # Build qmodel config on top of the prepared run_root and make logging dense.
    qconf = _build_qmodel_config(cfg, feature_dim=int(feature_dim), run_root=Path(run_root))
    qconf.log_every = 1
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
    return qconf


def _export_loss_curve_png(*, tb_dir: Path, out_png: Path) -> None:
    """Export the mean training loss curve PNG from TensorBoard event files."""
    # Ensure the output directory exists before exporting.
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    # Export the curve using the shared pipeline helper (tag: train/objective/loss_mean).
    _export_train_loss_curve(Path(tb_dir), Path(out_png))


def main() -> None:
    """Build a one-sample dataset, train with lr=1e-6, and write the loss curve PNG."""
    # Prepare local report directory.
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

    # Prepare the one-sample artifacts/npz dataset under the experiment root.
    exp_root = Path(EXP_ROOT)
    if exp_root.exists():
        shutil.rmtree(exp_root)
    (exp_root / "artifacts" / "npz").mkdir(parents=True, exist_ok=True)
    meta = _prepare_one_sample_npz(out_npz_dir=exp_root / "artifacts" / "npz", window_size=int(WINDOW_SIZE))
    feature_dim = int(len(list(meta["feature_names"])))

    # Train the model and export a loss curve plot from TensorBoard events.
    qconf = _train_one_sample(run_root=exp_root, feature_dim=int(feature_dim))
    loss_png = Path(REPORT_DIR) / "tiny_test_2_one_sample_lr1e-6_train_loss.png"
    _export_loss_curve_png(tb_dir=Path(qconf.root_dir) / "tb", out_png=loss_png)

    # Print the key output paths for the caller.
    print(
        yaml.safe_dump(
            {
                "run_root": str(exp_root.as_posix()),
                "tb_dir": str((Path(qconf.root_dir) / "tb").as_posix()),
                "loss_png": str(loss_png.as_posix()),
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )


if __name__ == "__main__":
    main()
