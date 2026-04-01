"""Provide a CPU-compatible evaluator that writes the same shard artifacts as CUDA evaluator."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from qmodel.components.amp_scaler import MyScaler
from qmodel.components.checkpoint import CheckpointSaver
from qmodel.core.predict_chunks import PredictChunkWriter
from qmodel.data.dataloader import setup_eval_dataloader
from qmodel.distributed import barrier, get_ddp_state
from qmodel.logger import logger
from qmodel.metrics.builtin import DEFAULT_EVAL_METRIC_FNS
from qmodel.metrics.core import write_tensorboard
from qmodel.metrics.eval_job import metric_fns_to_specs, spawn_eval_metrics_job
from qmodel.models import build_model
from qmodel.util import merge_date_time_dataframe

from qmodel.config import QConfig


class CpuEvaluator:
    """Evaluate a qmodel model on CPU and materialize per-iteration feather shards."""

    def __init__(
        self,
        config: QConfig,
        group: str,
        writer: SummaryWriter | None,
        enable_logging: bool,
    ) -> None:
        """Initialize evaluator state and optional TensorBoard writer."""
        # Validate group and store core knobs.
        if str(group) not in ["train", "test", "val", "predict"]:
            raise RuntimeError(f"Invalid group: {group}")
        self.config = config
        self.group = str(group)
        self.enable_logging = bool(enable_logging)

        # Store distributed state for shard naming and rank0-only metric jobs.
        ddp_enabled, rank, world_size, local_rank = get_ddp_state()
        self.ddp_enabled = bool(ddp_enabled)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)

        # Prepare a checkpoint saver to load model states.
        self.checkpointer = CheckpointSaver(config.root_dir, config, device=torch.device(config.device))

        # Manage the TensorBoard writer ownership like the CUDA evaluator does.
        if (not self.enable_logging) or self.group == "predict":
            self._owns_writer = False
            self.writer = writer
        else:
            self._owns_writer = writer is None
            if writer is not None:
                self.writer = writer
            else:
                tb_metrics_dir = os.path.join(config.root_dir, "tb")
                os.makedirs(tb_metrics_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=tb_metrics_dir)

        # Track one pending CPU metrics job to keep concurrency bounded.
        self._pending_metrics_process: mp.Process | None = None
        self._pending_metrics_path: Path | None = None
        self._pending_metrics_it: int | None = None
        self._pending_metrics_namespace: str | None = None

    def close(self) -> None:
        """Close an owned TensorBoard writer if needed."""
        # Close writer only when this evaluator created it.
        if self.enable_logging and self._owns_writer and self.writer is not None:
            self.writer.close()

    def _eval_iter_dir(self, it: int) -> Path:
        """Return the output directory for one checkpoint iteration."""
        # Select base output dir by dataset group to avoid overwriting.
        base = Path(self.config.root_dir) / ("eval_val" if self.group == "val" else "eval")
        return base / f"iter_{int(it)}"

    def _run_inference(self, *, it: int, n_iter: int) -> pd.DataFrame:
        """Run CPU inference for one checkpoint and return predictions with meta columns."""
        # Build model and load checkpoint weights.
        config = self.config
        device = torch.device(config.device)
        if device.type != "cpu":
            raise RuntimeError(f"CpuEvaluator requires CPU device, got: {device}")

        amp_scaler = MyScaler(config.use_amp, config.amp_dtype)
        model = build_model(config.model_class, config.model).to(device=device, dtype=config.eval_dtype)
        ckpt_path = Path(config.root_dir) / "ckpt" / f"iter_{int(it)}.pt"
        self.checkpointer.load(ckpt_path, model, None, None, amp_scaler)
        model.eval()

        # Build dataloader and collect outputs in numpy for predictable memory use.
        dataloader = setup_eval_dataloader(config, group=self.group, shuffle=False)
        pred_chunks: list[np.ndarray] = []
        tgt_chunks: list[np.ndarray] = []
        meta_chunks: list[np.ndarray] = []

        # Iterate over batches with an optional iteration cap.
        with torch.no_grad():
            for step_idx, batch in enumerate(dataloader):
                # Stop early when n_iter is configured for partial eval.
                if int(n_iter) > 0 and int(step_idx) >= int(n_iter):
                    break

                # Move input to device and run forward pass.
                data, target, meta = batch
                out = model(data.to(device))

                # Materialize arrays on CPU for concatenation.
                pred_chunks.append(out.detach().cpu().to(dtype=torch.float32).numpy())
                tgt_chunks.append(target.detach().cpu().to(dtype=torch.float32).numpy())
                meta_chunks.append(meta.detach().cpu().to(dtype=torch.int64).numpy())

        # Concatenate chunks into a flat table with stable column order.
        pred = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.empty((0, 1), dtype=np.float32)
        tgt = np.concatenate(tgt_chunks, axis=0) if tgt_chunks else np.empty((0, 1), dtype=np.float32)
        meta = np.concatenate(meta_chunks, axis=0) if meta_chunks else np.empty((0, 3), dtype=np.int64)

        # Build a dataframe that matches the CUDA evaluator schema.
        res_df = pd.DataFrame(
            {
                "prediction": pred[:, 0] if pred.ndim == 2 and pred.shape[1] > 0 else pred.reshape(-1),
                "target": tgt[:, 0] if tgt.ndim == 2 and tgt.shape[1] > 0 else tgt.reshape(-1),
                "code": meta[:, 0] if meta.shape[1] >= 1 else np.zeros((meta.shape[0],), dtype=np.int64),
                "date": meta[:, 1] if meta.shape[1] >= 2 else np.zeros((meta.shape[0],), dtype=np.int64),
                "time": meta[:, 2] if meta.shape[1] >= 3 else np.zeros((meta.shape[0],), dtype=np.int64),
            }
        )

        # Add convenience columns used by downstream analysis.
        res_df["StockCode"] = res_df["code"].astype(int)
        res_df["DateTime"] = merge_date_time_dataframe(res_df, "date", "time")
        return res_df

    def _run_predict_inference_to_manifest(self, *, it: int, n_iter: int, iter_dir: Path) -> Path:
        """Run CPU predict inference and stream outputs to parquet chunks with a manifest."""
        # Require a single-rank predict path so row_id is globally monotonic.
        if self.ddp_enabled:
            raise RuntimeError("Streaming predict chunks require non-DDP execution.")

        # Build model and load checkpoint weights.
        config = self.config
        device = torch.device(config.device)
        if device.type != "cpu":
            raise RuntimeError(f"CpuEvaluator requires CPU device, got: {device}")

        amp_scaler = MyScaler(config.use_amp, config.amp_dtype)
        model = build_model(config.model_class, config.model).to(device=device, dtype=config.eval_dtype)
        ckpt_path = Path(config.root_dir) / "ckpt" / f"iter_{int(it)}.pt"
        self.checkpointer.load(ckpt_path, model, None, None, amp_scaler)
        model.eval()

        # Create a chunk writer rooted at the iter directory.
        chunk_row_count = int(self.config.evaluator.predict_chunk_row_count)
        writer = PredictChunkWriter(iter_dir=Path(iter_dir), it=int(it), chunk_row_count=int(chunk_row_count), group=str(self.group))

        # Iterate over batches and append directly into the writer.
        dataloader = setup_eval_dataloader(config, group=self.group, shuffle=False)
        with torch.no_grad():
            for step_idx, batch in enumerate(dataloader):
                # Stop early when n_iter is configured for partial eval.
                if int(n_iter) > 0 and int(step_idx) >= int(n_iter):
                    break

                # Run forward pass and stream the CPU outputs into parquet buffers.
                data, target, meta = batch
                out = model(data.to(device))
                writer.append(out.detach().cpu(), target.detach().cpu(), meta.detach().cpu())

        # Close the writer and return the manifest path for downstream report code.
        manifest_path = writer.close()
        logger.info("finish streaming predict chunks")
        return Path(manifest_path)

    def _finish_pending_metrics(self, *, block: bool) -> dict[str, float]:
        """Join and log one pending metrics job if it has completed."""
        # Return early when there is no pending job.
        if self._pending_metrics_process is None:
            return {}

        # Wait for completion when requested, otherwise only handle finished jobs.
        proc = self._pending_metrics_process
        if block:
            proc.join()
        elif proc.is_alive():
            return {}

        # Require successful subprocess completion to avoid silent corruption.
        if proc.exitcode != 0:
            raise RuntimeError(f"Eval metrics job failed with exit code: {proc.exitcode}")

        # Load metric results and emit them to TensorBoard.
        assert self._pending_metrics_path is not None
        assert self._pending_metrics_it is not None
        with self._pending_metrics_path.open("r", encoding="utf-8") as f:
            scalar_metrics = json.load(f)
        if not isinstance(scalar_metrics, dict):
            raise RuntimeError(f"Invalid metrics JSON payload: {type(scalar_metrics)}")
        scalar_metrics = {str(k): float(v) for k, v in scalar_metrics.items()}

        # Write scalars under the configured step.
        if scalar_metrics and self.enable_logging and self.writer is not None:
            write_tensorboard(self.writer, scalar_metrics, int(self._pending_metrics_it))
            self.writer.flush()

        # Clear pending state after successful consumption.
        self._pending_metrics_process = None
        self._pending_metrics_path = None
        self._pending_metrics_it = None
        self._pending_metrics_namespace = None
        return scalar_metrics

    def poll_pending_metrics(self) -> None:
        """Poll and log any finished background metrics job."""
        # Consume finished jobs without blocking the caller.
        self._finish_pending_metrics(block=False)

    def finish_pending_metrics(self) -> None:
        """Block until any pending background metrics job finishes."""
        # Ensure background metric results are consumed before shutdown.
        self._finish_pending_metrics(block=True)

    def eval_sharded(self, *, it: int, n_iter: int, namespace: str, wait_metrics: bool) -> dict[str, float]:
        """Run CPU inference, write shard files, and optionally compute metrics."""
        # Block on previous metrics job to keep max concurrency=1.
        if self.enable_logging and self.rank == 0 and self._pending_metrics_process is not None:
            self._finish_pending_metrics(block=True)

        # Resolve the iter output directory early so predict can write its manifest there.
        iter_dir = self._eval_iter_dir(int(it))
        iter_dir.mkdir(parents=True, exist_ok=True)
        if self.group == "predict":
            self._run_predict_inference_to_manifest(it=int(it), n_iter=int(n_iter), iter_dir=Path(iter_dir))
        else:
            # Run inference and write per-rank shard feather.
            shard_path = iter_dir / f"rank{int(self.rank)}.feather"
            res_df = self._run_inference(it=int(it), n_iter=int(n_iter))
            res_df.to_feather(shard_path.as_posix())

        # Synchronize ranks so rank0 can safely consume all shard files.
        if self.ddp_enabled:
            barrier()

        # Skip metric logging for predict runs to keep outputs clean.
        if self.group == "predict":
            if wait_metrics and self.ddp_enabled:
                barrier()
            return {}

        # Spawn a CPU job to compute metrics from all shards on rank0 only.
        scalar_metrics: dict[str, float] = {}
        if self.enable_logging and self.rank == 0:
            metric_fns = self.config.eval_metric_fns if hasattr(self.config, "eval_metric_fns") else []
            if len(metric_fns) == 0:
                metric_fns = DEFAULT_EVAL_METRIC_FNS

            metric_specs = metric_fns_to_specs(list(metric_fns))
            metrics_path = iter_dir / "metrics.json"
            proc = spawn_eval_metrics_job(
                shard_dir=iter_dir.as_posix(),
                metric_specs=metric_specs,
                namespace=str(namespace),
                it=int(it),
                group=str(self.group),
                out_path=metrics_path.as_posix(),
            )
            self._pending_metrics_process = proc
            self._pending_metrics_path = metrics_path
            self._pending_metrics_it = int(it)
            self._pending_metrics_namespace = str(namespace)

        # Optionally wait for metrics and return scalar results.
        if wait_metrics and self.enable_logging and self.rank == 0:
            scalar_metrics = self._finish_pending_metrics(block=True)
        if wait_metrics and self.ddp_enabled:
            barrier()
        return scalar_metrics

    def eval_partial(self, it: int, n_iter: int, namespace: str) -> dict[str, float]:
        """Run a partial evaluation pass and return scalar-only metrics when ready."""
        # Run sharded eval without blocking on metrics.
        return self.eval_sharded(it=int(it), n_iter=int(n_iter), namespace=str(namespace), wait_metrics=False)

    def eval_single(self, it: int, n_iter: int, namespace: str) -> dict[str, float]:
        """Evaluate one checkpoint and wait for metric computation to finish."""
        # Run sharded eval and block on metrics completion.
        return self.eval_sharded(it=int(it), n_iter=int(n_iter), namespace=str(namespace), wait_metrics=True)

    def evaluate(self) -> None:
        """Evaluate configured checkpoints and optionally log metrics."""
        # Resolve configured iteration list and ensure output directories exist.
        files = [int(it) for it in self.config.evaluator.eval_checkpoint_iter if int(it) > 0]
        out_dir = "eval_val" if self.group == "val" else "eval"
        os.makedirs(os.path.join(self.config.root_dir, out_dir), exist_ok=True)

        # Evaluate each configured checkpoint sequentially.
        for it in files:
            self.eval_single(int(it), n_iter=int(self.config.evaluator.eval_all_num_iters), namespace="eval")

        # Flush and close writer resources.
        if self.enable_logging and self.writer is not None:
            self.writer.flush()
        self.close()
