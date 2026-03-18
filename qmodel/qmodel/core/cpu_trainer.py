"""Provide a CPU-compatible trainer that mirrors the CUDA trainer interface."""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from qmodel.components.amp_scaler import MyScaler
from qmodel.components.checkpoint import CheckpointSaver
from qmodel.components.lr_scheduler import custom_scheduler
from qmodel.components.timer import RollingTimer
from qmodel.core.cpu_evaluator import CpuEvaluator
from qmodel.data.dataloader import setup_train_dataloader
from qmodel.distributed import barrier, get_ddp_state, is_main_process
from qmodel.logger import logger
from qmodel.metrics.builtin import train_basic_metrics, train_timer_metrics
from qmodel.metrics.core import run_metric_fns, write_tensorboard
from qmodel.models import build_model

from qmodel.config import QConfig


class CpuTrainer:
    """Train a qmodel model on CPU using a synchronous loop with TensorBoard logging."""

    def __init__(self, config: QConfig) -> None:
        """Initialize CPU trainer state, model, optimizer, and writer."""
        # Record config and distributed state for consistent behavior with CUDA trainer.
        self.config = config
        ddp_enabled, rank, world_size, local_rank = get_ddp_state()
        self.ddp_enabled = bool(ddp_enabled)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self.is_main_process = bool(is_main_process())

        # Build model and core training components on the configured device.
        device = torch.device(config.device)
        if device.type != "cpu":
            raise RuntimeError(f"CpuTrainer requires CPU device, got: {device}")
        self.device = device

        # Instantiate model, loss, optimizer, and scheduler.
        dtype = config.train_dtype
        self.model = build_model(config.model_class, config.model).to(device=device, dtype=dtype)
        self.loss_fn = config.criterion
        self.optimizer = config.optimizer_class(self.model.parameters(), lr=float(config.learning_rate))
        self.lr_sched = custom_scheduler(self.optimizer, config)

        # Prepare AMP scaler in a strict CPU setting (caller should configure use_amp=none).
        self.amp_scaler = MyScaler(config.use_amp, config.amp_dtype)

        # Build dataloader and iteration helpers.
        self.dataloader, self.sampler = setup_train_dataloader(config, group="train", shuffle=True)
        self._it = iter(self.dataloader)

        # Prepare directories and TensorBoard writer.
        os.makedirs(config.root_dir, exist_ok=True)
        os.makedirs(config.tensorboard_dir, exist_ok=True)
        if self.is_main_process:
            tb_metrics_dir = os.path.join(config.root_dir, "tb")
            os.makedirs(tb_metrics_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_metrics_dir)
        else:
            self.writer = None

        # Build a CPU evaluator for validation during training.
        self.evaluator = CpuEvaluator(config, group="val", writer=self.writer, enable_logging=self.is_main_process)

        # Prepare checkpoint saver and resume logic.
        self.checkpointer = CheckpointSaver(config.root_dir, config, device=device)
        self.start_iter = 0
        self._load_from_checkpoint()

        # Prepare a rolling timer for train_timer_metrics compatibility.
        self.timer = RollingTimer(int(config.log_every))

    def _load_from_checkpoint(self) -> None:
        """Resume model/optimizer state from an existing checkpoint if configured."""
        # Resolve checkpoint path from config and load if present.
        from qmodel.util import find_checkpoint_path

        root_dir = Path(self.config.root_dir)
        iteration = self.config.load_from_iter
        ckpt_path = find_checkpoint_path(root_dir, iteration)

        # Load checkpoint and align sampler position to the saved iteration.
        if ckpt_path is None:
            logger.info("No checkpoint found, starting from scratch.")
            return

        logger.info(f"Found checkpoint at {ckpt_path}, loading...")
        self.checkpointer.load(ckpt_path, self.model, self.optimizer, self.lr_sched, self.amp_scaler)
        iter_str = ckpt_path.stem.split("_")[1]
        if not iter_str.isdigit():
            raise RuntimeError(f"Failed parsing iter number from ckpt file: {ckpt_path.name}")

        iter_num = int(iter_str)
        self.sampler.set_start_iter(iter_num * int(self.config.batch_size))
        self.start_iter = int(iter_num)
        logger.info(f"Resuming from iteration {iter_num}")

    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch the next (data, target, meta) batch from the dataloader iterator."""
        # Recreate iterator on StopIteration to support finite dataloaders in CPU mode.
        try:
            data, target, meta = next(self._it)
        except StopIteration:
            self._it = iter(self.dataloader)
            data, target, meta = next(self._it)
        return data, target, meta

    def _pre_step(self, curr_it: int) -> None:
        """Handle periodic checkpointing and evaluation during training."""
        # Decide whether to save or run eval at this iteration.
        do_save = int(curr_it) % int(self.config.save_every) == 0
        do_eval = bool(self.config.eval_during) and int(curr_it) % int(self.config.eval_every) == 0
        if not (do_save or do_eval):
            return

        # Synchronize ranks for deterministic side effects.
        if self.ddp_enabled:
            barrier()

        # Main process writes checkpoints only.
        if do_save and self.is_main_process:
            self.checkpointer.save(self.model, self.optimizer, self.lr_sched, self.amp_scaler, int(curr_it))

        # Synchronize again so all ranks see the checkpoint file before evaluation.
        if self.ddp_enabled:
            barrier()

        # All ranks can run eval, but metrics/logging happens on rank0 only via enable_logging.
        if do_eval:
            assert self.evaluator is not None
            self.evaluator.eval_partial(int(curr_it), int(self.config.eval_during_num_iters), namespace="ckpt_eval")

    def train(self) -> list[float]:
        """Run the synchronous CPU training loop and close writers cleanly."""
        # Initialize rolling loss buffer to compute stable loss_mean.
        loss_window = int(self.config.mean_loss_length)
        loss_hist: deque[float] = deque(maxlen=loss_window)

        # Prepare metric function list explicitly to avoid GPU-only defaults.
        metric_fns = list(self.config.train_metric_fns) if hasattr(self.config, "train_metric_fns") else []
        if len(metric_fns) == 0:
            metric_fns = [train_basic_metrics, train_timer_metrics]

        # Iterate for the configured number of steps.
        num_iters = int(self.config.num_iters)
        final_it = int(self.start_iter) + int(num_iters)
        try:
            for curr_it in range(int(self.start_iter), int(self.start_iter) + num_iters):
                # Do checkpoint/eval side effects before consuming this step's batch.
                self._pre_step(int(curr_it))

                # Measure data fetch and model compute times for timer metrics.
                iter_t0 = time.perf_counter()
                data, target, _meta = self._next_batch()
                data_t1 = time.perf_counter()

                # Move tensors to device and run one forward/backward/update step.
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with self.amp_scaler.autocast():
                    output = self.model(data)
                    loss = self.loss_fn(output, target.to(dtype=output.dtype))
                loss.backward()
                if self.config.grad_clip_norm is not None:
                    params = [p for g in self.optimizer.param_groups for p in g["params"]]
                    torch.nn.utils.clip_grad_norm_(params, max_norm=float(self.config.grad_clip_norm))
                self.optimizer.step()
                self.lr_sched.step()
                model_t2 = time.perf_counter()

                # Update rolling timer with ms samples.
                self.timer.add(
                    iter_ms=(model_t2 - iter_t0) * 1000.0,
                    data_ms=(data_t1 - iter_t0) * 1000.0,
                    model_ms=(model_t2 - data_t1) * 1000.0,
                )

                # Update loss history and emit metrics periodically.
                loss_val = float(loss.detach().cpu().item())
                loss_hist.append(loss_val)
                if self.is_main_process and int(curr_it) % int(self.config.log_every) == 0:
                    # Build a payload compatible with qmodel metric functions.
                    loss_mean = float(np.mean(np.asarray(loss_hist, dtype=float))) if len(loss_hist) > 0 else float("nan")
                    lr = float(self.optimizer.param_groups[0]["lr"]) if not isinstance(self.optimizer.param_groups[0]["lr"], torch.Tensor) else float(self.optimizer.param_groups[0]["lr"].detach().cpu().item())
                    payload = {
                        "namespace": "train",
                        "trainer": self,
                        "config": self.config,
                        "step": int(curr_it),
                        "loss": float(loss_val),
                        "loss_mean": float(loss_mean),
                        "lr": float(lr),
                        "amp_scale": 1.0,
                        "model": self.model,
                        "optimizer": self.optimizer,
                        "lr_sched": self.lr_sched,
                        "amp_scaler": self.amp_scaler,
                        "buffer": None,
                        "target": target.detach().cpu(),
                        "other_meta": None,
                    }

                    # Run metrics and write to TensorBoard.
                    metrics = run_metric_fns(metric_fns, payload)
                    if metrics:
                        assert self.writer is not None
                        write_tensorboard(self.writer, metrics, int(curr_it))
                        self.writer.flush()

            # Persist a final checkpoint at the logical end iteration for downstream evaluation.
            if self.is_main_process:
                self.checkpointer.save(self.model, self.optimizer, self.lr_sched, self.amp_scaler, int(final_it))
        finally:
            # Ensure eval background jobs are consumed before closing writers.
            if self.writer is not None:
                assert self.evaluator is not None
                self.evaluator.finish_pending_metrics()
                self.writer.close()
            if self.ddp_enabled:
                barrier()

        # Return a list of loss values for optional downstream usage.
        return list(loss_hist)
