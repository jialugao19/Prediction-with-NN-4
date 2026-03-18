import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from torch.nn.parallel import DistributedDataParallel

from qmodel.core.eval_trainer import AsyncTrainer, SyncTrainer
from qmodel.core.evaluator import Evaluator
from qmodel.components.lr_scheduler import custom_scheduler
from qmodel.components.checkpoint import CheckpointSaver
from qmodel.components.amp_scaler import MyScaler
from qmodel.components.profiler import MyProfiler
from qmodel.data.dataloader import setup_train_dataloader
from qmodel.logger import logger
from qmodel.models import build_model
from qmodel.distributed import barrier, get_ddp_state, is_main_process

from qmodel.config import QConfig
from qmodel.metrics.builtin import DEFAULT_TRAIN_METRIC_FNS
from qmodel.metrics.core import run_metric_fns, write_tensorboard
from qmodel.metrics.perf_console import NvmlReader, format_train_perf_line, torch_cuda_mem_snapshot

from line_profiler import profile


class Trainer(AsyncTrainer):
    def __init__(self, config: QConfig) -> None:
        self.config = config

        # Require an explicit rank logging policy to avoid silent behavior changes.
        self.console_log_all_ranks = bool(config.console_log_all_ranks)

        # Load DDP state that was initialized by entry_main when WORLD_SIZE>1.
        ddp_enabled, rank, world_size, local_rank = get_ddp_state()
        self.ddp_enabled = bool(ddp_enabled)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self.is_main_process = is_main_process()

        # train parameters retrieved from config
        device = config.device
        config.device = device
        dtype  = config.train_dtype
        model     = build_model(config.model_class, config.model).to(device=device, dtype=dtype)
        if ddp_enabled:
            if device.type == "cuda":
                model = DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)
            else:
                model = DistributedDataParallel(model)
        loss_fn   = config.criterion
        lr        = config.learning_rate
        optimizer = config.optimizer_class(model.parameters(), lr=lr)
        lr_sched  = custom_scheduler(optimizer, config)
        amp_scaler  = MyScaler(config.use_amp, config.amp_dtype)

        # profiler
        do_profile = config.profiler.profile_section == "train" and (self.is_main_process or config.profiler.all_ranks)
        if do_profile and ddp_enabled and config.profiler.all_ranks:
            config.profiler.profile_dir = os.path.join(config.profiler.profile_dir, f"rank{rank}")
        profiler    = MyProfiler(real_run=do_profile, config=config.profiler)

        # dataloader
        dataloader, sampler = setup_train_dataloader(config, group="train", shuffle=True)

        super().__init__(
            model=model, dataloader=dataloader, loss_fn=loss_fn, lr_sched=lr_sched, amp_scaler=amp_scaler, profiler=profiler,
            optimizer=optimizer, device=device, timer_window_size=int(config.log_every), grad_clip_norm=config.grad_clip_norm
        )

        # Prepare local artifacts (only rank 0 writes logs/checkpoints).
        os.makedirs(config.root_dir, exist_ok=True)
        os.makedirs(config.tensorboard_dir, exist_ok=True)

        if self.is_main_process:
            tb_metrics_dir = os.path.join(config.root_dir, "tb")
            os.makedirs(tb_metrics_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_metrics_dir)
        else:
            self.writer = None
        self.evaluator = Evaluator(config, group="val", writer=self.writer, enable_logging=self.is_main_process)

        self.checkpointer = CheckpointSaver(config.root_dir, config, device)
        self.sampler = sampler
        self.start_iter = 0

        self.load_from_checkpoint()

    def load_from_checkpoint(self):
        from qmodel.util import find_checkpoint_path
        root_dir = Path(self.config.root_dir)
        iteration = self.config.load_from_iter
        ckpt_path = find_checkpoint_path(root_dir, iteration)

        # load from checkpoint and write log
        if ckpt_path is not None:
            logger.info(f"Found checkpoint at {ckpt_path}, loading...")
            self.checkpointer.load(
                ckpt_path, self.model, self.optimizer, self.lr_sched, self.amp_scaler
            )
            iter_str = ckpt_path.stem.split("_")[1]
            if iter_str.isdigit():
                iter_num = int(iter_str)
                self.sampler.set_start_iter(iter_num * self.config.batch_size)
                self.start_iter = iter_num
                logger.info(f"Resuming from iteration {iter_num}")
            else:
                logger.warning(f"Fail parsing iter number from ckpt file: {ckpt_path.name}")
        else:
            logger.info("No checkpoint found, starting from scratch.")

    def train(self):
        """Run the main training loop and handle writer shutdown."""
        assert isinstance(self.dataloader, torch.utils.data.DataLoader)

        # Reset CUDA peak memory stats so max_* reflect this run only.
        if self.config.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.config.device)

        # Initialize NVML reader once so per-step reads are cheap.
        if self.config.device.type != "cuda":
            raise RuntimeError(f"Training perf console logging requires CUDA device, got: {self.config.device}")
        self._nvml = NvmlReader(self.config.device)

        n_dl = len(self.dataloader)
        n_iter = self.config.num_iters
        if n_iter > n_dl:
            logger.warning(f"num_iters {n_iter} is greater than the number of batches {n_dl}. This is approx {n_iter / n_dl:.2f} epochs.")
        try:
            res = self.run(n_iter=n_iter, start=self.start_iter, use_tqdm=False)
            return res
        finally:
            if self.writer is not None:
                assert self.evaluator is not None
                self.evaluator.finish_pending_metrics()
                self.writer.close()
            if self.ddp_enabled:
                barrier()

    def _pre_step(self, curr_it):
        """Handle periodic checkpointing and evaluation during training."""
        # Decide whether this step triggers checkpoint save or periodic validation.
        do_save = curr_it % self.config.save_every == 0
        do_eval = self.config.eval_during and curr_it % self.config.eval_every == 0
        if not (do_save or do_eval):
            return

        # Synchronize ranks so side effects occur at the same logical step.
        if self.ddp_enabled:
            barrier()

        # Main process only: save ckpt for the current iteration.
        if do_save and self.is_main_process:
            self.checkpointer.save(
                self.model, self.optimizer, self.lr_sched,
                self.amp_scaler, curr_it
            )

        # Synchronize again so all ranks see the checkpoint file before evaluation.
        if self.ddp_enabled:
            barrier()

        # All ranks: run sharded validation on val dataset and spawn metrics on rank0.
        if do_eval:
            torch.cuda.empty_cache()
            assert self.evaluator is not None
            self.evaluator.eval_partial(curr_it, self.config.eval_during_num_iters, namespace="ckpt_eval")
            torch.cuda.empty_cache()

    def _post_step(self, res, buffer, target, other_meta, curr_it):
        """Update loss history and emit training metrics periodically."""

        loss_length = self.config.mean_loss_length
        if len(res) < loss_length:
            res.extend([np.nan for _ in range(loss_length-len(res))])

        # Update rolling loss values and decide whether to emit logs on this step.
        loss_val = buffer.item()
        res[curr_it % loss_length] = loss_val
        if self.is_main_process and curr_it % self.config.log_every == 0:
            # Poll background eval metrics so they can be logged as soon as they are ready.
            assert self.evaluator is not None
            self.evaluator.poll_pending_metrics()

            metric_fns = self.config.train_metric_fns
            if len(metric_fns) == 0:
                metric_fns = DEFAULT_TRAIN_METRIC_FNS

            # Build a full payload so user metric fns can pick what they need.
            loss_mean = float(np.nanmean(res))
            lr = float(self.optimizer.param_groups[0]["lr"])
            amp_scale = float(self.amp_scaler.get_scale())
            payload = {
                "namespace": "train",
                "trainer": self,
                "config": self.config,
                "step": curr_it,
                "loss": float(loss_val),
                "loss_mean": loss_mean,
                "lr": lr,
                "amp_scale": amp_scale,
                "model": self.model,
                "optimizer": self.optimizer,
                "lr_sched": self.lr_sched,
                "amp_scaler": self.amp_scaler,
                "buffer": buffer,
                "target": target,
                "other_meta": other_meta,
            }

            # Run metrics and emit them to both TensorBoard and console.
            metrics = run_metric_fns(metric_fns, payload)
            if metrics:
                assert self.writer is not None
                write_tensorboard(self.writer, metrics, curr_it)

        # Emit a perf-focused console summary line according to rank policy.
        do_console = (curr_it % self.config.log_every == 0) and (self.is_main_process or self.console_log_all_ranks)
        if do_console:
            # Compute rolling timing means from the shared timer.
            means = self.timer.means()

            # Read memory and utilization from NVML + torch allocator.
            nvml = self._nvml.snapshot()
            torch_mem = torch_cuda_mem_snapshot(self.config.device)

            # Format and emit a stable single-line performance summary.
            loss_mean = float(np.nanmean(res))
            line = format_train_perf_line(
                step=int(curr_it),
                loss=float(loss_val),
                loss_mean=float(loss_mean),
                data_ms=float(means.data_ms),
                model_ms=float(means.model_ms),
                rank=int(self.rank),
                nvml=nvml,
                torch_mem=torch_mem,
            )
            logger.info(line)
