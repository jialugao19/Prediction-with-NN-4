import torch
import os
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
from pathlib import Path
import yaml

from torch.utils.tensorboard import SummaryWriter

from qmodel.logger import logger
from qmodel.distributed import barrier, get_ddp_state
from qmodel.metrics.builtin import DEFAULT_EVAL_METRIC_FNS
from qmodel.metrics.core import write_tensorboard
from qmodel.metrics.eval_job import metric_fns_to_specs, spawn_eval_metrics_job
from qmodel.metrics.perf_console import NvmlReader, format_eval_perf_line, torch_cuda_mem_snapshot
from qmodel.util import merge_date_time_dataframe
from qmodel.data.dataloader import setup_eval_dataloader
from qmodel.core.eval_trainer import AsyncInference
from qmodel.components.profiler import MyProfiler
from qmodel.components.checkpoint import CheckpointSaver
from qmodel.components.amp_scaler import MyScaler
from qmodel.models import build_model

from qmodel.config import QConfig
from qmodel.core.predict_chunks import PredictChunkWriter


class Inferencer(AsyncInference):
    def _post_step(self, res, buffer, target, other_meta, curr_it):
        res.append([buffer, target, other_meta])
        if len(res) >= 100:
            torch.cuda.synchronize()
            res2 = [torch.cat([r[i] for r in res]) for i in range(len(res[0]))]
            res.clear()
            res.append(res2)

    def _pre_step(self, curr_it: int):
        pass


class Evaluator:
    def __init__(
        self,
        config: QConfig,
        group: str,
        writer: SummaryWriter | None = None,
        enable_logging: bool = True,
    ):
        """Initialize an evaluator for one dataset group."""
        # Initialize evaluator for a dataset split and optional TensorBoard writer.
        assert group in ["train", "test", "val", "predict"]
        self.config = config
        self.group  = group
        self.enable_logging = enable_logging

        # Require an explicit rank logging policy to avoid silent behavior changes.
        self.console_log_all_ranks = bool(config.console_log_all_ranks)

        self.ddp_enabled, self.rank, self.world_size, self.local_rank = get_ddp_state()

        self.checkpointer = CheckpointSaver(config.root_dir, config, device=config.device)
        self._pending_metrics_process: mp.Process | None = None
        self._pending_metrics_path: Path | None = None
        self._pending_metrics_it: int | None = None
        self._pending_metrics_namespace: str | None = None

        # Disable metrics logging for predict to keep outputs clean.
        if not enable_logging or group == "predict":
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

        # Reset CUDA peak memory stats so max_* reflect this evaluator run only.
        if self.config.device.type != "cuda":
            raise RuntimeError(f"Eval perf console logging requires CUDA device, got: {self.config.device}")
        # Reset peak stats without explicit device to avoid torch builds rejecting device arguments pre-init.
        torch.cuda.reset_peak_memory_stats()

        # Initialize NVML reader once so per-checkpoint reads are cheap.
        self._nvml = NvmlReader(self.config.device)
        self._label_transform = self._load_label_transform()
        self._label_mean, self._label_std = self._load_label_zscore_params()

    def _load_label_transform(self) -> dict[str, object]:
        """Load the persisted label-transform contract from the sibling meta.yaml."""
        # Resolve the data-prep metadata path relative to the qmodel run directory.
        meta_path = Path(self.config.root_dir).parent / "artifacts" / "npz" / "meta.yaml"
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        return dict(meta.get("label_transform", {"type": "none"}))

    def _load_label_zscore_params(self) -> tuple[float, float]:
        """Load scalar label zscore parameters when label normalization is enabled."""
        # Return identity parameters when label zscore is disabled.
        if str(self._label_transform["type"]) != "pooled_zscore":
            return 0.0, 1.0

        # Read the persisted mean/std pair from the data-clean artifact.
        stats_path = Path(self.config.root_dir).parent / "artifacts" / "data_clean" / str(self._label_transform["params_path"])
        stats = yaml.safe_load(stats_path.read_text(encoding="utf-8"))
        return float(stats["label"]["mean"]), float(stats["label"]["std"])

    def _inverse_label_arrays(self, prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse-transform prediction and target arrays back to raw label scale."""
        # Return arrays unchanged when label zscore is disabled.
        if str(self._label_transform["type"]) != "pooled_zscore":
            return prediction, target

        # Load scalar zscore parameters once and map both arrays back to raw scale.
        mean = np.float32(self._label_mean)
        std = np.float32(self._label_std)
        pred_raw = prediction.astype(np.float32, copy=False) * std + mean
        target_raw = target.astype(np.float32, copy=False) * std + mean
        return pred_raw.astype(np.float32, copy=False), target_raw.astype(np.float32, copy=False)

    def _inverse_label_tensors(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse-transform prediction and target tensors back to raw label scale."""
        # Return tensors unchanged when label zscore is disabled.
        if str(self._label_transform["type"]) != "pooled_zscore":
            return prediction, target

        # Load scalar zscore parameters once and map both tensors back to raw scale.
        mean = torch.as_tensor(float(self._label_mean), dtype=prediction.dtype, device=prediction.device)
        std = torch.as_tensor(float(self._label_std), dtype=prediction.dtype, device=prediction.device)
        pred_raw = prediction * std + mean
        target_raw = target.to(dtype=prediction.dtype) * std + mean
        return pred_raw, target_raw

    def close(self) -> None:
        """Close owned TensorBoard writer if needed."""
        # Close TensorBoard writer if owned by this evaluator.
        if self.enable_logging and self._owns_writer and self.writer is not None:
            self.writer.close()

    def evaluate(self):
        """Evaluate configured checkpoints and optionally log metrics."""
        # Evaluate configured checkpoints and persist artifacts/metrics.
        files = [it for it in self.config.evaluator.eval_checkpoint_iter if it > 0]
        out_dir = "eval_val" if self.group == "val" else "eval"
        os.makedirs(os.path.join(self.config.root_dir, out_dir), exist_ok=True)

        for it in files:
            self.eval_single(it, n_iter=self.config.evaluator.eval_all_num_iters, namespace="eval")

        if self.enable_logging and self.writer is not None:
            self.writer.flush()
        self.close()

    def eval_partial(self, it: int, n_iter: int, namespace: str) -> dict[str, float]:
        """Run a partial evaluation pass for training-time checkpoint eval."""
        # Run a shorter sharded eval pass and return scalar-only metrics if immediately available.
        return self.eval_sharded(it=it, n_iter=n_iter, namespace=namespace, wait_metrics=False)

    def eval_single(self, it: int, n_iter=0, namespace: str = "eval") -> dict[str, float]:
        """Evaluate one checkpoint and return scalar-only metrics."""
        # Evaluate one checkpoint, persist artifacts, and wait for metric computation to finish.
        return self.eval_sharded(it=it, n_iter=n_iter, namespace=namespace, wait_metrics=True)


    def _eval_iter_dir(self, it: int) -> Path:
        """Return the output directory for one checkpoint iteration."""
        # Select base output dir by dataset group to avoid overwriting test artifacts with val shards.
        base = Path(self.config.root_dir) / ("eval_val" if self.group == "val" else "eval")
        return base / f"iter_{it}"


    def _run_inference(self, *, it: int, n_iter: int) -> pd.DataFrame:
        """Run inference for one checkpoint and return a dataframe of predictions/targets/meta."""
        # Build model/inferencer/dataloader for the configured group and checkpoint.
        config = self.config
        device = self.config.device
        amp_scaler = MyScaler(config.use_amp, config.amp_dtype)
        model = build_model(config.model_class, config.model).to(device=device, dtype=config.eval_dtype)
        do_profile = config.profiler.profile_section == "eval"
        profiler = MyProfiler(real_run=do_profile, config=config.profiler)
        dataloader = setup_eval_dataloader(self.config, self.group, shuffle=False)

        ckpt = Path(self.config.root_dir) / "ckpt" / f"iter_{it}.pt"
        logger.info(f"Evaluating checkpoint: {ckpt}, iter {it}, group={self.group}, rank={self.rank}/{self.world_size}")
        self.checkpointer.load(ckpt, model, None, None, None)

        inferencer = Inferencer(
            model=model,
            dataloader=dataloader,
            device=device,
            amp_scaler=amp_scaler,
            profiler=profiler,
            timer_window_size=int(self.config.log_every),
        )

        # Determine evaluation length with a strict min cap.
        if n_iter == 0:
            n_iter = len(dataloader)
        n_iter = min(int(n_iter), len(dataloader))

        # Run inference and concatenate buffered outputs into numpy arrays.
        res = inferencer.run(n_iter=n_iter, use_tqdm=self.enable_logging and self.rank == 0)
        logger.info("finish running inference")

        res2: list[np.ndarray] = []
        if len(res) > 0:
            for i in range(len(res[0])):
                res2.append(np.concatenate([r[i] for r in res], axis=0))
        logger.info("finish concat inference")

        # Build a dataframe with fixed column order expected by downstream metrics.
        cols = ["prediction", "target", "code", "date", "time"]
        col_idx = 0
        res3: dict[str, np.ndarray] = {}
        for arr in res2:
            assert len(arr.shape) == 2
            assert col_idx + arr.shape[1] <= len(cols), "no enough col labels"
            for k in range(arr.shape[1]):
                res3[cols[col_idx]] = arr[:, k]
                col_idx += 1
        if "prediction" in res3 and "target" in res3:
            res3["prediction"], res3["target"] = self._inverse_label_arrays(res3["prediction"], res3["target"])
        res_df = pd.DataFrame(res3)
        logger.info("finish building dataframe")

        # Convert date/time integer columns into a datetime column for analysis.
        res_df["StockCode"] = res_df["code"].astype(int)
        res_df["DateTime"] = merge_date_time_dataframe(res_df, "date", "time")
        logger.info("finish converting to datetime")

        # Emit a perf-focused console summary line only for full eval_single runs.
        if self.rank == 0 or self.console_log_all_ranks:
            means = inferencer.timer.means()
            nvml = self._nvml.snapshot()
            torch_mem = torch_cuda_mem_snapshot(device)
            line = format_eval_perf_line(
                it=int(it),
                group=str(self.group),
                data_ms=float(means.data_ms),
                model_ms=float(means.model_ms),
                rank=int(self.rank),
                nvml=nvml,
                torch_mem=torch_mem,
            )
            logger.info(line)

        return res_df


    def _run_predict_inference_to_manifest(self, *, it: int, n_iter: int, iter_dir: Path) -> Path:
        """Run predict inference and stream outputs to parquet chunks with a manifest."""
        # Require a single-rank predict path so row_id is globally monotonic.
        if self.ddp_enabled:
            raise RuntimeError("Streaming predict chunks require non-DDP execution.")

        # Build model/inferencer/dataloader for the predict group and checkpoint.
        config = self.config
        device = self.config.device
        amp_scaler = MyScaler(config.use_amp, config.amp_dtype)
        model = build_model(config.model_class, config.model).to(device=device, dtype=config.eval_dtype)
        do_profile = config.profiler.profile_section == "eval"
        profiler = MyProfiler(real_run=do_profile, config=config.profiler)
        dataloader = setup_eval_dataloader(self.config, self.group, shuffle=False)

        # Load model weights from checkpoint and create a chunk writer.
        ckpt = Path(self.config.root_dir) / "ckpt" / f"iter_{it}.pt"
        logger.info(f"Evaluating checkpoint: {ckpt}, iter {it}, group={self.group}, rank={self.rank}/{self.world_size}")
        self.checkpointer.load(ckpt, model, None, None, None)
        chunk_row_count = int(self.config.evaluator.predict_chunk_row_count)
        writer = PredictChunkWriter(iter_dir=Path(iter_dir), it=int(it), chunk_row_count=int(chunk_row_count), group=str(self.group))

        # Define an inferencer that writes each batch into the chunk writer.
        class _PredictInferencer(AsyncInference):
            def __init__(self, **kwargs):
                """Initialize the inferencer with an external chunk writer."""
                # Store writer so post-step can flush batches to disk.
                self._writer = kwargs.pop("writer")
                super().__init__(**kwargs)

            def _pre_step(self, curr_it: int):
                """Run no-op per-step hooks for predict mode."""
                # Keep this hook explicit to match AsyncInference protocol.
                return None

            def _post_step(self, res, buffer, target, other_meta, curr_it):
                """Write one batch into parquet chunk buffers."""
                # Append the batch into the streaming writer immediately.
                pred_raw, target_raw = self_outer._inverse_label_tensors(buffer, target)
                self._writer.append(pred_raw, target_raw, other_meta)

        # Capture the outer evaluator so the nested inferencer can reuse label inverse-transform logic.
        self_outer = self

        inferencer = _PredictInferencer(
            model=model,
            dataloader=dataloader,
            device=device,
            amp_scaler=amp_scaler,
            profiler=profiler,
            timer_window_size=int(self.config.log_every),
            writer=writer,
        )

        # Determine evaluation length with a strict min cap.
        if n_iter == 0:
            n_iter = len(dataloader)
        n_iter = min(int(n_iter), len(dataloader))

        # Run inference; post-step writes chunks asynchronously and flushes before return.
        _ = inferencer.run(n_iter=n_iter, use_tqdm=self.enable_logging and self.rank == 0)
        logger.info("finish running inference")
        manifest_path = writer.close()
        logger.info("finish streaming predict chunks")

        # Emit a perf-focused console summary line for predict as well.
        if self.rank == 0 or self.console_log_all_ranks:
            means = inferencer.timer.means()
            nvml = self._nvml.snapshot()
            torch_mem = torch_cuda_mem_snapshot(device)
            line = format_eval_perf_line(
                it=int(it),
                group=str(self.group),
                data_ms=float(means.data_ms),
                model_ms=float(means.model_ms),
                rank=int(self.rank),
                nvml=nvml,
                torch_mem=torch_mem,
            )
            logger.info(line)

        return Path(manifest_path)


    def _finish_pending_metrics(self, *, block: bool) -> dict[str, float]:
        """Join and log one pending metrics job if it has completed."""
        # Return early when there is no pending metrics job.
        if self._pending_metrics_process is None:
            return {}

        # Wait for the job to finish when requested, otherwise only handle completed jobs.
        proc = self._pending_metrics_process
        if block:
            proc.join()
        elif proc.is_alive():
            return {}

        # Require successful subprocess completion to avoid silent metric corruption.
        if proc.exitcode != 0:
            raise RuntimeError(f"Eval metrics job failed with exit code: {proc.exitcode}")

        # Load metric results and emit them to TensorBoard and console.
        assert self._pending_metrics_path is not None
        assert self._pending_metrics_it is not None
        assert self._pending_metrics_namespace is not None
        with self._pending_metrics_path.open("r", encoding="utf-8") as f:
            scalar_metrics = json.load(f)
        if not isinstance(scalar_metrics, dict):
            raise RuntimeError(f"Invalid metrics JSON payload: {type(scalar_metrics)}")
        scalar_metrics = {str(k): float(v) for k, v in scalar_metrics.items()}

        if scalar_metrics and self.enable_logging:
            assert self.writer is not None
            write_tensorboard(self.writer, scalar_metrics, self._pending_metrics_it)
            self.writer.flush()

        # Clear pending state after successful consumption.
        self._pending_metrics_process = None
        self._pending_metrics_path = None
        self._pending_metrics_it = None
        self._pending_metrics_namespace = None

        return scalar_metrics


    def poll_pending_metrics(self) -> None:
        """Poll and log any finished background metrics job."""
        # Consume finished background metrics jobs without blocking the caller.
        self._finish_pending_metrics(block=False)

    def finish_pending_metrics(self) -> None:
        """Block until any pending background metrics job finishes."""
        # Ensure all background metric results are consumed before shutdown.
        self._finish_pending_metrics(block=True)


    def eval_sharded(self, *, it: int, n_iter: int, namespace: str, wait_metrics: bool) -> dict[str, float]:
        """Run a sharded eval pass and optionally wait for metrics completion."""
        # Block on any previous metrics job when starting a new one to enforce max concurrency=1.
        if self.enable_logging and self.rank == 0 and self._pending_metrics_process is not None:
            self._finish_pending_metrics(block=True)

        # Resolve the iter output directory early so predict can write its manifest there.
        iter_dir = self._eval_iter_dir(it)
        iter_dir.mkdir(parents=True, exist_ok=True)
        if self.group == "predict":
            self._run_predict_inference_to_manifest(it=int(it), n_iter=int(n_iter), iter_dir=Path(iter_dir))
        else:
            # Run inference and write a per-rank shard feather.
            shard_path = iter_dir / f"rank{self.rank}.feather"
            res_df = self._run_inference(it=int(it), n_iter=int(n_iter))
            res_df.to_feather(shard_path.as_posix())

        # Synchronize ranks so rank0 can safely consume all shard files.
        if self.ddp_enabled:
            barrier()

        # Skip metric logging for predict runs to reduce noise and overhead.
        if self.group == "predict":
            if wait_metrics and self.ddp_enabled:
                barrier()
            return {}

        # Spawn a CPU job to compute metrics from all shards on rank0 only.
        scalar_metrics: dict[str, float] = {}
        if self.enable_logging and self.rank == 0:
            # Resolve required metric function list from config.
            if not hasattr(self.config, "eval_metric_fns"):
                raise RuntimeError("Missing required config field: eval_metric_fns")
            metric_fns = self.config.eval_metric_fns
            if len(metric_fns) == 0:
                metric_fns = DEFAULT_EVAL_METRIC_FNS

            # Spawn a CPU job to compute metrics from all shards.
            metric_specs = metric_fns_to_specs(list(metric_fns))
            metrics_path = iter_dir / "metrics.json"
            proc = spawn_eval_metrics_job(
                shard_dir=iter_dir.as_posix(),
                metric_specs=metric_specs,
                namespace=namespace,
                it=it,
                group=self.group,
                out_path=metrics_path.as_posix(),
            )
            self._pending_metrics_process = proc
            self._pending_metrics_path = metrics_path
            self._pending_metrics_it = int(it)
            self._pending_metrics_namespace = str(namespace)

            # Wait for metrics when requested (full evaluation), otherwise return immediately.
            if wait_metrics:
                scalar_metrics = self._finish_pending_metrics(block=True)

        # Synchronize ranks at the end for full evaluation to avoid barrier mismatches.
        if wait_metrics and self.ddp_enabled:
            barrier()

        return scalar_metrics


    def eval_metrics(self, df):
        pass
        # todo: include this part
        # res = []

        # df1 = pl.from_pandas(df)
        # assert df1.select(pl.col(["code", "date", "time"])).is_duplicated().max() == False
        # df2 = pl.scan_ipc("tmp_data/test.feather", memory_map=False).with_columns(pl.col("date") % 1000000).collect()
        # ret_col = self.config.ret_col_name

        # # df3 = pd.merge(df, df2, on=["code", "date", "time"], validate="1:1")
        # df3 = df1.join(df2, on=["code", "date", "time"], how="inner").to_pandas()

        # # todo: check uniqueness of df ["code", "date", "time"]: wrong eval?

        # # todo: accelerate below, too slow due to large feather
        # for d, grp in df3.groupby("date"):
        #     ic = grp["prediction"].corr(grp["target"])
        #     sic = grp["prediction"].corr(grp["target"], method="spearman")
        #     top = grp.nlargest(50, 'prediction')[ret_col].mean() * 10000
        #     res.append([d, ic, sic, top])
        # df4 = pd.DataFrame(res, columns=["date", "ic", "sic", "top50ret"])

        # ic = df["prediction"].corr(df["target"])
        # sic = df["prediction"].corr(df["target"], method="spearman")

        # loss = self.config.criterion(torch.from_numpy(df.prediction.values), torch.from_numpy(df.target.values)) / len(df)

        # return {
        #     "loss": loss.item(),
        #     "ic": ic,
        #     "sic": sic,
        #     "daily_ic": df4["ic"].mean(),
        #     "daily_sic": df4["sic"].mean(),
        #     "topret": df4["top50ret"].mean(),
        # }
