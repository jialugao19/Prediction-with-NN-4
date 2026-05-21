"""Provide built-in metric functions for train and eval."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from qmodel.metrics.core import Histogram, MetricValue


GiB = 1024.0 * 1024.0 * 1024.0

_NVML_INITIALIZED = False
_NVML_HANDLES_BY_UUID: dict[str, object] = {}


def _normalize_namespace(namespace: str) -> str:
    """Normalize a metrics namespace for stable TensorBoard naming."""
    # Map training-time ckpt eval namespace into val for clearer dashboards.
    if namespace == "ckpt_eval":
        return "val"
    return str(namespace)


def _to_gib(bytes_value: float) -> float:
    """Convert bytes into GiB."""
    # Keep conversion as a tiny helper so metric functions stay readable.
    return float(bytes_value) / GiB


def _nvml_handle_for_device(device: torch.device):
    """Resolve and cache an NVML handle for one torch CUDA device."""
    # Import pynvml lazily so non-GPU runs fail only when the metric is used.
    import pynvml

    # Initialize NVML once per process.
    global _NVML_INITIALIZED
    if not _NVML_INITIALIZED:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True

    # Resolve device UUID via torch so we bind the right GPU under CUDA_VISIBLE_DEVICES/DDP.
    props = torch.cuda.get_device_properties(device.index)
    uuid = getattr(props, "uuid", None)
    if uuid is None:
        raise RuntimeError("torch cuda device properties missing uuid; cannot bind NVML device")
    nvml_uuid = f"GPU-{uuid}"

    # Cache the handle so per-step reads are cheap.
    handle = _NVML_HANDLES_BY_UUID.get(nvml_uuid)
    if handle is None:
        handle = pynvml.nvmlDeviceGetHandleByUUID(nvml_uuid)
        _NVML_HANDLES_BY_UUID[nvml_uuid] = handle
    return handle


def train_basic_metrics(
    namespace: str,
    step: int,
    loss: float,
    loss_mean: float,
    lr: float,
    amp_scale: float,
    **kwargs: Any,
) -> dict[str, MetricValue]:
    """Compute basic training scalars for TensorBoard and console."""
    # Normalize namespace to keep tags stable across callers.
    namespace = _normalize_namespace(namespace)

    # Emit core training scalars under a stable objective/optim hierarchy.
    return {
        f"{namespace}/objective/loss": float(loss),
        f"{namespace}/objective/loss_mean": float(loss_mean),
        f"{namespace}/optim/lr": float(lr),
        f"{namespace}/optim/amp_scale": float(amp_scale),
    }


def train_timer_metrics(
    namespace: str,
    step: int,
    trainer: Any,
    **kwargs: Any,
) -> dict[str, MetricValue]:
    """Compute rolling timing metrics from the trainer timer."""
    # Normalize namespace to keep tags stable across callers.
    namespace = _normalize_namespace(namespace)

    # Read rolling means from the trainer timer.
    means = trainer.timer.means()

    # Emit timings under a dedicated time subtree.
    return {
        f"{namespace}/time/iter_ms": float(means.iter_ms),
        f"{namespace}/time/data_ms": float(means.data_ms),
        f"{namespace}/time/model_ms": float(means.model_ms),
        f"{namespace}/time/loader_cpu_ms": float(means.loader_cpu_ms),
        f"{namespace}/time/h2d_submit_ms": float(means.h2d_submit_ms),
        f"{namespace}/time/h2d_gpu_ms": float(means.h2d_gpu_ms),
        f"{namespace}/time/forward_ms": float(means.forward_ms),
        f"{namespace}/time/backward_ms": float(means.backward_ms),
        f"{namespace}/time/optimizer_ms": float(means.optimizer_ms),
        f"{namespace}/time/checkpoint_ms": float(means.checkpoint_ms),
    }


def train_gpu_nvml_metrics(
    namespace: str,
    step: int,
    config: Any,
    **kwargs: Any,
) -> dict[str, MetricValue]:
    """Compute NVML-based GPU utilization and memory metrics."""
    # Normalize namespace to keep tags stable across callers.
    namespace = _normalize_namespace(namespace)

    # Resolve NVML handle and query current GPU stats.
    device = torch.device(config.device)
    handle = _nvml_handle_for_device(device)
    import pynvml
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

    # Emit NVML readings under the gpu subtree.
    return {
        f"{namespace}/gpu/nvml_used_gib": float(_to_gib(float(mem.used))),
        f"{namespace}/gpu/nvml_total_gib": float(_to_gib(float(mem.total))),
        f"{namespace}/gpu/nvml_gpu_util_pct": float(int(util.gpu)),
        f"{namespace}/gpu/nvml_mem_util_pct": float(int(util.memory)),
    }


def train_gpu_torch_mem_metrics(
    namespace: str,
    step: int,
    config: Any,
    **kwargs: Any,
) -> dict[str, MetricValue]:
    """Compute torch CUDA allocator memory metrics."""
    # Normalize namespace to keep tags stable across callers.
    namespace = _normalize_namespace(namespace)

    # Read torch allocator stats for the configured device.
    device = torch.device(config.device)
    alloc = float(torch.cuda.memory_allocated(device))
    reserved = float(torch.cuda.memory_reserved(device))
    max_alloc = float(torch.cuda.max_memory_allocated(device))
    max_reserved = float(torch.cuda.max_memory_reserved(device))

    # Emit torch allocator stats under the gpu subtree.
    return {
        f"{namespace}/gpu/torch_alloc_gib": float(_to_gib(alloc)),
        f"{namespace}/gpu/torch_reserved_gib": float(_to_gib(reserved)),
        f"{namespace}/gpu/torch_max_alloc_gib": float(_to_gib(max_alloc)),
        f"{namespace}/gpu/torch_max_reserved_gib": float(_to_gib(max_reserved)),
    }


def train_param_grad_norm_metrics(
    namespace: str,
    step: int,
    model: torch.nn.Module,
    **kwargs: Any,
) -> dict[str, MetricValue]:
    """Compute per-parameter and per-layer norm/grad statistics."""
    # Normalize namespace to keep tags stable across callers.
    namespace = _normalize_namespace(namespace)

    # Walk parameters and compute per-parameter norms and per-layer sumsq aggregations.
    # Track weight/bias norms separately to support focused dashboards.
    weight_param_norms: list[tuple[str, float]] = []
    bias_param_norms: list[tuple[str, float]] = []
    weight_grad_norms: list[tuple[str, float]] = []
    bias_grad_norms: list[tuple[str, float]] = []

    # Track per-layer weight/bias parameter and gradient norms.
    layer_weight_param: dict[str, float] = {}
    layer_bias_param: dict[str, float] = {}
    layer_weight_grad: dict[str, float] = {}
    layer_bias_grad: dict[str, float] = {}

    for name, param in model.named_parameters():
        if param is None:
            continue

        tensor = param.detach()
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        norm = float(torch.linalg.vector_norm(tensor).item())

        layer = name.rsplit(".", 1)[0] if "." in name else "root"
        leaf = name.rsplit(".", 1)[-1]
        if "weight" in leaf:
            weight_param_norms.append((name, norm))
            layer_weight_param[layer] = layer_weight_param.get(layer, 0.0) + (norm * norm)
        if "bias" in leaf:
            bias_param_norms.append((name, norm))
            layer_bias_param[layer] = layer_bias_param.get(layer, 0.0) + (norm * norm)

        grad = param.grad
        if grad is not None:
            grad_tensor = grad.detach()
            if grad_tensor.dtype != torch.float32:
                grad_tensor = grad_tensor.to(dtype=torch.float32)
            grad_norm = float(torch.linalg.vector_norm(grad_tensor).item())
            if "weight" in leaf:
                weight_grad_norms.append((name, grad_norm))
                layer_weight_grad[layer] = layer_weight_grad.get(layer, 0.0) + (grad_norm * grad_norm)
            if "bias" in leaf:
                bias_grad_norms.append((name, grad_norm))
                layer_bias_grad[layer] = layer_bias_grad.get(layer, 0.0) + (grad_norm * grad_norm)

    # Materialize metrics dict with a single pass to keep ordering predictable.
    metrics: dict[str, MetricValue] = {}

    # Add per-layer weight/bias norms for a dedicated debug subtree.
    for layer, sumsq in layer_weight_param.items():
        metrics[f"{namespace}/norm_layer/param/weight/{layer}"] = float(sumsq) ** 0.5
    for layer, sumsq in layer_bias_param.items():
        metrics[f"{namespace}/norm_layer/param/bias/{layer}"] = float(sumsq) ** 0.5
    for layer, sumsq in layer_weight_grad.items():
        metrics[f"{namespace}/norm_layer/grad/weight/{layer}"] = float(sumsq) ** 0.5
    for layer, sumsq in layer_bias_grad.items():
        metrics[f"{namespace}/norm_layer/grad/bias/{layer}"] = float(sumsq) ** 0.5

    # Add aggregate max/min/mean summaries for quick dashboard scanning.
    if weight_param_norms:
        values = [v for _, v in weight_param_norms]
        metrics[f"{namespace}/norm/param/weight/max"] = float(max(values))
        metrics[f"{namespace}/norm/param/weight/min"] = float(min(values))
        metrics[f"{namespace}/norm/param/weight/mean"] = float(np.mean(values))
    if bias_param_norms:
        values = [v for _, v in bias_param_norms]
        metrics[f"{namespace}/norm/param/bias/max"] = float(max(values))
        metrics[f"{namespace}/norm/param/bias/min"] = float(min(values))
        metrics[f"{namespace}/norm/param/bias/mean"] = float(np.mean(values))
    if weight_grad_norms:
        values = [v for _, v in weight_grad_norms]
        metrics[f"{namespace}/norm/grad/weight/max"] = float(max(values))
        metrics[f"{namespace}/norm/grad/weight/min"] = float(min(values))
        metrics[f"{namespace}/norm/grad/weight/mean"] = float(np.mean(values))
    if bias_grad_norms:
        values = [v for _, v in bias_grad_norms]
        metrics[f"{namespace}/norm/grad/bias/max"] = float(max(values))
        metrics[f"{namespace}/norm/grad/bias/min"] = float(min(values))
        metrics[f"{namespace}/norm/grad/bias/mean"] = float(np.mean(values))

    return metrics


def eval_global_ic(
    namespace: str,
    it: int,
    pred: np.ndarray,
    target: np.ndarray,
    **kwargs: Any,
) -> dict[str, MetricValue]:
    """Compute the global IC scalar for evaluation."""
    # Normalize namespace to keep tags stable across train/eval callers.
    namespace = _normalize_namespace(namespace)

    # Compute correlation between prediction and target across the full dataframe.
    global_ic = float(np.corrcoef(pred, target)[0, 1])
    return {f"{namespace}/quality/global_ic": global_ic}


def _rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    """Compute 1-based ranks with average rank for ties."""
    # Sort values and prepare an output ranks array.
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)

    # Assign average ranks for each equal-value group in sorted order.
    n = int(values.shape[0])
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) * 0.5
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def eval_rank_ic(
    namespace: str,
    it: int,
    pred: np.ndarray,
    target: np.ndarray,
    **kwargs: Any,
) -> dict[str, MetricValue]:
    """Compute the rank IC (Spearman) scalar for evaluation."""
    # Normalize namespace to keep tags stable across train/eval callers.
    namespace = _normalize_namespace(namespace)

    # Convert pred/target into ranks and compute correlation between ranks.
    pred_rank = _rankdata_average_ties(pred)
    target_rank = _rankdata_average_ties(target)
    rank_ic = float(np.corrcoef(pred_rank, target_rank)[0, 1])

    return {f"{namespace}/quality/rank_ic": rank_ic}


def _dist_scalar_metrics(namespace: str, values: np.ndarray, base: str) -> dict[str, MetricValue]:
    """Compute distribution scalar summaries for one value array."""
    # Normalize namespace to keep tags stable across train/eval callers.
    namespace = _normalize_namespace(namespace)

    # Compute mean/std/abs_mean and key quantiles for fast monitoring.
    mean = float(values.mean(dtype=np.float64))
    std = float(values.std(dtype=np.float64))
    abs_mean = float(np.mean(np.abs(values), dtype=np.float64))
    p01, p50, p99 = [float(v) for v in np.quantile(values, [0.01, 0.5, 0.99])]

    # Emit distribution scalars under a normalized dist subtree.
    return {
        f"{namespace}/dist/mean/{base}": mean,
        f"{namespace}/dist/std/{base}": std,
        f"{namespace}/dist/abs_mean/{base}": abs_mean,
        f"{namespace}/dist/p01/{base}": p01,
        f"{namespace}/dist/p50/{base}": p50,
        f"{namespace}/dist/p99/{base}": p99,
    }


def eval_distribution_scalars(
    namespace: str,
    it: int,
    pred: np.ndarray,
    target: np.ndarray,
    **kwargs: Any,
) -> dict[str, MetricValue]:
    """Compute scalar distribution summaries for prediction and target."""
    # Normalize namespace to keep tags stable across train/eval callers.
    namespace = _normalize_namespace(namespace)

    # Emit scalar summaries for prediction and target distributions.
    metrics: dict[str, MetricValue] = {}
    metrics.update(_dist_scalar_metrics(namespace, pred, "pred"))
    metrics.update(_dist_scalar_metrics(namespace, target, "target"))

    # Emit a simple ratio diagnostic for scaling mismatch.
    metrics[f"{namespace}/dist/pred_std_over_target_std"] = float(
        pred.std(dtype=np.float64) / target.std(dtype=np.float64)
    )
    return metrics


def eval_distribution_hist(
    namespace: str,
    it: int,
    pred: np.ndarray,
    target: np.ndarray,
    **kwargs: Any,
) -> dict[str, MetricValue]:
    """Compute histogram distribution metrics for prediction and target."""
    # Normalize namespace to keep tags stable across train/eval callers.
    namespace = _normalize_namespace(namespace)

    # Emit histograms for deeper distribution inspection in TensorBoard.
    return {
        f"{namespace}/dist/hist/pred": Histogram(pred),
        f"{namespace}/dist/hist/target": Histogram(target),
    }


DEFAULT_TRAIN_METRIC_FNS = [
    train_basic_metrics,
    train_timer_metrics,
    train_gpu_nvml_metrics,
    train_gpu_torch_mem_metrics,
    train_param_grad_norm_metrics,
]
DEFAULT_EVAL_METRIC_FNS = [eval_global_ic, eval_rank_ic, eval_distribution_scalars]
