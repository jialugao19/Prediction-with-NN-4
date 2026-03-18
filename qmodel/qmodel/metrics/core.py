"""Provide a small metrics execution and logging pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Collection, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


Scalar = Union[int, float, np.integer, np.floating]


@dataclass(frozen=True)
class Histogram:
    """Represent one histogram metric to be logged to TensorBoard."""

    values: np.ndarray | torch.Tensor
    max_points: int = 100_000


MetricValue = Scalar | Histogram


def run_metric_fns(
    metric_fns: Collection[Callable[..., Mapping[str, MetricValue]]],
    payload: Mapping[str, Any],
) -> dict[str, MetricValue]:
    """Run metric callables and merge their outputs into one dict."""
    # Execute each metric callable and collect their output mappings.
    merged: dict[str, MetricValue] = {}
    payload_dict = dict(payload)
    for metric_fn in metric_fns:
        out = metric_fn(**payload_dict)
        if not isinstance(out, dict):
            raise RuntimeError(f"Metric must return dict[str, MetricValue], got: {type(out)} from {metric_fn}")

        for k, v in out.items():
            if k in merged:
                raise RuntimeError(f"Duplicate metric tag: {k}")
            merged[k] = v

    return merged


def _prepare_histogram_values(values: np.ndarray | torch.Tensor, max_points: int) -> np.ndarray | torch.Tensor:
    """Prepare histogram values by flattening and downsampling to max_points."""
    # Flatten values and ensure they live on CPU for stable logging.
    if isinstance(values, torch.Tensor):
        flat = values.detach()
        if flat.is_cuda:
            flat = flat.cpu()
        flat = flat.flatten()

        n = int(flat.numel())
        if n > max_points:
            stride = max(1, (n + max_points - 1) // max_points)
            flat = flat[::stride]
        return flat

    arr = np.asarray(values).reshape(-1)
    if arr.size > max_points:
        stride = max(1, (int(arr.size) + int(max_points) - 1) // int(max_points))
        arr = arr[::stride]
    return arr


def write_tensorboard(writer: SummaryWriter, metrics: Mapping[str, MetricValue], step: int) -> None:
    """Write merged metrics into TensorBoard."""
    # Dispatch scalars and histograms to their corresponding SummaryWriter APIs.
    for tag, value in metrics.items():
        if isinstance(value, Histogram):
            sampled = _prepare_histogram_values(value.values, value.max_points)
            writer.add_histogram(tag, sampled, step)
            continue

        writer.add_scalar(tag, float(value), step)


def _format_scalar(value: Scalar) -> str:
    """Format a scalar as a compact float string for logs."""
    # Normalize scalar-like values to float for consistent formatting.
    return f"{float(value):.6g}"


def _describe_histogram(values: np.ndarray | torch.Tensor) -> str:
    """Describe histogram values with a small numeric summary string."""
    # Convert values into a NumPy array for fast reductions and quantiles.
    if isinstance(values, torch.Tensor):
        arr = values.numpy()
    else:
        arr = np.asarray(values)

    n = int(arr.size)
    mean = float(arr.mean())
    std = float(arr.std())
    p01, p50, p99 = [float(v) for v in np.quantile(arr, [0.01, 0.5, 0.99])]

    return (
        f"[hist n={n} mean={mean:.6g} std={std:.6g} "
        f"p01={p01:.6g} p50={p50:.6g} p99={p99:.6g}]"
    )


def format_metrics_for_console(metrics: Mapping[str, MetricValue], step: int) -> str:
    """Format metrics as a single-line string for console logging."""
    # Produce a stable single-line `key=value` representation, with histogram summaries.
    parts: list[str] = [f"step={step}"]
    for tag, value in metrics.items():
        if isinstance(value, Histogram):
            sampled = _prepare_histogram_values(value.values, value.max_points)
            parts.append(f"{tag}={_describe_histogram(sampled)}")
            continue

        parts.append(f"{tag}={_format_scalar(value)}")
    return " ".join(parts)


def log_metrics(logger, metrics: Mapping[str, MetricValue], step: int) -> None:
    """Write metrics to the standard python logger."""
    # Delegate formatting to a single function to keep logs consistent across callers.
    logger.info(format_metrics_for_console(metrics, step))
