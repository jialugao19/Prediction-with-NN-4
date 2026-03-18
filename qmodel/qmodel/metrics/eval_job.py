"""Run evaluation metrics from feather shards in a spawn-safe subprocess."""

from __future__ import annotations

import functools
import importlib
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from qmodel.metrics.core import Histogram, MetricValue, run_metric_fns


@dataclass(frozen=True)
class MetricFnSpec:
    """Describe one metric callable in an importable and spawn-safe way."""

    module: str
    qualname: str
    partial_args: tuple[Any, ...]
    partial_kwargs: dict[str, Any]


def _require_top_level_fn(fn: Callable[..., Any]) -> None:
    """Require metric function to be importable as a top-level symbol."""
    # Reject nested functions and lambdas that cannot be imported by qualname.
    qualname = getattr(fn, "__qualname__", None)
    module = getattr(fn, "__module__", None)
    if qualname is None or module is None:
        raise RuntimeError(f"Metric function must have __module__/__qualname__: {fn}")
    if "<locals>" in qualname:
        raise RuntimeError(f"Metric function must be top-level (no <locals>): {module}:{qualname}")


def metric_fns_to_specs(metric_fns: Sequence[Callable[..., Mapping[str, MetricValue]]]) -> list[MetricFnSpec]:
    """Convert metric callables into spawn-safe specs."""
    # Normalize raw callables and functools.partial into a stable serializable spec list.
    specs: list[MetricFnSpec] = []
    for metric_fn in metric_fns:
        if isinstance(metric_fn, functools.partial):
            _require_top_level_fn(metric_fn.func)
            specs.append(
                MetricFnSpec(
                    module=metric_fn.func.__module__,
                    qualname=metric_fn.func.__qualname__,
                    partial_args=tuple(metric_fn.args),
                    partial_kwargs=dict(metric_fn.keywords or {}),
                )
            )
            continue

        _require_top_level_fn(metric_fn)
        specs.append(
            MetricFnSpec(
                module=metric_fn.__module__,
                qualname=metric_fn.__qualname__,
                partial_args=(),
                partial_kwargs={},
            )
        )

    return specs


def _import_metric_fn(spec: MetricFnSpec) -> Callable[..., Mapping[str, MetricValue]]:
    """Import one metric function from a MetricFnSpec."""
    # Resolve `module:qualname` into a python callable and re-apply partial args/kwargs.
    mod = importlib.import_module(spec.module)
    obj: Any = mod
    for part in spec.qualname.split("."):
        obj = getattr(obj, part)

    fn = obj
    if spec.partial_args or spec.partial_kwargs:
        fn = functools.partial(fn, *spec.partial_args, **spec.partial_kwargs)
    return fn


def compute_eval_metrics_from_shards(
    *,
    shard_dir: str,
    metric_specs: Sequence[MetricFnSpec],
    namespace: str,
    it: int,
    group: str,
    out_path: str,
) -> None:
    """Load shard feathers, compute metrics, and write scalar metrics as JSON."""
    # Collect shard feather files and build one merged dataframe.
    shard_path = Path(shard_dir)
    shard_files = sorted(shard_path.glob("rank*.feather"))
    if not shard_files:
        raise RuntimeError(f"No shard feather files found under: {shard_path}")

    dfs = [pd.read_feather(p) for p in shard_files]
    df = pd.concat(dfs, ignore_index=True)

    # Build the minimal payload required by evaluation metric functions.
    pred = df["prediction"].to_numpy()
    target = df["target"].to_numpy()
    payload = {
        "namespace": namespace,
        "it": it,
        "group": group,
        "df": df,
        "pred": pred,
        "target": target,
    }

    # Import metric callables and run them with strict duplicate-tag checks.
    metric_fns = [_import_metric_fn(spec) for spec in metric_specs]
    metrics = run_metric_fns(metric_fns, payload)

    # Enforce scalar-only output to keep IPC stable and small.
    scalar_metrics: dict[str, float] = {}
    for tag, value in metrics.items():
        if isinstance(value, Histogram):
            raise RuntimeError(f"Histogram metric is not supported in spawn eval job: {tag}")
        scalar_metrics[tag] = float(value)

    # Persist scalar metrics to a JSON file for the parent rank0 process to consume.
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(scalar_metrics, f, ensure_ascii=False, sort_keys=True)


def spawn_eval_metrics_job(
    *,
    shard_dir: str,
    metric_specs: Sequence[MetricFnSpec],
    namespace: str,
    it: int,
    group: str,
    out_path: str,
) -> mp.Process:
    """Spawn a CPU subprocess to compute eval metrics from shard feathers."""
    # Create a spawn context to avoid CUDA+fork issues in DDP training jobs.
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=compute_eval_metrics_from_shards,
        kwargs={
            "shard_dir": shard_dir,
            "metric_specs": list(metric_specs),
            "namespace": namespace,
            "it": int(it),
            "group": group,
            "out_path": out_path,
        },
        daemon=False,
    )
    p.start()
    return p
