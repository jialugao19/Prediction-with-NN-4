"""Provide performance-focused console logging helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import pynvml


GiB = 1024.0 * 1024.0 * 1024.0


def _require_cuda(device: torch.device) -> None:
    """Require a CUDA device for GPU perf logging."""
    # Validate device type early so callers fail loudly when misconfigured.
    if torch.device(device).type != "cuda":
        raise RuntimeError(f"Perf console logging requires CUDA device, got: {device}")


def _gib(bytes_value: float) -> float:
    """Convert bytes to GiB."""
    # Keep conversion as a tiny helper so formatting logic stays clean.
    return float(bytes_value) / GiB


@dataclass(frozen=True)
class NvmlSnapshot:
    """Hold a snapshot of NVML memory and utilization stats."""

    used_gib: float
    total_gib: float
    gpu_util_pct: int
    mem_util_pct: int


class NvmlReader:
    """Read NVML stats for the current process CUDA device."""

    def __init__(self, device: torch.device) -> None:
        """Initialize NVML and bind to the CUDA device UUID."""
        # Validate that the device is CUDA and normalize to a torch.device instance.
        _require_cuda(device)
        self.device = torch.device(device)

        # Initialize NVML once per process so reads are cheap.
        pynvml.nvmlInit()

        # Resolve NVML handle by UUID so it is stable under CUDA_VISIBLE_DEVICES/DDP.
        props = torch.cuda.get_device_properties(self.device.index)
        uuid = getattr(props, "uuid", None)
        if uuid is None:
            raise RuntimeError("torch cuda device properties missing uuid; cannot bind NVML device")
        self._nvml_uuid = f"GPU-{uuid}"
        self._handle = pynvml.nvmlDeviceGetHandleByUUID(self._nvml_uuid)

    def snapshot(self) -> NvmlSnapshot:
        """Read current NVML stats (memory used/total + utilization)."""
        # Query memory stats from NVML in bytes.
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        used_gib = _gib(float(mem.used))
        total_gib = _gib(float(mem.total))

        # Query utilization in percent from NVML.
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        gpu_util_pct = int(util.gpu)
        mem_util_pct = int(util.memory)

        return NvmlSnapshot(
            used_gib=used_gib,
            total_gib=total_gib,
            gpu_util_pct=gpu_util_pct,
            mem_util_pct=mem_util_pct,
        )


@dataclass(frozen=True)
class TorchCudaMemSnapshot:
    """Hold a snapshot of torch CUDA allocator memory stats."""

    alloc_gib: float
    reserved_gib: float
    max_alloc_gib: float
    max_reserved_gib: float


def torch_cuda_mem_snapshot(device: torch.device) -> TorchCudaMemSnapshot:
    """Read torch.cuda memory stats for one CUDA device."""
    # Validate device type and normalize for APIs that expect an index.
    _require_cuda(device)
    device = torch.device(device)

    # Read allocator stats in bytes and convert to GiB for readability.
    alloc_gib = _gib(float(torch.cuda.memory_allocated(device)))
    reserved_gib = _gib(float(torch.cuda.memory_reserved(device)))
    max_alloc_gib = _gib(float(torch.cuda.max_memory_allocated(device)))
    max_reserved_gib = _gib(float(torch.cuda.max_memory_reserved(device)))

    return TorchCudaMemSnapshot(
        alloc_gib=alloc_gib,
        reserved_gib=reserved_gib,
        max_alloc_gib=max_alloc_gib,
        max_reserved_gib=max_reserved_gib,
    )


def format_train_perf_line(
    *,
    step: int,
    loss: float,
    loss_mean: float,
    data_ms: float,
    model_ms: float,
    rank: int,
    nvml: NvmlSnapshot,
    torch_mem: TorchCudaMemSnapshot,
) -> str:
    """Format a single-line train perf summary for console logs."""
    # Assemble a stable, grep-friendly key=value line.
    return (
        "stage=train "
        f"step={int(step)} "
        f"loss={float(loss):.6g} "
        f"loss_mean={float(loss_mean):.6g} "
        f"data_ms={float(data_ms):.3f} "
        f"model_ms={float(model_ms):.3f} "
        f"rank={int(rank)} "
        f"nvml_used_gib={nvml.used_gib:.3f} "
        f"nvml_total_gib={nvml.total_gib:.3f} "
        f"nvml_gpu_util_pct={int(nvml.gpu_util_pct)} "
        f"nvml_mem_util_pct={int(nvml.mem_util_pct)} "
        f"torch_alloc_gib={torch_mem.alloc_gib:.3f} "
        f"torch_reserved_gib={torch_mem.reserved_gib:.3f} "
        f"torch_max_alloc_gib={torch_mem.max_alloc_gib:.3f} "
        f"torch_max_reserved_gib={torch_mem.max_reserved_gib:.3f}"
    )


def format_eval_perf_line(
    *,
    it: int,
    group: str,
    data_ms: float,
    model_ms: float,
    rank: int,
    nvml: NvmlSnapshot,
    torch_mem: TorchCudaMemSnapshot,
) -> str:
    """Format a single-line eval perf summary for console logs."""
    # Assemble a stable, grep-friendly key=value line.
    return (
        "stage=eval "
        f"group={str(group)} "
        f"it={int(it)} "
        f"data_ms={float(data_ms):.3f} "
        f"model_ms={float(model_ms):.3f} "
        f"rank={int(rank)} "
        f"nvml_used_gib={nvml.used_gib:.3f} "
        f"nvml_total_gib={nvml.total_gib:.3f} "
        f"nvml_gpu_util_pct={int(nvml.gpu_util_pct)} "
        f"nvml_mem_util_pct={int(nvml.mem_util_pct)} "
        f"torch_alloc_gib={torch_mem.alloc_gib:.3f} "
        f"torch_reserved_gib={torch_mem.reserved_gib:.3f} "
        f"torch_max_alloc_gib={torch_mem.max_alloc_gib:.3f} "
        f"torch_max_reserved_gib={torch_mem.max_reserved_gib:.3f}"
    )
