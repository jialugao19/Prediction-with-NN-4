"""Provide a small rolling timer for training/inference loops."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class TimerMeans:
    """Hold rolling mean timing values in milliseconds."""

    iter_ms: float
    data_ms: float
    model_ms: float
    loader_cpu_ms: float
    h2d_submit_ms: float
    h2d_gpu_ms: float
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    checkpoint_ms: float

    def __str__(self) -> str:
        """Format the timer means for human-friendly display."""
        # Include the core timings first so older log readers still scan naturally.
        return (
            f"iter={self.iter_ms:.3f}ms data={self.data_ms:.3f}ms model={self.model_ms:.3f}ms "
            f"loader_cpu={self.loader_cpu_ms:.3f}ms h2d_gpu={self.h2d_gpu_ms:.3f}ms "
            f"forward={self.forward_ms:.3f}ms backward={self.backward_ms:.3f}ms "
            f"optimizer={self.optimizer_ms:.3f}ms checkpoint={self.checkpoint_ms:.3f}ms"
        )


class RollingTimer:
    """Track rolling-window iteration/data/model times for logging."""

    def __init__(self, window_size: int) -> None:
        """Initialize a timer with a fixed rolling window length."""
        # Allocate bounded deques to store recent timing samples.
        self.window_size = int(window_size)
        self._iter_ms: deque[float] = deque(maxlen=self.window_size)
        self._data_ms: deque[float] = deque(maxlen=self.window_size)
        self._model_ms: deque[float] = deque(maxlen=self.window_size)
        self._loader_cpu_ms: deque[float] = deque(maxlen=self.window_size)
        self._h2d_submit_ms: deque[float] = deque(maxlen=self.window_size)
        self._h2d_gpu_ms: deque[float] = deque(maxlen=self.window_size)
        self._forward_ms: deque[float] = deque(maxlen=self.window_size)
        self._backward_ms: deque[float] = deque(maxlen=self.window_size)
        self._optimizer_ms: deque[float] = deque(maxlen=self.window_size)
        self._checkpoint_ms: deque[float] = deque(maxlen=self.window_size)

    def add(
        self,
        *,
        iter_ms: float,
        data_ms: float,
        model_ms: float,
        loader_cpu_ms: float = 0.0,
        h2d_submit_ms: float = 0.0,
        h2d_gpu_ms: float = 0.0,
        forward_ms: float = 0.0,
        backward_ms: float = 0.0,
        optimizer_ms: float = 0.0,
        checkpoint_ms: float = 0.0,
    ) -> None:
        """Add one iteration timing sample."""
        # Append the new sample into each rolling series.
        self._iter_ms.append(float(iter_ms))
        self._data_ms.append(float(data_ms))
        self._model_ms.append(float(model_ms))
        self._loader_cpu_ms.append(float(loader_cpu_ms))
        self._h2d_submit_ms.append(float(h2d_submit_ms))
        self._h2d_gpu_ms.append(float(h2d_gpu_ms))
        self._forward_ms.append(float(forward_ms))
        self._backward_ms.append(float(backward_ms))
        self._optimizer_ms.append(float(optimizer_ms))
        self._checkpoint_ms.append(float(checkpoint_ms))

    def means(self) -> TimerMeans:
        """Compute rolling means for each timing series."""
        # Compute means from rolling buffers without special-casing every field inline.
        def mean(values: deque[float]) -> float:
            """Return the arithmetic mean for one rolling buffer."""
            # Use zero for the startup state before the first sample lands.
            return sum(values) / len(values) if values else 0.0

        # Return one immutable snapshot for console and TensorBoard metrics.
        return TimerMeans(
            iter_ms=mean(self._iter_ms),
            data_ms=mean(self._data_ms),
            model_ms=mean(self._model_ms),
            loader_cpu_ms=mean(self._loader_cpu_ms),
            h2d_submit_ms=mean(self._h2d_submit_ms),
            h2d_gpu_ms=mean(self._h2d_gpu_ms),
            forward_ms=mean(self._forward_ms),
            backward_ms=mean(self._backward_ms),
            optimizer_ms=mean(self._optimizer_ms),
            checkpoint_ms=mean(self._checkpoint_ms),
        )
