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

    def __str__(self) -> str:
        """Format the timer means for human-friendly display."""
        return f"iter={self.iter_ms:.3f}ms data={self.data_ms:.3f}ms model={self.model_ms:.3f}ms"


class RollingTimer:
    """Track rolling-window iteration/data/model times for logging."""

    def __init__(self, window_size: int) -> None:
        """Initialize a timer with a fixed rolling window length."""
        # Allocate bounded deques to store recent timing samples.
        self.window_size = int(window_size)
        self._iter_ms: deque[float] = deque(maxlen=self.window_size)
        self._data_ms: deque[float] = deque(maxlen=self.window_size)
        self._model_ms: deque[float] = deque(maxlen=self.window_size)

    def add(self, *, iter_ms: float, data_ms: float, model_ms: float) -> None:
        """Add one iteration timing sample."""
        # Append the new sample into each rolling series.
        self._iter_ms.append(float(iter_ms))
        self._data_ms.append(float(data_ms))
        self._model_ms.append(float(model_ms))

    def means(self) -> TimerMeans:
        """Compute rolling means for each timing series."""
        # Compute means from rolling buffers without special-casing empty state.
        iter_ms = sum(self._iter_ms) / len(self._iter_ms) if self._iter_ms else 0.0
        data_ms = sum(self._data_ms) / len(self._data_ms) if self._data_ms else 0.0
        model_ms = sum(self._model_ms) / len(self._model_ms) if self._model_ms else 0.0
        return TimerMeans(iter_ms=iter_ms, data_ms=data_ms, model_ms=model_ms)
