"""Expose the vendored qmodel package for local imports."""

from __future__ import annotations

from pathlib import Path


# Extend package search path to include the actual source tree under `qmodel/qmodel/`.
__path__.append(str(Path(__file__).resolve().parent / "qmodel"))

