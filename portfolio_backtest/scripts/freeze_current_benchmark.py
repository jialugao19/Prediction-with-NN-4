"""Freeze the current prediction-NN-2 baseline benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prediction_nn2.benchmark_freeze import freeze_current_baseline


def main() -> None:
    """Run the current baseline freeze workflow."""
    # Execute the P0 freeze and print the benchmark manifest path.
    benchmark_path = freeze_current_baseline()
    print(Path(benchmark_path).as_posix())


if __name__ == "__main__":
    main()
