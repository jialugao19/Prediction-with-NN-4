"""Run simple return-feature baselines on the tiny sanity-check sample."""

from __future__ import annotations

import yaml

from common import run_baselines


def main() -> None:
    """Compute pooled and daily IC for simple return-feature baselines."""
    # Compute the baseline summary without training any neural model.
    summary = run_baselines("simple_baseline")

    # Print the compact YAML summary for terminal inspection.
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
