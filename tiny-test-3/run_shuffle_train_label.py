"""Run the shuffled-train-label leakage sanity check."""

from __future__ import annotations

import yaml

from common import SAVE_EVERY, prepare_tiny_dataset, run_manifest_eval, train_and_evaluate


def main() -> None:
    """Prepare shuffled train labels, train briefly, and report train/test IC."""
    # Build the perturbed tiny dataset under /data-cache/nn/tiny-test-3.
    dataset = prepare_tiny_dataset("shuffle_train", "shuffle_train_label")

    # Train for a few steps and compute evaluator metrics.
    eval_summary = train_and_evaluate(dataset)

    # Stream manifests so the report includes pooled and daily IC in main-pipeline style.
    manifest_summary = run_manifest_eval(dataset, int(SAVE_EVERY))

    # Print the important result paths and metrics.
    print(yaml.safe_dump({"eval": eval_summary, "manifest_ic": manifest_summary}, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
