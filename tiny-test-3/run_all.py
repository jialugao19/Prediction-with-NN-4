"""Run all tiny-test-3 sanity checks and write one aggregate summary."""

from __future__ import annotations

from pathlib import Path

import yaml

from common import EXP_BASE_DIR, SAVE_EVERY, prepare_tiny_dataset, run_baselines, run_manifest_eval, train_and_evaluate


def main() -> None:
    """Run shuffled-label, shifted-label, and simple-baseline checks."""
    # Run the shuffled-train-label NN check.
    shuffle_dataset = prepare_tiny_dataset("shuffle_train", "shuffle_train_label")
    shuffle_eval = train_and_evaluate(shuffle_dataset)
    shuffle_manifest = run_manifest_eval(shuffle_dataset, int(SAVE_EVERY))

    # Run the one-stock-shifted-label NN check.
    shift_dataset = prepare_tiny_dataset("shift_one_stock", "shift_one_stock_label")
    shift_eval = train_and_evaluate(shift_dataset)
    shift_manifest = run_manifest_eval(shift_dataset, int(SAVE_EVERY))

    # Run simple return-feature baselines without model training.
    baseline_summary = run_baselines("simple_baseline")

    # Persist and print one aggregate summary.
    aggregate = {
        "shuffle_train_label": {"eval": shuffle_eval, "manifest_ic": shuffle_manifest},
        "shift_one_stock_label": {"eval": shift_eval, "manifest_ic": shift_manifest},
        "simple_baseline": baseline_summary,
    }
    out_path = Path(EXP_BASE_DIR) / "summary.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(aggregate, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(yaml.safe_dump({"summary_path": str(out_path.as_posix()), "summary": aggregate}, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
