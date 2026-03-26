"""Run report postprocess only from existing prediction-NN-2 artifacts."""

from prediction_nn2.pipeline import _default_config, run_pipeline_postprocess_only


def main() -> None:
    """Run postprocess only with the module-level default config."""
    # Reuse the same default config as the full pipeline entrypoint.
    cfg = _default_config()

    # Execute report postprocess from existing checkpoints and eval outputs.
    run_pipeline_postprocess_only(cfg)


if __name__ == "__main__":
    main()
