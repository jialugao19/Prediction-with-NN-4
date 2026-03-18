import argparse
import os
import sys
import time

from qmodel.scheduling.prefect_deploy import ensure_experiment_deployment
from qmodel.scheduling.prefect_submit import submit_experiment


def _split_meta_and_train_args(argv: list[str]) -> tuple[list[str], list[str]]:
    # Split argv into meta args and train args using explicit markers.
    if "--args-to-experiment-meta" not in argv:
        raise RuntimeError("Missing --args-to-experiment-meta")
    if "--args-to-train-config" not in argv:
        raise RuntimeError("Missing --args-to-train-config")

    meta_i = argv.index("--args-to-experiment-meta")
    train_i = argv.index("--args-to-train-config")
    if train_i < meta_i:
        raise RuntimeError("--args-to-train-config must appear after --args-to-experiment-meta")

    meta_args = argv[meta_i + 1:train_i]
    train_args = argv[train_i + 1:]
    return meta_args, train_args


def _build_parser() -> argparse.ArgumentParser:
    # Build the CLI parser for submitting an experiment run.
    parser = argparse.ArgumentParser(prog="qmodel-submit")

    parser.add_argument("--work-pool", required=True, type=str)
    parser.add_argument("--deployment", required=True, type=str)
    parser.add_argument("--run-name", required=False, type=str)
    parser.add_argument("--cwd", required=False, type=str)

    parser.add_argument(
        "--args-to-experiment-meta",
        nargs="*",
        help="Meta segment marker; actual tokens are split by --args-to-train-config",
    )
    parser.add_argument(
        "--args-to-train-config",
        nargs=argparse.REMAINDER,
        help="Train config args marker; all remaining tokens are passed to the entry file",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    # Parse args, ensure deployment, then submit a single Prefect flow run.
    if argv is None:
        argv = sys.argv[1:]

    # Split by markers first to keep the user's desired CLI shape.
    meta_args, train_args = _split_meta_and_train_args(argv)

    # Parse standard options from the full argv.
    parser = _build_parser()
    ns = parser.parse_args(argv)

    # Resolve entry file from meta args (keep it strict to avoid ambiguity).
    if len(meta_args) != 1:
        raise RuntimeError("--args-to-experiment-meta must contain exactly one entry file")
    entry_file = meta_args[0]

    # Default cwd to current working directory to match 'at pwd' requirement.
    cwd = ns.cwd if ns.cwd is not None else os.getcwd()

    # Build a stable default run name if not provided.
    run_name = ns.run_name if ns.run_name is not None else f"qmodel_{time.strftime('%Y%m%d_%H%M%S')}"

    # Build the actual experiment command: python entry_file + remaining args.
    command = ["python", entry_file] + train_args

    # Upsert the deployment to bind it to the target work pool.
    ensure_experiment_deployment(
        work_pool_name=ns.work_pool,
        deployment_name=ns.deployment,
    )

    # Submit a flow run for the command.
    flow_run_id = submit_experiment(
        work_pool_name=ns.work_pool,
        deployment_name=ns.deployment,
        command=command,
        cwd=cwd,
        run_name=run_name,
    )

    print(flow_run_id)
