import argparse
import os
import sys
import time
from pathlib import Path

from qmodel.scheduling.orchestration import load_orchestration_yaml, require_dict, require_str


def main(argv: list[str] | None = None) -> None:
    """Submit a remote-orchestrator flow run to local Prefect using orchestration.yaml."""
    # Parse the orchestration yaml path as the only CLI input to keep UX simple.
    parser = argparse.ArgumentParser(prog="qmodel-remote-submit")
    parser.add_argument("orchestration_path", type=str)
    ns = parser.parse_args(argv if argv is not None else sys.argv[1:])
    orchestration_path_abs = str(Path(ns.orchestration_path).resolve())

    # Load orchestration yaml and resolve local orchestrator settings.
    orch = load_orchestration_yaml(orchestration_path_abs)
    remote = require_dict(orch, "remote_orchestrate")
    local_prefect_api_url = require_str(remote, "local_prefect_api_url")
    local_work_pool = require_str(remote, "local_work_pool")
    local_deployment = require_str(remote, "local_deployment")
    expm_tag = require_str(remote, "expm_tag")

    # Point Prefect client to the local server explicitly.
    os.environ["PREFECT_API_URL"] = local_prefect_api_url
    from qmodel.scheduling.prefect_remote_orchestrator_deploy import ensure_remote_orchestrator_deployment
    from qmodel.scheduling.prefect_remote_orchestrator_submit import submit_remote_orchestrator

    # Build run_id and run_name for traceability in Prefect UI.
    mm_dd = time.strftime("%m-%d")
    hh_mm_ss = time.strftime("%H-%M-%S")
    run_id = f"{mm_dd}/{expm_tag}/{hh_mm_ss}"
    run_name = f"remote-orch-{run_id}"

    # Ensure the orchestrator deployment exists and is bound to the desired pool.
    ensure_remote_orchestrator_deployment(work_pool_name=local_work_pool, deployment_name=local_deployment)

    # Submit one orchestrator run carrying the yaml path and chosen run_id.
    flow_run_id = submit_remote_orchestrator(
        work_pool_name=local_work_pool,
        deployment_name=local_deployment,
        orchestration_path=orchestration_path_abs,
        run_id=run_id,
        run_name=run_name,
    )

    print(flow_run_id)
