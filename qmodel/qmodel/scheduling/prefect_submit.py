from prefect.client.orchestration import get_client
from prefect.exceptions import ObjectNotFound

from qmodel.scheduling.prefect_deploy import ensure_experiment_deployment


def submit_experiment(
    *,
    work_pool_name: str,
    deployment_name: str,
    command: list[str],
    cwd: str,
    run_name: str,
) -> str:
    # Submit a single experiment run by creating a Prefect flow run from deployment.

    # Resolve the deployment full name (Prefect uses `flow_name/deployment_name`).
    deployment_full_name = f"qmodel-experiment/{deployment_name}"

    # Read deployment by name; if missing, create it once then re-read.
    with get_client(sync_client=True) as client:
        try:
            deployment = client.read_deployment_by_name(deployment_full_name)
        except ObjectNotFound:
            ensure_experiment_deployment(
                work_pool_name=work_pool_name,
                deployment_name=deployment_name,
            )
            deployment = client.read_deployment_by_name(deployment_full_name)

        # Create a flow run with parameters and optional job variables for the worker.
        flow_run = client.create_flow_run_from_deployment(
            deployment.id,
            parameters={"command": command, "cwd": cwd},
            name=run_name,
        )

    # Return the flow run id as a string for stable downstream storage/logging.
    return str(flow_run.id)
