from prefect.client.orchestration import get_client
from prefect.exceptions import ObjectNotFound

from qmodel.scheduling.prefect_remote_orchestrator_deploy import ensure_remote_orchestrator_deployment


def submit_remote_orchestrator(
    *,
    work_pool_name: str,
    deployment_name: str,
    orchestration_path: str,
    run_id: str,
    run_name: str,
) -> str:
    """Submit one remote-orchestrator flow run to Prefect using a deployment."""
    # Resolve the deployment full name (Prefect uses `flow_name/deployment_name`).
    deployment_full_name = f"qmodel-remote-orchestrator/{deployment_name}"

    # Read deployment by name; if missing, create it once then re-read.
    with get_client(sync_client=True) as client:
        try:
            deployment = client.read_deployment_by_name(deployment_full_name)
        except ObjectNotFound:
            ensure_remote_orchestrator_deployment(
                work_pool_name=work_pool_name,
                deployment_name=deployment_name,
            )
            deployment = client.read_deployment_by_name(deployment_full_name)

        # Create a flow run with orchestration parameters.
        flow_run = client.create_flow_run_from_deployment(
            deployment.id,
            parameters={"orchestration_path": orchestration_path, "run_id": run_id},
            name=run_name,
        )

    # Return the flow run id as a string for stable downstream storage/logging.
    return str(flow_run.id)
