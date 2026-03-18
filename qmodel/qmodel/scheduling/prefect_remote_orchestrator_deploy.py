from qmodel.scheduling.prefect_remote_orchestrator_flow import qmodel_remote_orchestrator_flow
from prefect.deployments.runner import EntrypointType


def ensure_remote_orchestrator_deployment(*, work_pool_name: str, deployment_name: str) -> str:
    """Upsert a stable Prefect deployment for the remote orchestrator flow."""
    # Create or update the deployment using Prefect 3.x flow.deploy API.
    deployment_id = qmodel_remote_orchestrator_flow.deploy(
        name=deployment_name,
        work_pool_name=work_pool_name,
        entrypoint_type=EntrypointType.MODULE_PATH,
        build=False,
        push=False,
        print_next_steps=False,
    )

    # Return the deployment id as a string for stable downstream storage/logging.
    return str(deployment_id)
