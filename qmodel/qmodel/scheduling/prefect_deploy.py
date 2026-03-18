from qmodel.scheduling.prefect_experiment_flow import qmodel_experiment_flow
from prefect.deployments.runner import EntrypointType


def ensure_experiment_deployment(
    *,
    work_pool_name: str,
    deployment_name: str,
) -> str:
    # Upsert a stable Prefect deployment bound to the given work pool.

    # Create or update the deployment using Prefect 3.x flow.deploy API.
    deployment_id = qmodel_experiment_flow.deploy(
        name=deployment_name,
        work_pool_name=work_pool_name,
        entrypoint_type=EntrypointType.MODULE_PATH,
        build=False,
        push=False,
        print_next_steps=False,
    )

    # Return the deployment id as a string for stable downstream storage/logging.
    return str(deployment_id)
