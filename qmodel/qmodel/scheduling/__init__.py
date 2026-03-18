from __future__ import annotations


def ensure_experiment_deployment(*, work_pool_name: str, deployment_name: str) -> str:
    """Lazily import and upsert a Prefect deployment for qmodel-experiment."""
    # Import implementation lazily to avoid importing Prefect on module import.
    from qmodel.scheduling.prefect_deploy import ensure_experiment_deployment as _impl

    # Delegate to the real implementation.
    return _impl(work_pool_name=work_pool_name, deployment_name=deployment_name)


def submit_experiment(
    *,
    work_pool_name: str,
    deployment_name: str,
    command: list[str],
    cwd: str,
    run_name: str,
) -> str:
    """Lazily import and submit one qmodel-experiment flow run."""
    # Import implementation lazily to avoid importing Prefect on module import.
    from qmodel.scheduling.prefect_submit import submit_experiment as _impl

    # Delegate to the real implementation.
    return _impl(
        work_pool_name=work_pool_name,
        deployment_name=deployment_name,
        command=command,
        cwd=cwd,
        run_name=run_name,
    )



__all__ = [
    "ensure_experiment_deployment",
    "submit_experiment",
]
