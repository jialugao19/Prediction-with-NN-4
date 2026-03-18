from prefect import flow
from prefect.logging.loggers import get_run_logger

from qmodel.scheduling.subprocess_runner import run_and_stream


@flow(name="qmodel-experiment")
def qmodel_experiment_flow(
    *,
    command: list[str],
    cwd: str,
) -> None:
    # Execute an experiment command on the Prefect worker and stream logs.

    # Get Prefect run logger to write subprocess output into the UI.
    logger = get_run_logger()

    # Run the command and forward each output line to Prefect logs.
    run_and_stream(command=command, cwd=cwd, env=None, on_line=logger.info)
