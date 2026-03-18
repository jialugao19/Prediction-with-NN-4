import os
import subprocess
from collections.abc import Callable


def run_and_stream(
    *,
    command: list[str],
    cwd: str,
    env: dict[str, str] | None,
    on_line: Callable[[str], None],
) -> None:
    # Run a command and stream its stdout line-by-line to the callback.

    # Merge the provided environment with the current process environment.
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)

    # Spawn the subprocess and redirect stderr to stdout for a single log stream.
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream logs as they arrive to keep Prefect UI responsive.
    assert proc.stdout is not None
    for line in proc.stdout:
        on_line(line.rstrip("\n"))

    # Wait for completion and fail the flow run if the command fails.
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed rc={rc}: {command} (cwd={cwd})")
