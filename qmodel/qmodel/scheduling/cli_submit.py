import argparse
import os
import sys
import time

from qmodel.scheduling.orchestration import load_orchestration_yaml, require_dict, require_str


def _find_override_value(tokens: list[str], key: str) -> str | None:
    """Find the value of `key` from `--key value` override tokens."""
    # Scan tokens linearly to keep behavior simple and predictable.
    for i in range(len(tokens) - 1):
        if tokens[i] == key:
            return tokens[i + 1]
    return None


def _derive_run_id_from_root_dir(root_dir: str) -> str:
    """Derive a short run_id suffix from a solver.root_dir path string."""
    # Prefer the suffix after `train_logs/` to keep names readable.
    marker = "train_logs/"
    idx = root_dir.rfind(marker)
    if idx >= 0:
        return root_dir[idx + len(marker):].lstrip("/")
    return root_dir.strip("/")


def _resolve_launcher(prefect_submit: dict) -> list[str]:
    """Resolve the experiment launcher tokens from prefect_submit config."""
    # Read optional launcher field to support torchrun-style DDP launches.
    launcher = prefect_submit.get("launcher")
    if launcher is None:
        return ["python"]
    if not isinstance(launcher, list) or not all(isinstance(v, str) for v in launcher):
        raise RuntimeError(f"prefect_submit.launcher must be list[str], got: {type(launcher)}")
    if len(launcher) == 0:
        raise RuntimeError("prefect_submit.launcher must be non-empty list[str]")
    return launcher


def _expand_launcher_tokens(tokens: list[str]) -> list[str]:
    """Expand launcher placeholders like {cuda_device_count}."""
    # Replace {cuda_device_count} with torch.cuda.device_count() for single-yaml local/remote parity.
    placeholder = "{cuda_device_count}"
    if not any(placeholder in tok for tok in tokens):
        return tokens
    import torch

    n = int(torch.cuda.device_count())
    if n <= 0:
        raise RuntimeError("prefect_submit.launcher uses {cuda_device_count} but torch.cuda.device_count() <= 0")
    return [tok.replace(placeholder, str(n)) for tok in tokens]


def main(argv: list[str] | None = None) -> None:
    """Submit `<launcher> entry.py conf.py ...` to Prefect using orchestration.yaml."""
    # Parse positional args and keep all remaining tokens as train-config overrides.
    parser = argparse.ArgumentParser(prog="qmodel-submit")
    parser.add_argument("conf_path", type=str)
    parser.add_argument("orchestration_path", type=str)
    ns, rest = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    # Load Prefect submit config from orchestration yaml.
    orch = load_orchestration_yaml(ns.orchestration_path)
    prefect_submit = require_dict(orch, "prefect_submit")

    # Resolve required Prefect fields and set API url for the Prefect client.
    prefect_api_url = require_str(prefect_submit, "prefect_api_url")
    work_pool = require_str(prefect_submit, "work_pool")
    deployment = require_str(prefect_submit, "deployment")
    run_name_prefix = require_str(prefect_submit, "run_name_prefix")
    os.environ["PREFECT_API_URL"] = prefect_api_url
    from qmodel.scheduling.prefect_deploy import ensure_experiment_deployment
    from qmodel.scheduling.prefect_submit import submit_experiment

    # Build a stable run name using solver.root_dir override when available.
    root_dir_override = _find_override_value(rest, "--solver.root_dir")
    run_id = time.strftime("%m-%d/%H-%M-%S") if root_dir_override is None else _derive_run_id_from_root_dir(root_dir_override)
    run_name = f"{run_name_prefix}-{run_id}"

    # Build experiment command at current working directory.
    cwd = os.getcwd()
    launcher = _expand_launcher_tokens(_resolve_launcher(prefect_submit))
    command = launcher + ["entry.py", ns.conf_path] + rest

    # Ensure deployment exists for the target work pool.
    ensure_experiment_deployment(
        work_pool_name=work_pool,
        deployment_name=deployment,
    )

    # Submit one flow run to Prefect for execution by workers.
    flow_run_id = submit_experiment(
        work_pool_name=work_pool,
        deployment_name=deployment,
        command=command,
        cwd=cwd,
        run_name=run_name,
    )

    print(flow_run_id)
