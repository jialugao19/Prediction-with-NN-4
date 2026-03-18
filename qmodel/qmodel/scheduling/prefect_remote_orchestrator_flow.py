from __future__ import annotations

import ast
import shlex
import shutil
import subprocess
from pathlib import Path
import re

from prefect import flow
from prefect.logging.loggers import get_run_logger

from qmodel.scheduling.orchestration import (
    load_orchestration_yaml,
    require_bool,
    require_dict,
    require_list_str,
    require_str,
)
from qmodel.scheduling.subprocess_runner import run_and_stream


def _run_logged(*, command: list[str], cwd: str, logger) -> None:
    """Run a subprocess and stream stdout to Prefect logs."""
    # Delegate streaming to the shared runner to keep log behavior consistent.
    run_and_stream(command=command, cwd=cwd, env=None, on_line=logger.info)


def _run_capture_logged(*, command: list[str], cwd: str, logger) -> str:
    """Run a subprocess, stream stdout to logs, and return the full text output."""
    # Spawn the subprocess and merge stderr into stdout for a single log stream.
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream each output line into Prefect logs while capturing it for parsing.
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        s = line.rstrip("\n")
        logger.info(s)
        lines.append(s)

    # Fail fast if the command fails.
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed rc={rc}: {command} (cwd={cwd})")

    # Return captured output text for downstream parsing.
    return "\n".join(lines)


def _parse_expm_clone_output(text: str) -> dict[str, str]:
    """Parse expm clone stdout (flat `key: value` yaml) into a dict."""
    # Split each `key: value` line and keep only expected flat keys.
    res: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            raise RuntimeError(f"unexpected expm output line: {raw!r}")
        key, value = line.split(":", 1)
        v = value.strip()
        if len(v) >= 2 and ((v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"'))):
            v = v[1:-1]
        res[key.strip()] = v

    # Ensure required keys exist for downstream path computation.
    for k in ["tag", "hex", "path"]:
        if k not in res:
            raise RuntimeError(f"missing {k} in expm clone output: {res}")
    return res


def _resolve_home_path(template: str, home: str) -> str:
    """Resolve `$HOME/...` template into an absolute path using resolved home."""
    # Expand a leading $HOME prefix only (no other env expansion).
    if template.startswith("$HOME"):
        return home.rstrip("/") + template[len("$HOME"):]
    return template


def _parse_pairs(tokens: list[str]) -> dict[str, str]:
    """Parse a `--key value` flat token list into a dict."""
    # Validate the token list shape to avoid silent override bugs.
    if len(tokens) % 2 != 0:
        raise RuntimeError("override tokens must be '--key value' pairs (even length list[str])")

    # Build a strict map and fail fast on duplicates.
    res: dict[str, str] = {}
    for i in range(0, len(tokens), 2):
        k = tokens[i]
        v = tokens[i + 1]
        if not isinstance(k, str) or not isinstance(v, str):
            raise RuntimeError("override tokens must be list[str]")
        if not k.startswith("--"):
            raise RuntimeError(f"override key must start with '--': {k!r}")
        if k in res:
            raise RuntimeError(f"duplicate override key: {k}")
        res[k] = v
    return res


def _parse_eval_checkpoint_iter(value: str) -> list[int]:
    """Parse `--eval.evaluator.eval_checkpoint_iter` string into list[int]."""
    # Parse python literal form like "[100000,150000]" strictly.
    try:
        obj = ast.literal_eval(value)
    except Exception as exc:
        raise RuntimeError(f"invalid eval_checkpoint_iter literal: {value!r}") from exc

    # Normalize to a list of ints.
    if isinstance(obj, (list, tuple)):
        items = list(obj)
    else:
        items = [obj]

    res: list[int] = []
    for it in items:
        if isinstance(it, int):
            res.append(int(it))
        elif isinstance(it, str) and it.strip().lstrip("-").isdigit():
            res.append(int(it.strip()))
        else:
            raise RuntimeError(f"eval_checkpoint_iter items must be int-like: got {it!r}")

    if len(res) == 0:
        raise RuntimeError("eval_checkpoint_iter must be non-empty")
    return res


def _eval_dir_from_group(group: str) -> str:
    """Map group name to evaluation output directory name."""
    # Follow Evaluator's convention: val uses eval_val, others use eval.
    if group == "val":
        return "eval_val"
    if group == "test":
        return "eval"
    raise RuntimeError(f"unsupported eval group for eval-only rsync: {group!r}")


def _build_eval_only_rsync_jobs(
    *,
    remote_root_dir: str,
    local_train_logs_remote_abs: str,
    target_run_id: str,
    group: str,
    eval_checkpoint_iter: list[int],
    extra_paths: list[str],
) -> list[tuple[str, str]]:
    """Build `(src, dst)` rsync directory jobs for eval-only artifact sync."""
    # Build eval shard jobs per iter for the selected group.
    eval_dir = _eval_dir_from_group(group)
    jobs: list[tuple[str, str]] = []
    for it in eval_checkpoint_iter:
        src = f"{remote_root_dir.rstrip('/')}/{eval_dir}/iter_{int(it)}/"
        dst = f"{local_train_logs_remote_abs.rstrip('/')}/{target_run_id}/{eval_dir}/iter_{int(it)}/"
        jobs.append((src, dst))

    # Add optional extra paths for convenience (e.g., tb/).
    for rel in extra_paths:
        if not isinstance(rel, str) or rel == "":
            raise RuntimeError(f"rsync_back_extra_paths must be non-empty strings: got {rel!r}")
        src = f"{remote_root_dir.rstrip('/')}/{rel.rstrip('/')}/"
        dst = f"{local_train_logs_remote_abs.rstrip('/')}/{target_run_id}/{rel.rstrip('/')}/"
        jobs.append((src, dst))

    return jobs


@flow(name="qmodel-remote-orchestrator")
def qmodel_remote_orchestrator_flow(*, orchestration_path: str, run_id: str) -> None:
    """Clone an experiment, rsync to remote, submit remote Prefect run, and rsync back train_logs."""
    # Load orchestration config for both remote orchestration and Prefect submit settings.
    logger = get_run_logger()
    orch = load_orchestration_yaml(orchestration_path)
    remote = require_dict(orch, "remote_orchestrate")
    prefect_submit = require_dict(orch, "prefect_submit")

    # Resolve required local/remote settings from yaml.
    repo_path = require_str(remote, "repo_path")
    expm_tag = require_str(remote, "expm_tag")
    conf_relpath = require_str(remote, "conf_relpath")
    local_train_logs_remote_root = require_str(remote, "local_train_logs_remote_root")
    remote_host = require_str(remote, "remote_host")
    remote_experiment_root = require_str(remote, "remote_experiment_root")
    remote_train_logs_root = require_str(remote, "remote_train_logs_root")
    rsync_excludes = require_list_str(remote, "rsync_excludes")
    remote_submit_overrides = require_list_str(remote, "submit_overrides")

    # Resolve optional mode and rsync-back policy.
    mode = remote.get("mode", "train")
    if not isinstance(mode, str):
        raise RuntimeError(f"remote_orchestrate.mode must be str, got {type(mode)}")
    if mode not in ["train", "eval"]:
        raise RuntimeError(f"remote_orchestrate.mode must be 'train'|'eval', got: {mode!r}")

    eval_run_id = remote.get("eval_run_id", None)
    if mode == "eval":
        if not isinstance(eval_run_id, str) or eval_run_id == "":
            raise RuntimeError("remote_orchestrate.eval_run_id must be a non-empty str when mode='eval'")

    rsync_back_mode = remote.get("rsync_back_mode", "all")
    if not isinstance(rsync_back_mode, str):
        raise RuntimeError(f"remote_orchestrate.rsync_back_mode must be str, got {type(rsync_back_mode)}")
    if rsync_back_mode not in ["all", "eval_only"]:
        raise RuntimeError(f"remote_orchestrate.rsync_back_mode must be 'all'|'eval_only', got: {rsync_back_mode!r}")

    rsync_back_extra_paths = remote.get("rsync_back_extra_paths", [])
    if not isinstance(rsync_back_extra_paths, list) or not all(isinstance(v, str) for v in rsync_back_extra_paths):
        raise RuntimeError("remote_orchestrate.rsync_back_extra_paths must be list[str]")

    # Resolve cache-check policy from yaml.
    cache_check = require_dict(remote, "cache_check")
    cache_check_enabled = require_bool(cache_check, "enabled")
    cache_check_config_relpath = require_str(cache_check, "config_relpath")
    cache_check_marker_filename = require_str(cache_check, "marker_filename")
    cache_check_rebuild_cmd = cache_check["rebuild_cmd"]
    if not isinstance(cache_check_rebuild_cmd, list) or not all(isinstance(v, str) for v in cache_check_rebuild_cmd):
        raise RuntimeError("cache_check.rebuild_cmd must be list[str]")

    # Resolve Prefect API url for remote submit and remote polling.
    prefect_api_url = require_str(prefect_submit, "prefect_api_url")

    # Resolve remote $HOME to expand $HOME-based paths.
    home_marker = "__QMODEL_REMOTE_HOME__"
    home_cmd = f'printf "{home_marker}%s{home_marker}\\n" "$HOME"'
    home_out = _run_capture_logged(command=["ssh", remote_host, "bash", "-lc", shlex.quote(home_cmd)], cwd=str(Path.cwd()), logger=logger)
    remote_home = None
    for line in home_out.splitlines():
        if home_marker in line:
            remote_home = line.split(home_marker)[1]
            break
    if remote_home is None:
        raise RuntimeError(f"failed to parse remote $HOME from output: {home_out!r}")
    if remote_home == "":
        raise RuntimeError("empty remote $HOME")
    remote_experiment_root_abs = _resolve_home_path(remote_experiment_root, remote_home)
    remote_train_logs_root_abs = _resolve_home_path(remote_train_logs_root, remote_home)

    # Prepare local train_logs/remote target path for rsync.
    local_train_logs_remote_abs = str((Path(repo_path) / local_train_logs_remote_root).resolve())
    Path(local_train_logs_remote_abs).mkdir(parents=True, exist_ok=True)

    # Select which run directory the remote job should write into.
    target_run_id = run_id if mode == "train" else str(eval_run_id)
    remote_root_dir = f"{remote_train_logs_root_abs.rstrip('/')}/{target_run_id}"
    remote_tb_dir = f"{remote_home.rstrip('/')}/runs/{target_run_id}"

    # Always attempt rsync-back at the end (even when the remote run fails).
    primary_exc: BaseException | None = None
    rsync_exc: BaseException | None = None

    try:
        # Clone a fresh experiment snapshot locally and parse expm stdout for path/tag/hex.
        clone_out = _run_capture_logged(command=["expm", "clone", expm_tag], cwd=repo_path, logger=logger)
        clone_meta = _parse_expm_clone_output(clone_out)
        local_experiment_path = require_str(clone_meta, "path")
        tag = require_str(clone_meta, "tag")
        hex6 = require_str(clone_meta, "hex")
        mm_dd = Path(local_experiment_path).parts[-3]
        logger.info(f"local_experiment_path={local_experiment_path} tag={tag} hex={hex6} mm_dd={mm_dd}")

        # Copy orchestration.yaml into the experiment snapshot so remote qmodel-submit can read it.
        orch_src = Path(orchestration_path).resolve()
        orch_dst = Path(local_experiment_path) / "orchestration.yaml"
        shutil.copy2(orch_src, orch_dst)

        # Compute remote experiment path and ensure the directory exists.
        remote_experiment_path = f"{remote_experiment_root_abs.rstrip('/')}/{mm_dd}/{tag}/{hex6}"
        mkdir_script = f"mkdir -p {remote_experiment_path}"
        _run_logged(
            command=["ssh", remote_host, "bash", "-lc", shlex.quote(mkdir_script)],
            cwd=str(Path.cwd()),
            logger=logger,
        )

        # Rsync experiment code to remote (exclude soft-links and local artifacts).
        rsync_cmd = ["rsync", "-az"]
        for pat in rsync_excludes:
            rsync_cmd += ["--exclude", pat]
        rsync_cmd += [f"{local_experiment_path.rstrip('/')}/", f"{remote_host}:{remote_experiment_path.rstrip('/')}/"]
        _run_logged(command=rsync_cmd, cwd=str(Path.cwd()), logger=logger)

        # Optionally rebuild remote cache if marker is missing (cache may be wiped after reboot).
        if cache_check_enabled:
            script = "\n".join(
                [
                    f"cd {shlex.quote(remote_experiment_path)}",
                    "python - <<'PY'",
                    "from pathlib import Path",
                    "import pickle",
                    "from omegaconf import OmegaConf",
                    "import subprocess",
                    f"cfg = OmegaConf.load({cache_check_config_relpath!r})",
                    "cache_folder = str(cfg.data_spec.cache_folder)",
                    f"marker = Path(cache_folder) / {cache_check_marker_filename!r}",
                    "if not marker.exists():",
                    f"    subprocess.check_call({cache_check_rebuild_cmd!r})",
                    "else:",
                    "    if marker.name == 'meta.pkl':",
                    "        meta = pickle.load(marker.open('rb'))",
                    "        if 'dates' not in meta:",
                    f"            subprocess.check_call({cache_check_rebuild_cmd!r})",
                    "PY",
                ]
            )
            _run_logged(command=["ssh", remote_host, "bash", "-lc", shlex.quote(script)], cwd=str(Path.cwd()), logger=logger)

        # Submit the remote Prefect run and capture flow_run_id.
        submit_tokens = (
            ["qmodel-submit", conf_relpath, "orchestration.yaml"]
            + remote_submit_overrides
            + ["--solver.root_dir", remote_root_dir, "--solver.tensorboard_dir", remote_tb_dir]
        )
        submit_argv = " ".join(shlex.quote(tok) for tok in submit_tokens)
        submit_cmd = "\n".join(
            [
                f"cd {shlex.quote(remote_experiment_path)}",
                f"PREFECT_API_URL={shlex.quote(prefect_api_url)} {submit_argv}",
            ]
        )
        submit_out = _run_capture_logged(command=["ssh", remote_host, "bash", "-lc", shlex.quote(submit_cmd)], cwd=str(Path.cwd()), logger=logger)
        flow_run_id_matches = re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", submit_out)
        flow_run_id = flow_run_id_matches[-1] if flow_run_id_matches else ""
        if flow_run_id == "":
            raise RuntimeError("empty remote flow_run_id from qmodel-submit")
        logger.info(f"remote_flow_run_id={flow_run_id} remote_root_dir={remote_root_dir}")

        # Poll remote flow state until terminal, then raise on failure.
        poll_script = "\n".join(
            [
                f"PREFECT_API_URL={shlex.quote(prefect_api_url)} python - <<'PY'",
                "import time",
                "from prefect.client.orchestration import get_client",
                "from prefect.states import StateType",
                f"flow_run_id = {flow_run_id!r}",
                "with get_client(sync_client=True) as client:",
                "    while True:",
                "        fr = client.read_flow_run(flow_run_id)",
                "        st = fr.state.type if fr.state is not None else None",
                "        print(st)",
                "        if st in {StateType.COMPLETED, StateType.FAILED, StateType.CANCELLED, StateType.CRASHED}:",
                "            if st != StateType.COMPLETED:",
                "                raise RuntimeError(f'remote flow failed: {st}')",
                "            break",
                "        time.sleep(10)",
                "PY",
            ]
        )
        _run_logged(command=["ssh", remote_host, "bash", "-lc", shlex.quote(poll_script)], cwd=str(Path.cwd()), logger=logger)

    except BaseException as exc:
        # Preserve the primary failure while still performing rsync-back in finally.
        primary_exc = exc
    finally:
        # Always rsync remote train_logs back to local for debugging and artifacts.
        try:
            if rsync_back_mode == "all":
                rsync_back_cmd = [
                    "rsync",
                    "-az",
                    f"{remote_host}:{remote_train_logs_root_abs.rstrip('/')}/",
                    f"{local_train_logs_remote_abs.rstrip('/')}/",
                ]
                _run_logged(command=rsync_back_cmd, cwd=str(Path.cwd()), logger=logger)
            else:
                overrides = _parse_pairs(remote_submit_overrides)
                if "--group" not in overrides:
                    raise RuntimeError("eval-only rsync requires '--group test|val' in remote_orchestrate.submit_overrides")
                if "--eval.evaluator.eval_checkpoint_iter" not in overrides:
                    raise RuntimeError(
                        "eval-only rsync requires '--eval.evaluator.eval_checkpoint_iter \"[... ]\"' in remote_orchestrate.submit_overrides"
                    )
                group = overrides["--group"]
                eval_iters = _parse_eval_checkpoint_iter(overrides["--eval.evaluator.eval_checkpoint_iter"])

                jobs = _build_eval_only_rsync_jobs(
                    remote_root_dir=remote_root_dir,
                    local_train_logs_remote_abs=local_train_logs_remote_abs,
                    target_run_id=target_run_id,
                    group=group,
                    eval_checkpoint_iter=eval_iters,
                    extra_paths=rsync_back_extra_paths,
                )

                for src, dst in jobs:
                    Path(dst).mkdir(parents=True, exist_ok=True)
                    _run_logged(command=["rsync", "-az", f"{remote_host}:{src}", dst], cwd=str(Path.cwd()), logger=logger)
        except BaseException as exc:
            rsync_exc = exc

    # Raise failures after cleanup, without swallowing either error.
    if primary_exc is not None and rsync_exc is not None:
        raise RuntimeError(f"remote orchestration failed: {primary_exc}; rsync-back failed: {rsync_exc}") from primary_exc
    if primary_exc is not None:
        raise primary_exc
    if rsync_exc is not None:
        raise rsync_exc
