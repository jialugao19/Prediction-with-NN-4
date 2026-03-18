import runpy
from typing import Any

from omegaconf import DictConfig, OmegaConf


TOP_KEYS = ["data", "model", "solver", "eval"]


def _require_dict(ns: dict[str, Any], key: str) -> dict[str, Any]:
    # Fetch a required dict variable from the executed config namespace.
    value = ns[key]
    if isinstance(value, dict):
        return value
    if isinstance(value, DictConfig):
        container = OmegaConf.to_container(value, resolve=False)
        if container is None or not isinstance(container, dict):
            raise RuntimeError(f"{key} must be a dict, got {type(container)}")
        return container
    raise RuntimeError(f"{key} must be a dict, got {type(value)}")


def _pairs_to_dotlist(tokens: list[str]) -> list[str]:
    # Convert `--key value` tokens into OmegaConf dotlist entries `key=value`.
    if len(tokens) % 2 != 0:
        raise RuntimeError("Overrides must be provided as --key value pairs")

    dotlist: list[str] = []
    for i in range(0, len(tokens), 2):
        key = tokens[i]
        value = tokens[i + 1]
        if not key.startswith("--"):
            raise RuntimeError(f"Override key must start with '--': {key}")
        dot_key = key[2:]
        dotlist.append(f"{dot_key}={value}")
    return dotlist


def load_cfg(
    *,
    conf_path: str,
    override_tokens: list[str],
) -> DictConfig:
    # Execute conf.py, build DictConfig for fixed top keys, then apply overrides.
    ns = runpy.run_path(conf_path, run_name="__qmodel_conf__")

    base = {k: _require_dict(ns, k) for k in TOP_KEYS}
    cfg = OmegaConf.create(base, flags={"allow_objects": True})

    dotlist = _pairs_to_dotlist(override_tokens)
    over = OmegaConf.from_dotlist(dotlist)
    return OmegaConf.merge(cfg, over)
