from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


def load_orchestration_yaml(path: str) -> dict[str, Any]:
    """Load orchestration YAML into a plain python dict."""
    # Load yaml via OmegaConf to keep dependencies minimal and behavior consistent.
    cfg = OmegaConf.load(path)

    # Convert OmegaConf nodes into plain python containers.
    container = OmegaConf.to_container(cfg, resolve=False)
    if container is None or not isinstance(container, dict):
        raise RuntimeError("orchestration yaml must be a dict at top-level")
    return container


def require_dict(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """Fetch a required dict section from a config dict."""
    # Read the required section and fail fast on missing keys.
    if key not in cfg:
        raise RuntimeError(f"missing required section: {key}")
    value = cfg[key]
    if not isinstance(value, dict):
        raise RuntimeError(f"section must be a dict: {key} (got {type(value)})")
    return value


def require_str(cfg: dict[str, Any], key: str) -> str:
    """Fetch a required non-empty string field from a config dict."""
    # Read the required field and fail fast on missing keys.
    if key not in cfg:
        raise RuntimeError(f"missing required field: {key}")
    value = cfg[key]
    if not isinstance(value, str):
        raise RuntimeError(f"field must be a string: {key} (got {type(value)})")
    if value == "":
        raise RuntimeError(f"field must be non-empty: {key}")
    return value


def require_bool(cfg: dict[str, Any], key: str) -> bool:
    """Fetch a required bool field from a config dict."""
    # Read the required field and fail fast on missing keys.
    if key not in cfg:
        raise RuntimeError(f"missing required field: {key}")
    value = cfg[key]
    if not isinstance(value, bool):
        raise RuntimeError(f"field must be a bool: {key} (got {type(value)})")
    return value


def require_list_str(cfg: dict[str, Any], key: str) -> list[str]:
    """Fetch a required list[str] field from a config dict."""
    # Read the required field and fail fast on missing keys.
    if key not in cfg:
        raise RuntimeError(f"missing required field: {key}")
    value = cfg[key]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RuntimeError(f"field must be list[str]: {key} (got {type(value)})")
    return value
