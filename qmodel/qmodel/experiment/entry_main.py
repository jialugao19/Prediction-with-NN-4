import argparse
import os
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf

from qmodel.core.evaluator import Evaluator
from qmodel.core.cpu_evaluator import CpuEvaluator
from qmodel.core.trainer import Trainer
from qmodel.core.cpu_trainer import CpuTrainer
from qmodel.distributed import barrier, destroy_process_group, get_ddp_state, get_dist_env, init_process_group_from_env
from qmodel.experiment.conf_runtime import TOP_KEYS, load_cfg
from qmodel.logger import logger


def _flatten_sections(cfg) -> dict:
    # Merge data/model/solver/eval sections into a single flat config dict.
    sections = ["data", "model", "solver", "eval"]

    flat: dict = {}
    for section in sections:
        node = getattr(cfg, section)
        if node is None:
            raise RuntimeError(f"Missing config section: {section}")
        if not hasattr(node, "items"):
            raise RuntimeError(f"Config section must be dict-like: {section} (got {type(node)})")

        for key, value in node.items():
            if key in flat:
                raise RuntimeError(f"Config key conflict while flattening: {key}")
            flat[key] = value
    return flat


def _copy_conf_files(*, conf_dir: Path, root_dir: Path, conf_files: list[str]) -> None:
    # Copy reproducibility artifacts from the conf directory into the run folder.
    root_dir.mkdir(parents=True, exist_ok=True)
    for name in conf_files:
        shutil.copy2(conf_dir / name, root_dir / name)


def main(argv: list[str] | None = None) -> None:
    # Load config from conf.py, apply overrides, then run train/eval/predict.
    parser = argparse.ArgumentParser(prog="entry.py")
    parser.add_argument("conf_path", type=str)
    parser.add_argument("--group", type=str, choices=["train", "test", "predict"], default="train")

    ns, unknown = parser.parse_known_args(argv)

    # Load and validate config sections from the conf.py entrypoint.
    cfg = load_cfg(conf_path=ns.conf_path, override_tokens=unknown)
    for k in TOP_KEYS:
        if not hasattr(cfg, k):
            raise RuntimeError(f"Missing top-level config section: {k}")

    # Flatten nested OmegaConf sections into a single namespace config object.
    flat = _flatten_sections(cfg)
    config_obj = SimpleNamespace(**flat)

    # Create run directories early to keep downstream code simple.
    root_dir = Path(config_obj.root_dir)
    tensorboard_dir = Path(config_obj.tensorboard_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # Configure logger outputs for reproducibility and debugging.
    logger.add_file_log(str(root_dir / "logs.log"))

    # Seed python/numpy/torch RNGs for deterministic behavior.
    torch.manual_seed(config_obj.seed)
    np.random.seed(config_obj.seed)
    random.seed(config_obj.seed)

    # Initialize torch.distributed if torchrun env vars indicate a multi-rank run.
    _, world_size, _ = get_dist_env()
    if world_size > 1:
        if not hasattr(config_obj, "dist_backend"):
            raise RuntimeError("Missing required config field for DDP: dist_backend")
        config_obj.device = init_process_group_from_env(backend=config_obj.dist_backend)
    ddp_enabled, rank, world_size, _ = get_ddp_state()

    # Copy user-provided conf artifacts into the run directory if configured.
    conf_files = cfg.solver.get("conf_files")
    if conf_files is not None:
        if isinstance(conf_files, ListConfig):
            conf_files = list(conf_files)
        if not isinstance(conf_files, list):
            raise RuntimeError("solver.conf_files must be a list[str]")
        conf_dir = Path(ns.conf_path).resolve().parent
        _copy_conf_files(conf_dir=conf_dir, root_dir=root_dir, conf_files=conf_files)

    # Execute the selected group with a strict distributed cleanup in finally.
    try:
        if ns.group == "train":
            device = torch.device(config_obj.device)
            trainer = CpuTrainer(config_obj) if device.type == "cpu" else Trainer(config_obj)
            trainer.train()
            return

        enable_logging = (not ddp_enabled) or (rank == 0)
        device = torch.device(config_obj.device)
        eval_cls = CpuEvaluator if device.type == "cpu" else Evaluator
        evaluator = eval_cls(config_obj, group=ns.group, writer=None, enable_logging=enable_logging)
        evaluator.evaluate()
    finally:
        if world_size > 1:
            barrier()
            destroy_process_group()
