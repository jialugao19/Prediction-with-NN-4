import os

import torch

from qmodel.logger import logger


def _require_int_env(name: str) -> int:
    # Parse a required integer environment variable.
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Missing env var {name} for DDP")
    return int(value)


def get_dist_env() -> tuple[int, int, int]:
    # Read (rank, world_size, local_rank) from torchrun-style env vars.
    world_size_str = os.environ.get("WORLD_SIZE")
    if world_size_str is None:
        logger.warning("WORLD_SIZE env var is not set; treating as single-process non-DDP")
        return 0, 1, 0

    world_size = int(world_size_str)
    if world_size <= 1:
        logger.warning(f"WORLD_SIZE={world_size} is not >1; treating as single-process non-DDP")
        return 0, 1, 0

    rank = _require_int_env("RANK")
    local_rank = _require_int_env("LOCAL_RANK")
    logger.info(f"DDP env vars: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    return rank, world_size, local_rank


def init_process_group_from_env(*, backend: str | None) -> torch.device:
    """Initialize torch.distributed from torchrun env vars and return the selected device."""
    # Read DDP env vars and require a multi-process world.
    rank, world_size, local_rank = get_dist_env()
    if world_size == 1:
        raise RuntimeError("init_process_group_from_env requires WORLD_SIZE>1")

    # Validate torch.distributed availability and initialization state.
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available but WORLD_SIZE>1 was set")
    if torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed was already initialized")

    # Validate the requested backend string.
    use_cuda = torch.cuda.is_available()
    if backend is not None and backend not in ["gloo", "nccl"]:
        raise RuntimeError(f"dist_backend must be 'gloo' or 'nccl', got: {backend}")

    # Select a backend with a strict single-GPU multi-rank rule.
    explicit_backend = backend
    selected_backend = explicit_backend
    if selected_backend is None:
        selected_backend = "nccl" if use_cuda else "gloo"

    # Select device and handle local verification on single GPU.
    if use_cuda:
        n = torch.cuda.device_count()
        if n <= 0:
            raise RuntimeError("CUDA is available but device_count() returned 0")
        if n == 1 and world_size > 1:
            logger.warning(
                f"DDP with world_size={world_size} on a single GPU; all ranks will use cuda:0. "
                f"This is only for local verification and may OOM or be very slow."
            )
            if explicit_backend == "nccl":
                raise RuntimeError("dist_backend='nccl' is not supported for single-GPU multi-rank verification; use dist_backend='gloo'")
            if explicit_backend is None:
                logger.warning("Using gloo backend because NCCL rejects duplicate GPU across ranks.")
                selected_backend = "gloo"
        device_index = 0 if n == 1 else local_rank % n
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    else:
        device = torch.device("cpu")

    # Initialize the process group with env:// rendezvous.
    torch.distributed.init_process_group(backend=selected_backend, init_method="env://", rank=rank, world_size=world_size)
    logger.info(f"Initialized torch.distributed with backend={selected_backend}, rank={rank}, world_size={world_size}, local_rank={local_rank}, device={device}")

    return device


def get_ddp_state() -> tuple[bool, int, int, int]:
    """Load DDP runtime state and require init when WORLD_SIZE>1 is set."""
    # Read env vars and treat WORLD_SIZE<=1 as non-DDP.
    rank_env, world_size_env, local_rank_env = get_dist_env()
    if world_size_env == 1:
        return False, 0, 1, 0

    # Require an initialized torch.distributed process group for multi-rank runs.
    if not torch.distributed.is_available():
        raise RuntimeError("WORLD_SIZE>1 but torch.distributed is not available")
    if not torch.distributed.is_initialized():
        raise RuntimeError("WORLD_SIZE>1 but torch.distributed is not initialized; init in entry_main first")

    # Cross-check env values against torch.distributed runtime values.
    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    if rank != rank_env:
        raise RuntimeError(f"DDP rank mismatch: env rank={rank_env}, torch.distributed rank={rank}")
    if world_size != world_size_env:
        raise RuntimeError(f"DDP world_size mismatch: env world_size={world_size_env}, torch.distributed world_size={world_size}")

    return True, rank, world_size, local_rank_env


def maybe_init_process_group(*, backend: str | None) -> tuple[bool, int, int, int, torch.device]:
    # Initialize torch.distributed process group if WORLD_SIZE>1 is set.
    rank, world_size, local_rank = get_dist_env()
    if world_size == 1:
        device = torch.device("cuda")
        return False, 0, 1, 0, device

    # Delegate initialization to the shared env-based initializer.
    device = init_process_group_from_env(backend=backend)
    return True, rank, world_size, local_rank, device


def is_main_process() -> bool:
    # Return True if current process is rank 0 (or not in DDP).
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def barrier() -> None:
    # Synchronize all ranks if DDP is enabled.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def destroy_process_group() -> None:
    # Tear down torch.distributed process group if enabled.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
