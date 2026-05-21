"""Build qmodel DataLoader instances with explicit throughput knobs."""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader

from qmodel.config import QConfig
from qmodel.data.sampler import CustomSampler


def custom_collate_fn(batch):
    """Return pre-batched dataset output without extra collation work."""
    # Stock1mNpzDataset implements __getitems__, so DataLoader receives a batch object directly.
    return batch


def setup_train_dataloader(config: QConfig, group: str, shuffle: bool = True) -> Tuple[DataLoader, CustomSampler]:
    """Create the train DataLoader and its resume-aware sampler."""
    # Build the split dataset with the configured training dtype.
    dataset = config.dataset_class(group, config.train_dtype)

    # Create the sampler before the loader so checkpoint resume can adjust it later.
    sampler = CustomSampler(dataset, seed=int(config.seed), shuffle=bool(shuffle), infinite=True)

    # Pass throughput-related DataLoader knobs explicitly from config.
    return DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.dataloader_pin_memory),
        prefetch_factor=int(config.dataloader_prefetch_factor),
        persistent_workers=bool(config.dataloader_persistent_workers),
    ), sampler


def setup_eval_dataloader(config: QConfig, group, shuffle: bool = False):
    """Create an eval DataLoader with the same worker pipeline settings."""
    # Build the split dataset with the configured eval dtype.
    dataset = config.dataset_class(group, config.eval_dtype)

    # Use finite sampling for eval and prediction passes.
    sampler = CustomSampler(dataset, seed=int(config.seed), shuffle=bool(shuffle), infinite=False)

    # Keep eval loader settings aligned with train to avoid a separate IO bottleneck.
    return DataLoader(
        dataset,
        batch_size=int(config.evaluator.eval_batch_size),
        sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.dataloader_pin_memory),
        prefetch_factor=int(config.dataloader_prefetch_factor),
        persistent_workers=bool(config.dataloader_persistent_workers),
    )
