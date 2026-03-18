import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path


def merge_date_time_dataframe(df: pd.DataFrame, date_col="date", time_col="time") -> pd.Series:

    tmp_df = pl.from_pandas(df[[date_col, time_col]])

    if df[date_col].max() > 100000:
        tmp_df = tmp_df.lazy().select(
            pl.col(date_col).add(2*10**7).cast(pl.Int64).mul(10**6)
            .add(pl.col(time_col).cast(pl.Int64))
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime("ms"), "%Y%m%d%H%M%S")
            .alias("DateTime")
        ).collect()

    else:
        tmp_df = tmp_df.select(
            pl.col(date_col).cast(pl.Date).cast(pl.Datetime).add(
                pl.col(time_col).cast(pl.Int64).mul(1000).cast(pl.Duration("ms"))
            ).alias("DateTime")
        )
    return tmp_df.to_pandas().DateTime


def find_checkpoint_path(root_dir: Path, iteration: int | None) -> Path | None:
    """iteration: None means no checkpoint, -1 means latest checkpoint, other int means specific checkpoint.
    If not found, return None.
    """
    if iteration is None:
        return None

    ckpt_dir = root_dir / "ckpt"
    if not ckpt_dir.exists():
        return None

    ckpt_files = list(ckpt_dir.glob("iter_*.pt"))
    if not ckpt_files:
        return None

    if iteration == -1:
        latest_ckpt = max(ckpt_files, key=lambda x: int(x.stem.split("_")[1]))
        return latest_ckpt
    else:
        target_ckpt = ckpt_dir / f"iter_{iteration}.pt"
        if target_ckpt.exists():
            return target_ckpt
        return None
