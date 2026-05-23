"""Export TensorBoard scalar events into stable tabular artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export_tensorboard_scalars(tb_dir: Path, run_root: Path, experiment_id: str) -> pd.DataFrame:
    """Export all TensorBoard scalar tags into a long-form DataFrame."""
    # Load scalar events from the TensorBoard directory.
    acc = EventAccumulator(Path(tb_dir).as_posix(), size_guidance={"scalars": 0})
    acc.Reload()

    # Convert scalar events into a stable long-form schema.
    rows: list[dict[str, object]] = []
    for tag in acc.Tags().get("scalars", []):
        for scalar in acc.Scalars(str(tag)):
            rows.append(
                {
                    "tag": str(tag),
                    "step": int(scalar.step),
                    "wall_time": float(scalar.wall_time),
                    "value": float(scalar.value),
                    "run_root": Path(run_root).as_posix(),
                    "experiment_id": str(experiment_id),
                }
            )
    return pd.DataFrame(rows)


def write_tensorboard_scalar_export(tb_dir: Path, run_root: Path, experiment_id: str, out_dir: Path) -> dict[str, object]:
    """Write TensorBoard scalar parquet and manifest artifacts."""
    # Export scalar rows into a parquet-friendly table.
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    scalar_df = export_tensorboard_scalars(Path(tb_dir), Path(run_root), str(experiment_id))
    scalar_parquet = out_path / "tensorboard_scalars.parquet"
    scalar_df.to_parquet(scalar_parquet, index=False)

    # Record source event files and scalar tags for report consumers.
    event_files = sorted(Path(tb_dir).glob("events.out.tfevents*"))
    manifest = {
        "source_dir": Path(tb_dir).as_posix(),
        "scalar_parquet": scalar_parquet.as_posix(),
        "scalar_rows": int(scalar_df.shape[0]),
        "tags": sorted(set(str(tag) for tag in scalar_df["tag"].tolist())) if int(scalar_df.shape[0]) > 0 else [],
        "event_files": [path.as_posix() for path in event_files],
        "run_root": Path(run_root).as_posix(),
        "experiment_id": str(experiment_id),
    }
    manifest_path = out_path / "tensorboard_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return manifest

