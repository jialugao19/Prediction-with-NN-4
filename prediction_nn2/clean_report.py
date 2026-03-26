"""Render the data-clean markdown report from persisted artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prediction_nn2.data_prep import _write_feature_distribution_artifacts_from_bins


def render_clean_report_from_meta(meta_path: Path) -> Path:
    """Render `data_clean/report.md` from an existing `meta.yaml` and data-clean artifacts."""
    # Resolve artifact paths relative to meta.yaml so callers can move run roots freely.
    meta_path = Path(meta_path)
    import yaml

    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    out_dir = meta_path.parent.parent
    stats_dir = Path(out_dir) / "data_clean"
    report_path = Path(stats_dir) / "report.md"

    # Load the required lightweight artifacts produced during data prep.
    invalid_stats_path = Path(stats_dir) / "invalid_feature_stats.csv"
    invalid_report_path = Path(stats_dir) / "invalid_feature_report.md"
    moment_path = Path(stats_dir) / "feature_moments.csv"
    overview_png = Path(stats_dir) / "pooled_feature_grid.png"
    pooled_zscore_yaml = Path(stats_dir) / "pooled_zscore.yaml"
    invalid_stats = pd.read_csv(invalid_stats_path)
    invalid_md = invalid_report_path.read_text(encoding="utf-8")

    # Rebuild distribution artifacts from existing bins when they are missing.
    if moment_path.exists() and overview_png.exists():
        moments = pd.read_csv(moment_path)
    else:
        # Resolve the scope into binary feature matrices referenced by meta.yaml.
        scope = str(meta["feature_transform"]["stock_norm"].get("scope", "train_val_test"))
        groups = dict(meta["storage"]["groups"])
        npz_dir = meta_path.parent
        if scope == "train_only":
            x_paths = [npz_dir / str(groups["train"]["x"])]
            rows_list = [int(groups["train"]["rows"])]
        else:
            x_paths = [npz_dir / str(groups["train"]["x"]), npz_dir / str(groups["val"]["x"]), npz_dir / str(groups["test"]["x"])]
            rows_list = [int(groups["train"]["rows"]), int(groups["val"]["rows"]), int(groups["test"]["rows"])]
        feature_names = list(meta["feature_names"])
        moments = _write_feature_distribution_artifacts_from_bins(list(x_paths), list(rows_list), int(len(feature_names)), list(feature_names), stats_dir)

    # Summarize invalid-value and standardized-moment diagnostics into compact scalars.
    invalid_stats = invalid_stats.copy()
    invalid_stats["invalid_ratio"] = invalid_stats["invalid_count"] / invalid_stats["total_count"]
    top_invalid = invalid_stats.sort_values(["invalid_ratio", "invalid_count"], ascending=False).head(10).reset_index(drop=True)
    max_abs_mean = float(moments["mean"].abs().max())
    max_abs_std_shift = float((moments["std"] - 1.0).abs().max())
    max_abs_skew = float(moments["skew"].abs().max())
    max_abs_kurt = float(moments["kurtosis"].abs().max())

    # Render a single markdown report that links to the heavier numerical artifacts.
    dates = dict(meta["dates"])
    audit = dict(meta["audit"])
    audit_rates = dict(meta["audit_rates"])
    groups = dict(meta["storage"]["groups"])
    norm = dict(meta["feature_transform"]["stock_norm"])
    lines: list[str] = []
    lines.extend(
        [
            "# Data Clean Report",
            "",
            "## Summary",
            "",
            f"- meta: `{meta_path.as_posix()}`",
            f"- stock_norm: `{norm['type']}` scope=`{norm.get('scope', 'n/a')}`",
            f"- groups: {', '.join(sorted(list(groups.keys())))}",
            f"- train_days={len(dates['train'])}, val_days={len(dates['val'])}, test_days={len(dates['test'])}",
            f"- rows: train={int(groups['train']['rows'])}, val={int(groups['val']['rows'])}, test={int(groups['test']['rows'])}",
        ]
    )
    if "predict" in groups:
        # Record optional predict group stats when it exists.
        lines.append(f"- rows: predict={int(groups['predict']['rows'])}")
    lines.extend(
        [
            "",
            "## Audit Rates",
            "",
            f"- train kept_rate={float(audit_rates['train']['kept_rate']):.2%}, sampled_rate_vs_raw={float(audit_rates['train']['sampled_rate_vs_raw']):.2%}",
            f"- val kept_rate={float(audit_rates['val']['kept_rate']):.2%}, sampled_rate_vs_raw={float(audit_rates['val']['sampled_rate_vs_raw']):.2%}",
            f"- test kept_rate={float(audit_rates['test']['kept_rate']):.2%}, sampled_rate_vs_raw={float(audit_rates['test']['sampled_rate_vs_raw']):.2%}",
            "",
            "## Invalid Values (Top 10 by ratio)",
            "",
            f"- stats: `{invalid_stats_path.as_posix()}`",
            f"- report: `{invalid_report_path.as_posix()}`",
            "",
            "| field | field_type | invalid_ratio | invalid_count | total_count |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in top_invalid.to_dict(orient="records"):
        # Append a compact invalid-value table row per field.
        lines.append(
            f"| {row['field']} | {row['field_type']} | {float(row['invalid_ratio']):.4%} | {int(row['invalid_count'])} | {int(row['total_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Standardized Feature Moments",
            "",
            f"- moments: `{moment_path.as_posix()}`",
            f"- pooled_zscore: `{pooled_zscore_yaml.as_posix()}`" if pooled_zscore_yaml.exists() else "- pooled_zscore: (missing)",
            f"- Max abs mean: {max_abs_mean:.6f}",
            f"- Max abs std shift from 1: {max_abs_std_shift:.6f}",
            f"- Max abs skew: {max_abs_skew:.6f}",
            f"- Max abs kurtosis: {max_abs_kurt:.6f}",
            "",
            "## Pooled Distribution Overview",
            "",
            f"- overview: `{overview_png.as_posix()}`",
            "",
            f"![]({overview_png.name})",
            "",
            "## Notes",
            "",
            f"- audit_elapsed_seconds: train={float(audit['train']['elapsed_seconds']):.2f}, val={float(audit['val']['elapsed_seconds']):.2f}, test={float(audit['test']['elapsed_seconds']):.2f}",
            "",
            "## Invalid Feature Report (Raw)",
            "",
        ]
    )
    lines.append(invalid_md.rstrip("\n"))
    lines.append("")

    # Persist the report next to the data-clean artifacts so downstream stages can rebuild cheaply.
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
