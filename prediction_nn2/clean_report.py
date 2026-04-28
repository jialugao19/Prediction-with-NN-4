"""Render the data-clean self-contained HTML report from persisted artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prediction_nn2.data_prep import _write_feature_distribution_artifacts_from_bins
from prediction_nn2.html_report import build_page, render_figure, render_section, render_table, render_value_rows


def render_clean_report_from_meta(meta_path: Path) -> Path:
    """Render `data_clean/report.html` from an existing `meta.yaml` and data-clean artifacts."""
    # Resolve artifact paths relative to meta.yaml so callers can move run roots freely.
    meta_path = Path(meta_path)
    import yaml

    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    out_dir = meta_path.parent.parent
    stats_dir = Path(out_dir) / "data_clean"
    report_path = Path(stats_dir) / "report.html"

    # Load the required lightweight artifacts produced during data prep.
    invalid_stats_path = Path(stats_dir) / "invalid_feature_stats.csv"
    invalid_report_path = Path(stats_dir) / "invalid_feature_report.html"
    moment_path = Path(stats_dir) / "feature_moments.csv"
    overview_png = Path(stats_dir) / "pooled_feature_grid.png"
    pooled_zscore_yaml = Path(stats_dir) / "pooled_zscore.yaml"
    invalid_stats = pd.read_csv(invalid_stats_path)

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
    invalid_sorted = invalid_stats.sort_values(["invalid_ratio", "invalid_count", "field"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
    max_abs_mean = float(moments["mean"].abs().max())
    max_abs_std_shift = float((moments["std"] - 1.0).abs().max())
    max_abs_skew = float(moments["skew"].abs().max())
    max_abs_kurt = float(moments["kurtosis"].abs().max())

    # Build the scalar summary blocks used by the HTML page.
    dates = dict(meta["dates"])
    audit = dict(meta["audit"])
    audit_rates = dict(meta["audit_rates"])
    groups = dict(meta["storage"]["groups"])
    norm = dict(meta["feature_transform"]["stock_norm"])
    summary_rows = [
        ("meta", meta_path.as_posix()),
        ("stock_norm", f"{norm['type']} / scope={norm.get('scope', 'n/a')}"),
        ("groups", ", ".join(sorted(list(groups.keys())))),
        ("train_days", str(len(dates["train"]))),
        ("val_days", str(len(dates["val"]))),
        ("test_days", str(len(dates["test"]))),
        ("train_rows", str(int(groups["train"]["rows"]))),
        ("val_rows", str(int(groups["val"]["rows"]))),
        ("test_rows", str(int(groups["test"]["rows"]))),
    ]
    for group_name in ["inference_train", "inference_val", "inference_test"]:
        if group_name in groups:
            summary_rows.append((f"{group_name}_rows", str(int(groups[group_name]["rows"]))))

    # Build the audit-rate rows as one vertical block.
    audit_rows = [
        ("train", f"kept_rate={float(audit_rates['train']['kept_rate']):.2%}, sampled_rate_vs_raw={float(audit_rates['train']['sampled_rate_vs_raw']):.2%}"),
        ("val", f"kept_rate={float(audit_rates['val']['kept_rate']):.2%}, sampled_rate_vs_raw={float(audit_rates['val']['sampled_rate_vs_raw']):.2%}"),
        ("test", f"kept_rate={float(audit_rates['test']['kept_rate']):.2%}, sampled_rate_vs_raw={float(audit_rates['test']['sampled_rate_vs_raw']):.2%}"),
    ]

    # Convert the invalid-value top table into HTML rows.
    invalid_table = render_table(
        ["field", "field_type", "invalid_ratio", "invalid_count", "total_count"],
        [
            [
                str(row["field"]),
                str(row["field_type"]),
                f"{float(row['invalid_ratio']):.4%}",
                str(int(row["invalid_count"])),
                str(int(row["total_count"])),
            ]
            for row in invalid_sorted.to_dict(orient="records")
        ],
    )

    # Convert the standardized-moment summary into stacked rows.
    moment_rows = [
        ("moments", moment_path.as_posix()),
        ("pooled_zscore", pooled_zscore_yaml.as_posix() if pooled_zscore_yaml.exists() else "(missing)"),
        ("max_abs_mean", f"{max_abs_mean:.6f}"),
        ("max_abs_std_shift", f"{max_abs_std_shift:.6f}"),
        ("max_abs_skew", f"{max_abs_skew:.6f}"),
        ("max_abs_kurtosis", f"{max_abs_kurt:.6f}"),
    ]

    # Build the final HTML document in a single vertical column.
    sections = [
        render_section("Summary", render_value_rows(summary_rows)),
        render_section("Audit Rates", render_value_rows(audit_rows)),
        render_section(
            "Invalid Values",
            render_value_rows([("stats_csv", invalid_stats_path.as_posix()), ("invalid_report_html", invalid_report_path.as_posix())]) + invalid_table,
        ),
        render_section("Standardized Feature Moments", render_value_rows(moment_rows)),
        render_figure("Pooled Distribution Overview", overview_png, "Data clean pooled feature distributions."),
        render_section(
            "Notes",
            render_value_rows(
                [
                    (
                        "audit_elapsed_seconds",
                        f"train={float(audit['train']['elapsed_seconds']):.2f}, val={float(audit['val']['elapsed_seconds']):.2f}, test={float(audit['test']['elapsed_seconds']):.2f}",
                    )
                ]
            ),
        ),
    ]

    # Persist the report next to the data-clean artifacts so downstream stages can rebuild cheaply.
    html = build_page("Data Clean Report", "Self-contained HTML generated from persisted data-clean artifacts.", sections)
    report_path.write_text(html, encoding="utf-8")
    return report_path
