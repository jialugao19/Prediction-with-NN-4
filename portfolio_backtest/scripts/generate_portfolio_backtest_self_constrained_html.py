"""Generate one self-contained HTML report from canonical portfolio_backtest outputs."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd
import yaml


def read_yaml(path: Path) -> dict:
    """Load one YAML file."""
    # Read the UTF-8 payload once.
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def image_data_uri(path: Path) -> str:
    """Encode one image as an inline data URI."""
    # Read the bytes and encode them in base64.
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return f"data:{mime};base64,{payload}"


def fmt_pct(value: float, digits: int) -> str:
    """Format one ratio as percentage."""
    # Return a stable label for missing metrics.
    if pd.isna(value):
        return "N/A"

    # Convert the ratio into percent units.
    return f"{float(value) * 100:.{digits}f}%"


def fmt_num(value: float, digits: int) -> str:
    """Format one scalar with fixed digits."""
    # Return a stable label for missing metrics.
    if pd.isna(value):
        return "N/A"

    # Convert the scalar into a readable string.
    return f"{float(value):.{digits}f}"


def fmt_money_m(value: float) -> str:
    """Format one AUM value in millions of CNY."""
    # Return a stable label for missing metrics.
    if pd.isna(value):
        return "N/A"

    # Convert the scalar into million-CNY units.
    return f"{float(value) / 1_000_000:.0f}M"


def kv_rows(items: list[tuple[str, str]]) -> str:
    """Render one key-value block."""
    # Serialize the rows in display order.
    rows: list[str] = []
    for key, value in items:
        rows.append(
            "<div class='kv-row'>"
            f"<div class='kv-key'>{html.escape(str(key))}</div>"
            f"<div class='kv-value'>{html.escape(str(value))}</div>"
            "</div>"
        )
    return "".join(rows)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render one plain HTML table."""
    # Render the table head first.
    thead = "<thead><tr>" + "".join(f"<th>{html.escape(col)}</th>" for col in headers) + "</tr></thead>"

    # Render the table body row by row.
    body_rows: list[str] = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return "<table>" + thead + tbody + "</table>"


def baseline_rows(summary: dict) -> list[list[str]]:
    """Build the baseline strategy table."""
    # Read the gross and turnover blocks.
    gross = dict(summary["gross"])
    turnover = dict(summary["turnover"])

    # Assemble the display rows.
    return [
        ["mean daily return", fmt_pct(gross["mean_daily_return"], 4)],
        ["std daily return", fmt_pct(gross["std_daily_return"], 4)],
        ["annualized return", fmt_pct(gross["annualized_return"], 2)],
        ["annualized Sharpe", fmt_num(gross["annualized_sharpe"], 3)],
        ["t-stat", fmt_num(gross["t_stat"], 2)],
        ["cumulative return", fmt_pct(gross["cum_return"], 2)],
        ["max drawdown", fmt_pct(gross["max_drawdown"], 2)],
        ["positive day ratio", fmt_pct(gross["positive_day_ratio"], 2)],
        ["mean daily turnover", fmt_num(turnover["mean_daily_turnover"], 4)],
    ]


def realistic_rows(summary: dict, aum_list: list[float]) -> tuple[list[str], list[list[str]]]:
    """Build the realistic strategy table."""
    # Collect the gross and net blocks in one display order.
    headers = ["metric", "Gross"]
    blocks = [("Gross", dict(summary["gross"]))]
    for aum in list(aum_list):
        key = f"net_{int(float(aum) / 1_000_000):d}m"
        headers.append(f"Net {fmt_money_m(float(aum))}")
        blocks.append((key, dict(summary[key])))

    # Build the metric rows.
    metric_keys = [
        ("mean daily return", "mean_daily_return", "pct", 4),
        ("std daily return", "std_daily_return", "pct", 4),
        ("annualized return", "annualized_return", "pct", 2),
        ("annualized Sharpe", "annualized_sharpe", "num", 3),
        ("t-stat", "t_stat", "num", 2),
        ("cumulative return", "cum_return", "pct", 2),
        ("max drawdown", "max_drawdown", "pct", 2),
        ("positive day ratio", "positive_day_ratio", "pct", 2),
    ]
    rows: list[list[str]] = []
    for metric_name, metric_key, metric_type, digits in list(metric_keys):
        row = [metric_name]
        for _, block in list(blocks):
            value = block[metric_key]
            if str(metric_type) == "pct":
                row.append(fmt_pct(value, digits))
            else:
                row.append(fmt_num(value, digits))
        rows.append(row)

    # Append turnover separately because it is AUM-independent.
    rows.append(["mean daily turnover", fmt_num(summary["turnover"]["mean_daily_turnover"], 4)] + ["-"] * (len(headers) - 2))
    return headers, rows


def slot_rows(slot_summary: pd.DataFrame) -> list[list[str]]:
    """Build one per-slot summary table."""
    # Sort the slots from 0 to 9.
    ordered = slot_summary.sort_values("minute_slot").reset_index(drop=True)

    # Serialize each slot row.
    rows: list[list[str]] = []
    for _, row in ordered.iterrows():
        rows.append(
            [
                str(int(row["minute_slot"])),
                fmt_pct(row["mean_daily_return"], 4),
                fmt_pct(row["std_daily_return"], 4),
                fmt_num(row["annualized_sharpe"], 3),
                str(int(row["day_count"])),
            ]
        )
    return rows


def capacity_rows(summary: dict) -> list[list[str]]:
    """Build the capacity table."""
    # Sort the impact-budget keys numerically.
    ordered = sorted(dict(summary["capacity"]).items(), key=lambda item: int(str(item[0]).split("_")[-1].replace("bps", "")))

    # Serialize each budget row.
    rows: list[list[str]] = []
    for key, value in ordered:
        rows.append([str(key), fmt_money_m(float(value))])
    return rows


def figure_block(path: Path, title: str, caption: str) -> str:
    """Render one inline figure block."""
    # Encode the local file into an embeddable data URI.
    uri = image_data_uri(path)

    # Assemble the HTML figure markup.
    return (
        "<figure class='figure'>"
        f"<img src='{uri}' alt='{html.escape(title)}' />"
        f"<figcaption><strong>{html.escape(title)}</strong><br>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def build_html(backtest_dir: Path) -> str:
    """Assemble the full self-contained HTML document."""
    # Load all structured inputs first.
    strategy_summary = read_yaml(backtest_dir / "strategy_summary.yaml")
    runtime_contract = read_yaml(backtest_dir / "portfolio_backtest_contract.yaml")
    research_md = (backtest_dir / "research_report.md").read_text(encoding="utf-8")
    baseline_slot_summary = pd.read_csv(backtest_dir / "baseline_open_slot_summary.csv")
    realistic_slot_summary = pd.read_csv(backtest_dir / "realistic_vwap_slot_summary.csv")

    # Build the summary tables and overview key-values.
    aum_list = list(strategy_summary["cost_model"]["aum_list"])
    baseline_table = baseline_rows(dict(strategy_summary["baseline_open"]))
    realistic_headers, realistic_table = realistic_rows(dict(strategy_summary["realistic_vwap"]), list(aum_list))
    baseline_slot_table = slot_rows(baseline_slot_summary)
    realistic_slot_table = slot_rows(realistic_slot_summary)
    capacity_table = capacity_rows(dict(strategy_summary["realistic_vwap"]))
    overview = [
        ("canonical_module", "portfolio_backtest"),
        ("report_scope", "0424 portfolio_backtest self-constrained report"),
        ("input_manifest_type", str(runtime_contract["input_contract"]["manifest_type"])),
        ("input_manifest_split", str(runtime_contract["input_contract"]["expected_inference_split"])),
        ("required_output_artifacts", str(len(list(runtime_contract["output_contract"]["required_artifacts"])))),
    ]

    # Escape the markdown appendix for inline display.
    appendix = html.escape(research_md)

    # Assemble the final HTML document.
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>0424 Portfolio Backtest Self-Constrained Report</title>
  <style>
    body {{
      margin: 0;
      background: #f4f1ea;
      color: #1f252b;
      font-family: "Georgia", "Times New Roman", serif;
      line-height: 1.6;
    }}
    .page {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 40px 28px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, #153243 0%, #284b63 100%);
      color: #f7f4ea;
      border-radius: 18px;
      padding: 28px 30px;
      box-shadow: 0 18px 45px rgba(21, 50, 67, 0.18);
    }}
    .hero h1 {{
      margin: 0;
      font-size: 36px;
    }}
    .subtitle {{
      margin-top: 10px;
      font-size: 16px;
      color: #d9e2ec;
    }}
    .section {{
      margin-top: 26px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(40, 75, 99, 0.10);
      border-radius: 16px;
      padding: 22px 22px 18px;
      box-shadow: 0 8px 24px rgba(31, 37, 43, 0.05);
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 24px;
    }}
    .grid2 {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .figure {{
      margin: 0;
      background: #fcfbf7;
      border: 1px solid rgba(40, 75, 99, 0.10);
      border-radius: 14px;
      padding: 14px;
    }}
    .figure img {{
      width: 100%;
      border-radius: 10px;
      display: block;
    }}
    .figure figcaption {{
      margin-top: 10px;
      font-size: 14px;
      color: #43515c;
    }}
    .kv {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px 20px;
    }}
    .kv-row {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px dashed rgba(67, 81, 92, 0.20);
      padding: 7px 0;
    }}
    .kv-key {{
      color: #52606d;
    }}
    .kv-value {{
      color: #102a43;
      text-align: right;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 14px;
      background: #fffdfa;
    }}
    th, td {{
      border-bottom: 1px solid rgba(67, 81, 92, 0.16);
      padding: 10px 12px;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      background: #e6ecf1;
      color: #102a43;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #fbfaf5;
      border: 1px solid rgba(40, 75, 99, 0.10);
      border-radius: 12px;
      padding: 16px;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>0424 Portfolio Backtest Report</h1>
      <div class="subtitle">Self-contained HTML report for the canonical <code>portfolio_backtest</code> output. All images are embedded inline.</div>
    </section>

    <section class="section">
      <h2>Overview</h2>
      <div class="kv">{kv_rows(overview)}</div>
    </section>

    <section class="section">
      <h2>Baseline Open Strategy</h2>
      {html_table(["metric", "value"], baseline_table)}
      <div class="grid2">
        {figure_block(backtest_dir / "baseline_open_strategy.png", "Baseline open strategy curves", "Gross wealth, turnover, and cost diagnostics for the baseline open strategy.")}
        {figure_block(backtest_dir / "baseline_open_slot_sharpe.png", "Baseline open slot Sharpe", "Annualized Sharpe by minute slot for the baseline open strategy.")}
      </div>
      {html_table(["minute slot", "mean daily return", "std daily return", "annualized Sharpe", "day_count"], baseline_slot_table)}
    </section>

    <section class="section">
      <h2>Realistic VWAP Strategy</h2>
      {html_table(realistic_headers, realistic_table)}
      <div class="grid2">
        {figure_block(backtest_dir / "strategy_curves.png", "Realistic strategy curves", "Gross and net wealth curves under multiple AUM assumptions.")}
        {figure_block(backtest_dir / "drawdown_curve.png", "Realistic strategy drawdown", "Drawdown curves for gross and net paths under multiple AUM assumptions.")}
      </div>
      <div class="grid2">
        {figure_block(backtest_dir / "slot_sharpe.png", "Realistic slot Sharpe", "Annualized Sharpe by minute slot for the realistic VWAP strategy.")}
        {figure_block(backtest_dir / "capacity_sweep.png", "Capacity sweep", "Capacity diagnostics under multiple impact-budget assumptions.")}
      </div>
      {html_table(["minute slot", "mean daily return", "std daily return", "annualized Sharpe", "day_count"], realistic_slot_table)}
      {html_table(["impact budget", "estimated capacity"], capacity_table)}
    </section>

    <section class="section">
      <h2>Appendix</h2>
      <pre>{appendix}</pre>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    """Generate the HTML report into /data-cache/nn/0424."""
    # Resolve the fixed IO paths once.
    report_dir = Path("/data-cache/nn/0424")
    backtest_dir = Path("/data-cache/nn/0424/portfolio_backtest")
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build the HTML payload and write it once.
    html_text = build_html(backtest_dir)
    out_path = report_dir / "portfolio_backtest_report_0424_self_constrained.html"
    out_path.write_text(html_text, encoding="utf-8")
    print(out_path.as_posix())


if __name__ == "__main__":
    main()
