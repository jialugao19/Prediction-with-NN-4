"""Provide small helpers for self-contained single-column HTML reports."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import yaml


def file_to_data_uri(path: Path) -> str:
    """Encode one existing file as an inline PNG data URI."""
    # Read the binary payload and base64-encode it for HTML embedding.
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def render_value_rows(rows: list[tuple[str, str]]) -> str:
    """Render key-value rows as a single-column stacked HTML block."""
    # Convert each pair into one full-width row so the layout stays vertical.
    items: list[str] = []
    for key, value in list(rows):
        items.append(
            f"""
            <div class="kv-row">
              <div class="kv-key">{html.escape(str(key))}</div>
              <div class="kv-value">{html.escape(str(value))}</div>
            </div>
            """
        )
    return '<div class="kv-list">' + "\n".join(items) + "</div>"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render one HTML table with escaped header and cell values."""
    # Build the table head first so column order stays explicit.
    head_html = "".join([f"<th>{html.escape(str(header))}</th>" for header in list(headers)])

    # Build the body row-by-row to keep formatting stable.
    body_rows: list[str] = []
    for row in list(rows):
        cell_html = "".join([f"<td>{html.escape(str(cell))}</td>" for cell in list(row)])
        body_rows.append(f"<tr>{cell_html}</tr>")
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_html_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render one HTML table with already-rendered cell fragments."""
    # Escape headers while allowing trusted cell fragments from local report code.
    head_html = "".join([f"<th>{html.escape(str(header))}</th>" for header in list(headers)])

    # Build each body row without escaping cells again.
    body_rows: list[str] = []
    for row in list(rows):
        cell_html = "".join([f"<td>{str(cell)}</td>" for cell in list(row)])
        body_rows.append(f"<tr>{cell_html}</tr>")
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_code_block(text: str) -> str:
    """Render one escaped preformatted text block."""
    # Escape the full payload once so YAML and paths stay readable in HTML.
    return f"<pre><code>{html.escape(str(text))}</code></pre>"


def render_yaml_block(data: dict[str, object]) -> str:
    """Render one YAML mapping inside a preformatted code block."""
    # Serialize YAML with stable key order preserved from the caller.
    return render_code_block(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def render_figure(title: str, path: Path, caption: str) -> str:
    """Render one full-width self-contained figure block."""
    # Reuse the embedded figure block so section and inline rendering stay consistent.
    return render_section(str(title), render_embedded_figure(str(title), Path(path), str(caption)))


def render_embedded_figure(title: str, path: Path, caption: str) -> str:
    """Render one self-contained figure block without creating a section."""
    # Inline the referenced image so the HTML stays single-file.
    data_uri = file_to_data_uri(Path(path))
    return f"""
    <div class="figure">
      <img src="{data_uri}" alt="{html.escape(str(title))}" />
      <div class="caption">{html.escape(str(caption))}</div>
    </div>
    """


def render_section(title: str, body: str) -> str:
    """Wrap one HTML fragment inside a titled full-width section."""
    # Keep every section in the same single-column page flow.
    return f"""
    <section class="section">
      <h2>{html.escape(str(title))}</h2>
      {body}
    </section>
    """


def render_subsection(title: str, body: str) -> str:
    """Wrap one HTML fragment inside a titled subsection."""
    # Keep h3 blocks nested inside their parent section without adding a new page break.
    return f"""
    <section class="subsection">
      <h3>{html.escape(str(title))}</h3>
      {body}
    </section>
    """


def render_block_title(title: str) -> str:
    """Render one compact local block title."""
    # Use h4 for table groups and figure groups inside a subsection.
    return f'<h4 class="block-title">{html.escape(str(title))}</h4>'


def build_page(title: str, subtitle: str, sections: list[str]) -> str:
    """Assemble one self-contained single-column HTML page."""
    # Build the page shell with a strictly vertical layout and simple typography.
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(str(title))}</title>
  <style>
    :root {{
      --bg: #f5efe6;
      --paper: #fffaf3;
      --ink: #1f2933;
      --muted: #5b6570;
      --line: #d7cab5;
      --accent: #184e77;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f7f1e8 0%, #f1e9dd 100%);
      color: var(--ink);
      font-family: "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      line-height: 1.7;
    }}
    .page {{
      width: min(1180px, calc(100vw - 32px));
      margin: 18px auto 36px auto;
      padding: 26px 28px 36px 28px;
      background: var(--paper);
      border: 1px solid var(--line);
      box-shadow: 0 18px 50px rgba(31, 41, 51, 0.10);
    }}
    h1, h2, h3, h4 {{
      font-family: Georgia, "Times New Roman", serif;
      color: #17222e;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 38px;
    }}
    h2 {{
      margin: 0 0 12px 0;
      font-size: 26px;
    }}
    h3 {{
      margin: 18px 0 10px 0;
      font-size: 21px;
    }}
    h4 {{
      margin: 14px 0 6px 0;
      font-size: 16px;
    }}
    .subtitle {{
      margin-bottom: 18px;
      color: var(--muted);
      font-size: 16px;
    }}
    .section {{
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }}
    .subsection {{
      margin-top: 18px;
    }}
    .block-title {{
      color: var(--accent);
      font-family: "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      font-weight: 700;
    }}
    .section-label {{
      margin: 0 0 8px 0;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .summary-card {{
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.78);
    }}
    .summary-card-key {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .summary-card-value {{
      margin-top: 4px;
      font-size: 22px;
      font-weight: 700;
      line-height: 1.25;
    }}
    .summary-card-note {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .takeaways {{
      margin: 12px 0 0 0;
      padding: 0;
      list-style: none;
    }}
    .takeaways li {{
      margin: 7px 0;
      padding: 9px 11px;
      border-left: 4px solid var(--accent);
      background: rgba(24, 78, 119, 0.06);
    }}
    .badge {{
      display: inline-block;
      min-width: 54px;
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.35;
      text-align: center;
      white-space: nowrap;
    }}
    .badge-good {{
      color: #842029;
      background: #f7d7da;
      border-color: #e2a6ad;
    }}
    .badge-watch {{
      color: #664d03;
      background: #fff0bf;
      border-color: #e3c96a;
    }}
    .badge-bad {{
      color: #0f5132;
      background: #d9f0e3;
      border-color: #a6d5ba;
    }}
    .badge-neutral {{
      color: #334155;
      background: #e8edf3;
      border-color: #c7d1df;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
    }}
    .details {{
      margin-top: 12px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.58);
    }}
    .details summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
    }}
    .field-notes {{
      margin-top: 12px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.70);
    }}
    .field-notes-title {{
      margin: 0 0 8px 0;
      color: var(--accent);
      font-size: 15px;
      font-weight: 700;
    }}
    .field-note {{
      margin: 8px 0;
      color: var(--ink);
      font-size: 14px;
      line-height: 1.65;
    }}
    .field-note strong {{
      color: #17222e;
    }}
    .kv-list {{
      display: block;
    }}
    .kv-row {{
      display: block;
      width: 100%;
      margin-bottom: 10px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.76);
    }}
    .kv-key {{
      color: var(--muted);
      font-size: 13px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .kv-value {{
      margin-top: 4px;
      font-size: 16px;
      font-weight: 600;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .figure {{
      padding: 12px;
      border: 1px solid var(--line);
      background: #fffdf9;
    }}
    .figure img {{
      width: 100%;
      display: block;
    }}
    .caption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 0 0;
      font-size: 14px;
    }}
    .table-wrap table {{
      min-width: 720px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }}
    th {{
      background: rgba(24, 78, 119, 0.08);
      position: sticky;
      top: 0;
    }}
    pre {{
      margin: 10px 0 0 0;
      padding: 14px;
      overflow-x: auto;
      background: #1f2933;
      color: #eef4f8;
      border-radius: 4px;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1>{html.escape(str(title))}</h1>
    <div class="subtitle">{html.escape(str(subtitle))}</div>
    {"".join(list(sections))}
  </main>
</body>
</html>
"""
