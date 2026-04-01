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
    h1, h2 {{
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
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: rgba(24, 78, 119, 0.08);
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
