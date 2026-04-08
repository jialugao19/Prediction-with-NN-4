"""Render the self-contained tiny-overfit HTML report."""

from __future__ import annotations

import base64
from pathlib import Path


def _png_data_uri(path: Path) -> str:
    """Embed one PNG file as a data URI for the self-contained HTML report."""
    # Read the PNG bytes and encode them into a stable data URI string.
    png_bytes = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def render_tiny_overfit_report(summary: dict[str, object], train_loss_png: Path, checkpoint_png: Path) -> str:
    """Render a self-contained HTML report for the tiny overfit test."""
    # Format the checkpoint table rows once for direct HTML interpolation.
    metric_rows = list(summary["checkpoint_rows"])
    table_rows = "\n".join(
        [
            "<tr>"
            f"<td>{int(row['iter'])}</td>"
            f"<td>{float(row['train_mse']):.8e}</td>"
            f"<td>{float(row['train_ic']):.6f}</td>"
            f"<td>{float(row['val_mse']):.8e}</td>"
            f"<td>{float(row['val_ic']):.6f}</td>"
            "</tr>"
            for row in list(metric_rows)
        ]
    )

    # Pull the headline numbers into shorter local names.
    train_loss = dict(summary["train_loss"])
    checkpoint_eval = dict(summary["checkpoint_eval"])
    subset = dict(summary["subset"])
    train_group = dict(subset["groups"]["train"])
    val_group = dict(subset["groups"]["val"])
    test_group = dict(subset["groups"]["test"])

    # Build the self-contained HTML body with embedded figures.
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NN Tiny Overfit Report</title>
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
    h1 {{ margin: 0 0 8px 0; font-size: 38px; }}
    h2 {{ margin: 0 0 12px 0; font-size: 26px; }}
    .subtitle {{ margin-bottom: 18px; color: var(--muted); font-size: 16px; }}
    .section {{ margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--line); }}
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
    .caption {{ margin-top: 10px; color: var(--muted); font-size: 14px; }}
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
    th {{ background: rgba(24, 78, 119, 0.08); }}
  </style>
</head>
<body>
  <main class="page">
    <h1>NN Tiny Overfit Report</h1>
    <div class="subtitle">小样本过拟合诊断. 目的不是估计泛化, 而是验证当前 NN 是否具备把几千样本训练集明显拟合下去的能力.</div>

    <section class="section">
      <h2>实验结论</h2>
      <div class="kv-row"><div class="kv-key">结论</div><div class="kv-value">当前 GRU+MLP 模型在 2,576 个 train windows 的小样本设置下{"可以" if float(train_loss["drop_ratio"]) > 0.5 else "未明显"}把 train loss 打下去. raw train loss 从 {float(train_loss["first_loss"]):.6f} 降到 {float(train_loss["last_loss"]):.6f}, 降幅 {float(train_loss["drop_ratio"]) * 100:.2f}%.</div></div>
      <div class="kv-row"><div class="kv-key">Train Eval MSE</div><div class="kv-value">ckpt0: {float(checkpoint_eval["first"]["train_mse"]):.8e} -> 最后 ckpt: {float(checkpoint_eval["last"]["train_mse"]):.8e}</div></div>
      <div class="kv-row"><div class="kv-key">Val Eval MSE</div><div class="kv-value">ckpt0: {float(checkpoint_eval["first"]["val_mse"]):.8e} -> 最后 ckpt: {float(checkpoint_eval["last"]["val_mse"]):.8e}</div></div>
      <div class="kv-row"><div class="kv-key">解读</div><div class="kv-value">如果 train loss 和 train MSE 明显下降, 而 val MSE 没有同步下降, 则说明模型容量和训练代码本身没有被根本卡死, 更可能是正式大样本任务受限于信号弱、目标噪声大、特征信息不足或优化设置不匹配.</div></div>
    </section>

    <section class="section">
      <h2>数据子集</h2>
      <div class="kv-row"><div class="kv-key">source_out_root</div><div class="kv-value">{subset["source_out_root"]}</div></div>
      <div class="kv-row"><div class="kv-key">subset_rule</div><div class="kv-value">从 source train split 的第一个 trade date 扫描完整 stock-day run. 每个 run 为 220 条原始分钟行, 对应 161 个 window=60 的有效样本. train/val/test 分别取前 16/4/4 个 stock-day runs, 彼此按 stock code 切开.</div></div>
      <div class="kv-row"><div class="kv-key">train_group</div><div class="kv-value">date={train_group["date"]}, runs={train_group["run_count"]}, raw_rows={train_group["row_count"]}, valid_windows={train_group["valid_windows"]}</div></div>
      <div class="kv-row"><div class="kv-key">val_group</div><div class="kv-value">date={val_group["date"]}, runs={val_group["run_count"]}, raw_rows={val_group["row_count"]}, valid_windows={val_group["valid_windows"]}</div></div>
      <div class="kv-row"><div class="kv-key">test_group</div><div class="kv-value">date={test_group["date"]}, runs={test_group["run_count"]}, raw_rows={test_group["row_count"]}, valid_windows={test_group["valid_windows"]}</div></div>
      <div class="kv-row"><div class="kv-key">feature_set</div><div class="kv-value">{", ".join(list(subset["feature_names"]))}</div></div>
    </section>

    <section class="section">
      <h2>训练配置</h2>
      <div class="kv-row"><div class="kv-key">model</div><div class="kv-value">GruMlpRegressor, GRU hidden=256, layers=2, MLP hidden_dims=[512, 512], dropout=0.0</div></div>
      <div class="kv-row"><div class="kv-key">optimizer</div><div class="kv-value">AdamW, learning_rate={float(summary["experiment"]["learning_rate"]):.6g}</div></div>
      <div class="kv-row"><div class="kv-key">batching</div><div class="kv-value">batch_size={int(summary["experiment"]["batch_size"])}, eval_batch_size={int(summary["experiment"]["eval_batch_size"])}, window_size={int(summary["experiment"]["window_size"])}</div></div>
      <div class="kv-row"><div class="kv-key">iterations</div><div class="kv-value">num_iters={int(summary["experiment"]["num_iters"])}, save_every={int(summary["experiment"]["save_every"])}, checkpoints={", ".join(str(int(row["iter"])) for row in list(metric_rows))}</div></div>
    </section>

    <section class="section">
      <h2>Train Loss Curve</h2>
      <div class="figure">
        <img src="{_png_data_uri(train_loss_png)}" alt="train loss curve" />
        <div class="caption">使用 TensorBoard 原始标量 `train/objective/loss`, 不是 rolling `loss_mean`.</div>
      </div>
    </section>

    <section class="section">
      <h2>Checkpoint Eval</h2>
      <div class="figure">
        <img src="{_png_data_uri(checkpoint_png)}" alt="checkpoint mse curve" />
        <div class="caption">对每个 checkpoint 在 tiny train/tiny val 上做完整 evaluation. 指标是 raw 10-minute return space 的 MSE.</div>
      </div>
      <table>
        <thead>
          <tr>
            <th>iter</th>
            <th>train_mse</th>
            <th>train_ic</th>
            <th>val_mse</th>
            <th>val_ic</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    return html
