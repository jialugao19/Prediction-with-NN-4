"""Build the complete frozen baseline evaluation bundle."""

from __future__ import annotations

import hashlib
import html
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

from prediction_nn2.html_report import build_page, render_block_title, render_code_block, render_embedded_figure, render_html_table, render_section, render_subsection, render_table, render_value_rows


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path("/home/maomao/prediction-NN-2")
BENCHMARK_ID = "20260518_current_baseline"
BENCHMARK_ROOT = Path("/data-cache/nn/benchmarks/prediction_nn2") / BENCHMARK_ID
SOURCE_RUN_ROOT = Path("/data-cache/nn/0428/date_ranges")
SOURCE_SIGNAL_ROOT = Path("/data-cache/nn/trade_plan_experiments/0516_model_signal_diagnostics")
SOURCE_TRADING_ROOT = Path("/data-cache/nn/trade_plan_experiments/0516_percentile_hysteresis_baseline")
SOURCE_FEATURE_ROOT = Path("/data-cache/nn/trade_plan_experiments/0515/features/entry1_h60_slot60")
REPO_BENCHMARK_REPORT_DIR = REPO_ROOT / "report" / "benchmarks"
BEST_CHECKPOINT_ITER = 140000
REPORT_FIGURE_DIR = BENCHMARK_ROOT / "reports" / "figures"


FIELD_DEFINITIONS: dict[str, dict[str, str]] = {
    "top_decile_return_bps": {
        "display_name": "Top decile 平均收益",
        "unit": "bps",
        "direction": "higher is better",
        "explanation": "prediction 排名最高 10% 样本的平均 forward return, 用来判断模型最高分样本是否真的更容易上涨。",
    },
    "bottom_decile_return_bps": {
        "display_name": "Bottom decile 平均收益",
        "unit": "bps",
        "direction": "lower is expected",
        "explanation": "prediction 排名最低 10% 样本的平均 forward return, 越负说明模型越能识别差股票或弱机会。",
    },
    "top_minus_bottom_bps": {
        "display_name": "Top-Bottom signal spread",
        "unit": "bps",
        "direction": "higher is better",
        "explanation": "Top decile 平均收益减 Bottom decile 平均收益, 是 signal 横截面排序能力的核心读数。",
    },
    "high_liq_top_decile_net_proxy_bps": {
        "display_name": "高流动性 Top decile 净收益 proxy",
        "unit": "bps",
        "direction": "higher is better",
        "explanation": "高流动性股票中 top decile 扣除 entry spread 与 fee 后的近似收益, 用来判断信号在可交易区域是否仍然有效。",
    },
    "q95_q80_gross_daily_return_bps": {
        "display_name": "q95/q80 策略 gross 日收益",
        "unit": "bps",
        "direction": "higher is better",
        "explanation": "开仓阈值 95%, 平仓阈值 80% 的 trading rule 在扣成本前的日均收益。",
    },
    "q95_q80_net_daily_return_bps": {
        "display_name": "q95/q80 策略 net 日收益",
        "unit": "bps",
        "direction": "higher is better",
        "explanation": "q95/q80 trading rule 扣除 spread 与 fee 后的日均收益, 小于 0 表示当前成本假设下不可直接交易。",
    },
    "q95_q80_net_bps_per_turnover": {
        "display_name": "q95/q80 单位 turnover 净收益",
        "unit": "bps/turnover",
        "direction": "higher is better",
        "explanation": "每单位 turnover 对应的 net bps, 用来衡量交易效率, 比单纯日收益更能反映 cost drag。",
    },
    "best_time_bucket_net_bps_per_turnover": {
        "display_name": "最佳时间桶单位 turnover 净收益",
        "unit": "bps/turnover",
        "direction": "higher is better",
        "explanation": "所有 intraday time bucket 中 net bps per turnover 最高的 bucket, 用来定位信号最有效的交易时段。",
    },
    "worst_time_bucket_net_bps_per_turnover": {
        "display_name": "最差时间桶单位 turnover 净收益",
        "unit": "bps/turnover",
        "direction": "higher is better",
        "explanation": "所有 intraday time bucket 中 net bps per turnover 最低的 bucket, 是后续交易过滤需要重点关注的风险区间。",
    },
    "test_ic": {
        "display_name": "Test daily IC",
        "unit": "correlation",
        "direction": "higher is better",
        "explanation": "test split 上 prediction 与 forward return 的 daily Pearson IC, 用来衡量样本外横截面预测能力。",
    },
    "icir": {
        "display_name": "ICIR",
        "unit": "ratio",
        "direction": "higher is better",
        "explanation": "daily IC mean 除以 daily IC std, 用来衡量 IC 的稳定性。",
    },
    "val/objective/mse": {
        "display_name": "Validation MSE",
        "unit": "MSE",
        "direction": "lower is better",
        "explanation": "validation split 上的 MSE, 当前 checkpoint selector 使用这个指标选择 best checkpoint。",
    },
    "val/quality/global_ic": {
        "display_name": "Validation global IC",
        "unit": "correlation",
        "direction": "higher is better",
        "explanation": "validation split 上全局 prediction 与 target 的 Pearson correlation, 用来辅助判断 checkpoint 质量。",
    },
    "val/quality/rank_ic": {
        "display_name": "Validation rank IC",
        "unit": "correlation",
        "direction": "higher is better",
        "explanation": "validation split 上 prediction rank 与 target rank 的 correlation, 更关注排序能力。",
    },
    "val/dist/pred_std_over_target_std": {
        "display_name": "Prediction/Target std 比例",
        "unit": "ratio",
        "direction": "closer to reasonable scale is better",
        "explanation": "prediction 标准差除以 target 标准差, 太低说明 prediction 可能过度收缩, 太高说明输出可能过度放大。",
    },
    "mean_return_bps": {
        "display_name": "Bucket 平均收益",
        "unit": "bps",
        "direction": "higher is better",
        "explanation": "每个 signal bucket 内样本的平均 forward return, 用来检查 signal 是否随 bucket 单调变强。",
    },
    "hit_rate": {
        "display_name": "Hit rate",
        "unit": "ratio",
        "direction": "higher is better",
        "explanation": "bucket 内 forward return 为正的比例, 用来辅助判断信号方向是否稳定。",
    },
    "net_bps_per_turnover": {
        "display_name": "单位 turnover 净收益",
        "unit": "bps/turnover",
        "direction": "higher is better",
        "explanation": "扣除交易成本后的收益除以 turnover, 是比较不同交易规则和时段效率的关键指标。",
    },
    "entry_net_proxy_bps": {
        "display_name": "Entry 净收益 proxy",
        "unit": "bps",
        "direction": "higher is better",
        "explanation": "按 entry spread 和 fee 近似扣成本后的 bucket 收益, 用来判断 liquidity x signal 区域是否有交易价值。",
    },
    "ci_025": {
        "display_name": "Bootstrap 2.5% 分位",
        "unit": "same as metric",
        "direction": "interval lower bound",
        "explanation": "bootstrap confidence interval 的下界, 用来判断指标在抽样扰动下的最差合理区间。",
    },
    "ci_975": {
        "display_name": "Bootstrap 97.5% 分位",
        "unit": "same as metric",
        "direction": "interval upper bound",
        "explanation": "bootstrap confidence interval 的上界, 与下界一起衡量指标不确定性。",
    },
}

FIELD_DEFINITIONS.update(
    {
        "benchmark": {"display_name": "benchmark", "unit": "id", "direction": "identifier", "explanation": "本次 benchmark 的唯一标识, 用于连接报告、输出目录和后续实验对比。"},
        "selected_checkpoint": {"display_name": "selected_checkpoint", "unit": "iteration", "direction": "selected by rule", "explanation": "checkpoint selector 选出的模型迭代点。"},
        "train_ic": {"display_name": "train_ic", "unit": "correlation", "direction": "higher is better", "explanation": "train split 上逐日 Pearson IC 的均值, 用来和 test IC 比较泛化差距。"},
        "test_pooled_ic": {"display_name": "test_pooled_ic", "unit": "correlation", "direction": "higher is better", "explanation": "test split 全样本 pooling 后 prediction 与 target 的 Pearson correlation。"},
        "train_pooled_ic": {"display_name": "train_pooled_ic", "unit": "correlation", "direction": "higher is better", "explanation": "train split 全样本 pooling 后 prediction 与 target 的 Pearson correlation。"},
        "join_validation_status": {"display_name": "join_validation_status", "unit": "status", "direction": "passed is expected", "explanation": "prediction、feature 和 horizon label 的连接契约校验状态。"},
        "train_dates": {"display_name": "train_dates", "unit": "date range", "direction": "coverage field", "explanation": "训练集日期范围, 用于判断模型学习样本覆盖的市场环境。"},
        "val_dates": {"display_name": "val_dates", "unit": "date range", "direction": "coverage field", "explanation": "validation 日期范围, 用于 checkpoint selection 和调参监控。"},
        "test_dates": {"display_name": "test_dates", "unit": "date range", "direction": "coverage field", "explanation": "test 日期范围, 是样本外指标的评估区间。"},
        "raw_date_range": {"display_name": "raw_date_range", "unit": "date range", "direction": "coverage field", "explanation": "原始数据准备阶段覆盖的完整日期范围。"},
        "train_rows": {"display_name": "train_rows", "unit": "rows", "direction": "sample size", "explanation": "train split 的样本行数。"},
        "val_rows": {"display_name": "val_rows", "unit": "rows", "direction": "sample size", "explanation": "validation split 的样本行数。"},
        "test_rows": {"display_name": "test_rows", "unit": "rows", "direction": "sample size", "explanation": "test split 的样本行数。"},
        "feature_dim": {"display_name": "feature_dim", "unit": "count", "direction": "context field", "explanation": "模型输入 feature 数量。"},
        "features": {"display_name": "features", "unit": "names", "direction": "context field", "explanation": "本次训练使用的 feature 列表。"},
        "label": {"display_name": "label", "unit": "name", "direction": "context field", "explanation": "监督学习 target label。"},
        "feature_normalization": {"display_name": "feature_normalization", "unit": "method", "direction": "context field", "explanation": "feature 标准化设定, 影响模型输入 scale。"},
        "label_normalization": {"display_name": "label_normalization", "unit": "method", "direction": "context field", "explanation": "label 标准化设定, 影响 training loss 和 prediction scale。"},
        "prediction_rows": {"display_name": "prediction_rows", "unit": "rows", "direction": "data coverage", "explanation": "prediction 输出总行数。"},
        "feature_rows": {"display_name": "feature_rows", "unit": "rows", "direction": "data coverage", "explanation": "feature 数据总行数。"},
        "prediction_mean": {"display_name": "prediction_mean", "unit": "prediction", "direction": "near zero is expected", "explanation": "prediction 均值, 用于检查模型输出是否存在整体 bias。"},
        "prediction_std": {"display_name": "prediction_std", "unit": "prediction", "direction": "context field", "explanation": "prediction 标准差, 用于检查模型输出 scale。"},
        "target_mean": {"display_name": "target_mean", "unit": "target", "direction": "near zero is expected", "explanation": "target 均值, 用于检查 label 是否存在整体 bias。"},
        "target_std": {"display_name": "target_std", "unit": "target", "direction": "context field", "explanation": "target 标准差, 用于和 prediction scale 对比。"},
        "pred_std_over_target_std": {"display_name": "pred_std_over_target_std", "unit": "ratio", "direction": "closer to reasonable scale is better", "explanation": "prediction std 除以 target std, 用于判断 prediction 是否过度收缩或放大。"},
        "prediction_p01": {"display_name": "prediction_p01", "unit": "prediction", "direction": "tail quantile", "explanation": "prediction 1% 分位, 用于检查左尾输出。"},
        "prediction_p50": {"display_name": "prediction_p50", "unit": "prediction", "direction": "median", "explanation": "prediction median, 用于检查输出中心位置。"},
        "prediction_p99": {"display_name": "prediction_p99", "unit": "prediction", "direction": "tail quantile", "explanation": "prediction 99% 分位, 用于检查右尾输出。"},
        "target_p01": {"display_name": "target_p01", "unit": "target", "direction": "tail quantile", "explanation": "target 1% 分位, 用于确认 label 左尾风险。"},
        "target_p50": {"display_name": "target_p50", "unit": "target", "direction": "median", "explanation": "target median, 用于确认 label 中心位置。"},
        "target_p99": {"display_name": "target_p99", "unit": "target", "direction": "tail quantile", "explanation": "target 99% 分位, 用于确认 label 右尾风险。"},
        "joined_rows": {"display_name": "joined_rows", "unit": "rows", "direction": "higher coverage is better", "explanation": "prediction 与 feature/horizon join 后可进入 evaluation 的行数。"},
        "prediction_duplicate_keys": {"display_name": "prediction_duplicate_keys", "unit": "count", "direction": "lower is better", "explanation": "prediction 侧重复 key 数量, 正常应为 0。"},
        "feature_duplicate_keys": {"display_name": "feature_duplicate_keys", "unit": "count", "direction": "lower is better", "explanation": "feature 侧重复 key 数量, 正常应为 0。"},
        "unmatched_prediction_rows": {"display_name": "unmatched_prediction_rows", "unit": "rows", "direction": "lower is better", "explanation": "prediction 中无法连接 feature/horizon 的行数。"},
        "unmatched_feature_rows": {"display_name": "unmatched_feature_rows", "unit": "rows", "direction": "lower is better", "explanation": "feature 中没有 prediction 的行数。"},
        "horizon_alignment": {"display_name": "horizon_alignment", "unit": "status", "direction": "passed is expected", "explanation": "horizon label 是否按预期时间对齐。"},
        "null_prediction_rows": {"display_name": "null_prediction_rows", "unit": "rows", "direction": "lower is better", "explanation": "prediction 缺失行数, 用于检查推理输出完整性。"},
        "null_horizon_rows": {"display_name": "null_horizon_rows", "unit": "rows", "direction": "context field", "explanation": "horizon label 缺失行数, 通常来自无法形成 forward horizon 的样本。"},
        "status": {"display_name": "status", "unit": "status", "direction": "passed or complete is expected", "explanation": "对应表或校验流程的汇总状态。"},
        "experiment": {"display_name": "experiment", "unit": "id", "direction": "identifier", "explanation": "实验唯一标识。"},
        "date": {"display_name": "date", "unit": "date", "direction": "context field", "explanation": "实验运行或报告生成日期。"},
        "model": {"display_name": "model", "unit": "name", "direction": "context field", "explanation": "模型结构或模型类名。"},
        "best_iter": {"display_name": "best_iter", "unit": "iteration", "direction": "selected by rule", "explanation": "checkpoint selector 选出的 best iteration。"},
        "cost_model": {"display_name": "cost_model", "unit": "method", "direction": "context field", "explanation": "交易成本模型, 不同成本模型下的 net 指标不可直接比较。"},
        "verdict": {"display_name": "verdict", "unit": "text", "direction": "summary field", "explanation": "对该实验的简短判断。"},
        "metric": {"display_name": "metric", "unit": "name", "direction": "identifier", "explanation": "对比指标的 raw field name。"},
        "metric_name": {"display_name": "metric_name", "unit": "name", "direction": "identifier", "explanation": "normalization 或 diagnostics 表中的指标 raw field name。"},
        "value": {"display_name": "value", "unit": "mixed", "direction": "depends on metric", "explanation": "当前诊断项的数值或状态。"},
        "scope": {"display_name": "scope", "unit": "scope", "direction": "context field", "explanation": "统计或变换使用的数据范围。"},
        "baseline": {"display_name": "baseline", "unit": "same as metric", "direction": "reference value", "explanation": "baseline 实验中的指标值。"},
        "current": {"display_name": "current", "unit": "same as metric", "direction": "current value", "explanation": "当前实验中的指标值。"},
        "delta": {"display_name": "delta", "unit": "same as metric", "direction": "depends on metric", "explanation": "当前实验值减 baseline 值。"},
        "note": {"display_name": "note", "unit": "text", "direction": "context field", "explanation": "对比备注。"},
        "mean_ic": {"display_name": "mean_ic", "unit": "correlation", "direction": "higher is better", "explanation": "daily IC 均值, 衡量平均横截面预测能力。"},
        "std_ic": {"display_name": "std_ic", "unit": "correlation", "direction": "lower is more stable", "explanation": "daily IC 标准差, 衡量 IC 波动。"},
        "t_stat": {"display_name": "t_stat", "unit": "statistic", "direction": "larger absolute value is stronger", "explanation": "均值相对 0 的 t-stat。"},
        "ic_positive_ratio": {"display_name": "ic_positive_ratio", "unit": "ratio", "direction": "higher is better", "explanation": "daily IC 为正的比例。"},
        "daily_turnover": {"display_name": "daily_turnover", "unit": "turnover", "direction": "lower cost pressure is better", "explanation": "日均换手, 越高代表成本压力越大。"},
        "split": {"display_name": "split", "unit": "name", "direction": "context field", "explanation": "数据切分, 例如 train 或 test。"},
        "count": {"display_name": "count", "unit": "count", "direction": "sample size", "explanation": "参与统计的样本数量。"},
        "positive_ratio": {"display_name": "positive_ratio", "unit": "ratio", "direction": "higher is better", "explanation": "指标为正的比例。"},
        "pooled_ic": {"display_name": "pooled_ic", "unit": "correlation", "direction": "higher is better", "explanation": "全样本 pooling 后的 Pearson correlation。"},
        "pooled_rank_ic": {"display_name": "pooled_rank_ic", "unit": "correlation", "direction": "higher is better", "explanation": "全样本 pooling 后的 rank correlation。"},
        "signal_bucket": {"display_name": "signal_bucket", "unit": "bucket", "direction": "higher prediction bucket", "explanation": "按 prediction 排序后的分桶编号。"},
        "row_count": {"display_name": "row_count", "unit": "rows", "direction": "sample size", "explanation": "当前分组内样本数。"},
        "mean_prediction": {"display_name": "mean_prediction", "unit": "prediction", "direction": "monotonic by bucket", "explanation": "当前 bucket 内 prediction 均值。"},
        "mean_spread_bps": {"display_name": "mean_spread_bps", "unit": "bps", "direction": "lower cost is better", "explanation": "当前 bucket 内平均 bid-ask spread。"},
        "strategy_name": {"display_name": "strategy_name", "unit": "name", "direction": "identifier", "explanation": "backtest 使用的 trading rule 名称。"},
        "open_quantile": {"display_name": "open_quantile", "unit": "quantile", "direction": "rule parameter", "explanation": "开仓 signal 分位阈值。"},
        "close_quantile": {"display_name": "close_quantile", "unit": "quantile", "direction": "rule parameter", "explanation": "平仓 signal 分位阈值。"},
        "gross_daily_return_bps": {"display_name": "gross_daily_return_bps", "unit": "bps", "direction": "higher is better", "explanation": "扣成本前日均策略收益。"},
        "net_daily_return_bps": {"display_name": "net_daily_return_bps", "unit": "bps", "direction": "higher is better", "explanation": "扣 spread 和 fee 后日均策略收益。"},
        "gross_sharpe": {"display_name": "gross_sharpe", "unit": "ratio", "direction": "higher is better", "explanation": "gross return 的 Sharpe。"},
        "net_sharpe": {"display_name": "net_sharpe", "unit": "ratio", "direction": "higher is better", "explanation": "net return 的 Sharpe。"},
        "max_drawdown": {"display_name": "max_drawdown", "unit": "return", "direction": "less negative is better", "explanation": "策略净值最大回撤。"},
        "spread_cost_bps": {"display_name": "spread_cost_bps", "unit": "bps", "direction": "lower is better", "explanation": "bid-ask spread 成本。"},
        "fee_cost_bps": {"display_name": "fee_cost_bps", "unit": "bps", "direction": "lower is better", "explanation": "手续费成本。"},
        "gross_bps_per_turnover": {"display_name": "gross_bps_per_turnover", "unit": "bps/turnover", "direction": "higher is better", "explanation": "每单位 turnover 的 gross return。"},
        "active_names": {"display_name": "active_names", "unit": "count", "direction": "context field", "explanation": "平均持仓或活跃标的数量。"},
        "time_bucket": {"display_name": "time_bucket", "unit": "time bucket", "direction": "context field", "explanation": "intraday 时间分段。"},
        "gross_return_bps": {"display_name": "gross_return_bps", "unit": "bps", "direction": "higher is better", "explanation": "当前分组扣成本前收益。"},
        "turnover": {"display_name": "turnover", "unit": "turnover", "direction": "lower cost pressure is better", "explanation": "当前分组换手。"},
        "net_return_bps": {"display_name": "net_return_bps", "unit": "bps", "direction": "higher is better", "explanation": "当前分组扣成本后收益。"},
        "liq_bucket": {"display_name": "liq_bucket", "unit": "bucket", "direction": "context field", "explanation": "按 liquidity 排序后的分桶编号。"},
        "entry_spread_cost_bps": {"display_name": "entry_spread_cost_bps", "unit": "bps", "direction": "lower is better", "explanation": "建仓时按 spread 估算的成本。"},
        "entry_fee_cost_bps": {"display_name": "entry_fee_cost_bps", "unit": "bps", "direction": "lower is better", "explanation": "建仓时按 fee 估算的成本。"},
        "mean_signal_amount": {"display_name": "mean_signal_amount", "unit": "signal amount", "direction": "context field", "explanation": "当前 liquidity/signal bucket 的平均 signal amount。"},
        "mean": {"display_name": "mean", "unit": "same as metric", "direction": "depends on metric", "explanation": "bootstrap 或分组统计均值。"},
        "std": {"display_name": "std", "unit": "same as metric", "direction": "lower uncertainty is better", "explanation": "bootstrap 或分组统计标准差。"},
        "ci_500": {"display_name": "ci_500", "unit": "same as metric", "direction": "median", "explanation": "bootstrap median。"},
        "sample_count": {"display_name": "sample_count", "unit": "count", "direction": "sample size", "explanation": "bootstrap 使用的原始样本数。"},
        "bootstrap_samples": {"display_name": "bootstrap_samples", "unit": "count", "direction": "sample size", "explanation": "bootstrap 重采样次数。"},
        "month": {"display_name": "month", "unit": "month", "direction": "context field", "explanation": "月份分组。"},
        "positive_top_minus_bottom_ratio": {"display_name": "positive_top_minus_bottom_ratio", "unit": "ratio", "direction": "higher is better", "explanation": "月内 top-bottom spread 为正的比例。"},
        "q95_q80_net_daily_bps": {"display_name": "q95_q80_net_daily_bps", "unit": "bps", "direction": "higher is better", "explanation": "当月 q95/q80 trading rule 的 net daily bps。"},
        "regime": {"display_name": "regime", "unit": "bucket", "direction": "context field", "explanation": "market regime 分组。"},
        "day_count": {"display_name": "day_count", "unit": "days", "direction": "sample size", "explanation": "当前 regime 或分组内天数。"},
        "gross_daily_bps": {"display_name": "gross_daily_bps", "unit": "bps", "direction": "higher is better", "explanation": "当前 regime 内 gross daily bps。"},
        "net_daily_bps": {"display_name": "net_daily_bps", "unit": "bps", "direction": "higher is better", "explanation": "当前 regime 内 net daily bps。"},
        "volatility_bucket": {"display_name": "volatility_bucket", "unit": "bucket", "direction": "context field", "explanation": "按 volatility rank 划分的分桶。"},
        "mean_rank_ic": {"display_name": "mean_rank_ic", "unit": "correlation", "direction": "higher is better", "explanation": "rank correlation 的均值。"},
        "center_rank_min": {"display_name": "center_rank_min", "unit": "rank", "direction": "bucket boundary", "explanation": "volatility bucket 的 center rank 下界。"},
        "center_rank_max": {"display_name": "center_rank_max", "unit": "rank", "direction": "bucket boundary", "explanation": "volatility bucket 的 center rank 上界。"},
        "iter": {"display_name": "iter", "unit": "iteration", "direction": "context field", "explanation": "checkpoint 对应的训练 iteration。"},
        "selected": {"display_name": "selected", "unit": "boolean", "direction": "true is selected", "explanation": "当前 checkpoint 是否被最终选择。"},
        "metric_rank": {"display_name": "metric_rank", "unit": "rank", "direction": "lower is better", "explanation": "checkpoint selector 主指标排序。"},
        "decision": {"display_name": "decision", "unit": "text", "direction": "selector output", "explanation": "selector 对 checkpoint 的处理结果。"},
        "model_class": {"display_name": "model_class", "unit": "class", "direction": "context field", "explanation": "模型实现类, 用于区分 architecture。"},
        "input_tensor": {"display_name": "input_tensor", "unit": "shape", "direction": "context field", "explanation": "模型输入 tensor shape, 用于确认序列长度和 feature 维度。"},
        "gru_input_size": {"display_name": "gru_input_size", "unit": "dimension", "direction": "context field", "explanation": "GRU 每个 time step 的输入维度。"},
        "gru_hidden_size": {"display_name": "gru_hidden_size", "unit": "dimension", "direction": "capacity field", "explanation": "GRU hidden state 维度。"},
        "gru_num_layers": {"display_name": "gru_num_layers", "unit": "count", "direction": "capacity field", "explanation": "GRU 堆叠层数。"},
        "gru_bidirectional": {"display_name": "gru_bidirectional", "unit": "boolean", "direction": "architecture field", "explanation": "GRU 是否使用 bidirectional 结构。"},
        "mlp_effective_hidden_width": {"display_name": "mlp_effective_hidden_width", "unit": "dimension", "direction": "capacity field", "explanation": "GRU 输出后 MLP 的有效 hidden width。"},
        "trainable_parameters": {"display_name": "trainable_parameters", "unit": "count", "direction": "capacity field", "explanation": "模型可训练参数数量。"},
        "parameter_initialization": {"display_name": "parameter_initialization", "unit": "method", "direction": "context field", "explanation": "参数初始化方式。"},
        "best_checkpoint": {"display_name": "best_checkpoint", "unit": "path", "direction": "artifact field", "explanation": "用于 prediction 和 evaluation 的 checkpoint artifact。"},
        "state_dict_summary": {"display_name": "state_dict_summary", "unit": "path", "direction": "artifact field", "explanation": "state dict 结构摘要文件。"},
        "optimizer": {"display_name": "optimizer", "unit": "method", "direction": "training field", "explanation": "训练使用的优化器。"},
        "criterion": {"display_name": "criterion", "unit": "loss", "direction": "training field", "explanation": "训练目标函数。"},
        "final_iter": {"display_name": "final_iter", "unit": "iteration", "direction": "context field", "explanation": "训练结束 iteration。"},
        "checkpoint_selection": {"display_name": "checkpoint_selection", "unit": "rule", "direction": "selection field", "explanation": "checkpoint 选择规则。"},
        "checkpoint_saved_to": {"display_name": "checkpoint_saved_to", "unit": "path", "direction": "artifact field", "explanation": "best checkpoint 写入位置。"},
        "prediction_manifest": {"display_name": "prediction_manifest", "unit": "path", "direction": "artifact field", "explanation": "prediction 输出 manifest。"},
        "post_train_processing": {"display_name": "post_train_processing", "unit": "procedure", "direction": "context field", "explanation": "训练后处理步骤。"},
        "backtest_method": {"display_name": "backtest_method", "unit": "method", "direction": "context field", "explanation": "backtest 计算方法。"},
        "evaluation_contract": {"display_name": "evaluation_contract", "unit": "scope", "direction": "context field", "explanation": "evaluation 覆盖的数据和质量契约。"},
        "step": {"display_name": "step", "unit": "step", "direction": "time index", "explanation": "TensorBoard 或训练日志中的 step。"},
        "val_mse": {"display_name": "val_mse", "unit": "MSE", "direction": "lower is better", "explanation": "validation MSE。"},
        "train_loss_mean": {"display_name": "train_loss_mean", "unit": "loss", "direction": "lower is better", "explanation": "train loss 的窗口均值。"},
        "val_minus_train": {"display_name": "val_minus_train", "unit": "loss", "direction": "smaller gap is better", "explanation": "validation MSE 和 train loss mean 的绝对差。"},
        "val_over_train": {"display_name": "val_over_train", "unit": "ratio", "direction": "closer to 1 is expected", "explanation": "validation MSE 和 train loss mean 的比例。"},
        "tag": {"display_name": "tag", "unit": "name", "direction": "identifier", "explanation": "TensorBoard scalar tag。"},
        "last_step": {"display_name": "last_step", "unit": "step", "direction": "time index", "explanation": "当前 scalar 最后记录的 step。"},
        "last_value": {"display_name": "last_value", "unit": "scalar", "direction": "depends on tag", "explanation": "当前 scalar 最后记录值。"},
        "tail100_mean": {"display_name": "tail100_mean", "unit": "scalar", "direction": "depends on tag", "explanation": "当前 scalar 末尾 100 个点均值。"},
        "tail100_std": {"display_name": "tail100_std", "unit": "scalar", "direction": "lower is more stable", "explanation": "当前 scalar 末尾 100 个点标准差。"},
        "min_value": {"display_name": "min_value", "unit": "scalar", "direction": "depends on tag", "explanation": "当前 scalar 历史最小值。"},
        "max_value": {"display_name": "max_value", "unit": "scalar", "direction": "depends on tag", "explanation": "当前 scalar 历史最大值。"},
    }
)


def sha256_file(path: Path) -> str:
    """Compute the sha256 digest for one file."""
    # Stream the file in chunks.
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping."""
    # Parse the file as a mapping.
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    """Write one YAML mapping."""
    # Serialize with human-readable key order.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def copy_file(src: Path, dst: Path) -> Path:
    """Copy one artifact file."""
    # Create the destination parent.
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Copy metadata with the payload.
    shutil.copy2(Path(src), dst)
    return dst


def run_text(args: list[str], cwd: Path) -> str:
    """Run one command and return stdout."""
    # Execute without shell interpolation.
    result = subprocess.run(list(args), cwd=Path(cwd), check=True, capture_output=True, text=True)
    return str(result.stdout).strip()


def benchmark_path(relative_path: str) -> Path:
    """Resolve one path inside the benchmark root."""
    # Join the relative artifact path.
    return BENCHMARK_ROOT / str(relative_path)


def parquet_paths_from_manifest(manifest_path: Path, key: str) -> list[Path]:
    """Resolve parquet chunk paths from a manifest."""
    # Load the manifest payload.
    manifest = read_yaml(Path(manifest_path))

    # Resolve paths relative to the benchmark root or manifest parent.
    paths: list[Path] = []
    for raw_path in list(manifest[str(key)]):
        p = Path(str(raw_path))
        if p.is_absolute():
            paths.append(p)
        elif str(raw_path).startswith("predictions/"):
            paths.append(BENCHMARK_ROOT / p)
        else:
            paths.append(Path(manifest_path).parent / p)
    return paths


def duckdb_path_list(paths: list[Path]) -> str:
    """Render a DuckDB path list literal."""
    # Quote each path for SQL.
    quoted = [f"'{Path(path).as_posix()}'" for path in list(paths)]
    return "[" + ", ".join(quoted) + "]"


def build_evaluation_input_manifest() -> dict[str, Any]:
    """Build the fixed baseline evaluation input manifest."""
    # Load existing frozen benchmark metadata.
    benchmark = read_yaml(BENCHMARK_ROOT / "benchmark.yaml")
    git_commit = run_text(["git", "rev-parse", "HEAD"], REPO_ROOT)
    git_status = run_text(["git", "status", "--short"], REPO_ROOT)

    # Assemble the manifest payload.
    payload: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": BENCHMARK_ID,
        "experiment_id": BENCHMARK_ID,
        "parent_baseline_id": BENCHMARK_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code": {
            "repo_root": REPO_ROOT.as_posix(),
            "git_commit": git_commit,
            "git_status_short": git_status,
            "frozen_git_commit": benchmark["source"]["git_commit"],
            "frozen_diff_patch": "code/diff.patch",
        },
        "inputs": {
            "train_run_root": SOURCE_RUN_ROOT.as_posix(),
            "checkpoint_iter": int(BEST_CHECKPOINT_ITER),
            "prediction_manifest": (SOURCE_RUN_ROOT / "run" / "inference_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "inference_manifest.yaml").as_posix(),
            "enriched_prediction_manifest": "predictions/enriched_test_predictions_manifest.yaml",
            "feature_manifest": (SOURCE_FEATURE_ROOT / "feature_manifest.yaml").as_posix(),
            "trading_baseline_root": SOURCE_TRADING_ROOT.as_posix(),
            "signal_diagnostics_root": SOURCE_SIGNAL_ROOT.as_posix(),
            "cost_model": "spread_plus_fee",
            "holding": "10min",
            "join_key": ["date", "time", "code"],
        },
        "controlled_components": {
            "data_split": "data/split_dates.yaml",
            "feature_set": "data/feature_names.yaml",
            "normalization": "data/normalization_contract.yaml",
            "label": "data/npz_meta.yaml",
            "model_architecture": "model/effective_model_summary.yaml",
            "train_config": "train/train_config.yaml",
            "trading_rule": "10min_long_only_percentile_hysteresis",
        },
    }
    write_yaml(BENCHMARK_ROOT / "evaluation" / "evaluation_input_manifest.yaml", payload)
    return payload


def run_join_validation() -> dict[str, Any]:
    """Validate the prediction-feature join contract."""
    # Resolve prediction and enriched chunk inputs.
    pred_manifest = Path(SOURCE_RUN_ROOT / "run" / "inference_test" / f"iter_{BEST_CHECKPOINT_ITER}" / "inference_manifest.yaml")
    enriched_manifest = BENCHMARK_ROOT / "predictions" / "enriched_test_predictions_manifest.yaml"
    pred_paths = parquet_paths_from_manifest(pred_manifest, "chunk_files")
    enriched_paths = parquet_paths_from_manifest(enriched_manifest, "benchmark_chunk_files")
    pred_sql_paths = duckdb_path_list(pred_paths)
    enriched_sql_paths = duckdb_path_list(enriched_paths)

    # Compute row counts and duplicate checks.
    con = duckdb.connect()
    con.execute("set threads to 8")
    pred_stats = con.execute(
        f"""
        select
            count(*)::BIGINT as prediction_rows,
            count(distinct (date, time, code))::BIGINT as prediction_distinct_keys,
            min(date)::BIGINT as prediction_date_min,
            max(date)::BIGINT as prediction_date_max,
            min(time)::BIGINT as prediction_time_min,
            max(time)::BIGINT as prediction_time_max
        from read_parquet({pred_sql_paths})
        """
    ).fetchone()
    feature_stats = con.execute(
        f"""
        select
            count(*)::BIGINT as feature_rows,
            count(*) filter (where prediction_available)::BIGINT as joined_rows,
            count(distinct (date, time, code))::BIGINT as feature_distinct_keys,
            min(date)::BIGINT as feature_date_min,
            max(date)::BIGINT as feature_date_max,
            min(time)::BIGINT as feature_time_min,
            max(time)::BIGINT as feature_time_max,
            count(*) filter (where prediction is null)::BIGINT as null_prediction_rows,
            count(*) filter (where prediction_available and ret_vwap_exec_10 is null)::BIGINT as null_horizon_rows
        from read_parquet({enriched_sql_paths})
        """
    ).fetchone()
    con.close()

    # Convert query results to named values.
    prediction_rows = int(pred_stats[0])
    prediction_distinct_keys = int(pred_stats[1])
    feature_rows = int(feature_stats[0])
    joined_rows = int(feature_stats[1])
    feature_distinct_keys = int(feature_stats[2])
    prediction_duplicate_keys = int(prediction_rows - prediction_distinct_keys)
    feature_duplicate_keys = int(feature_rows - feature_distinct_keys)
    unmatched_prediction_rows = int(prediction_rows - joined_rows)
    unmatched_feature_rows = int(feature_rows - joined_rows)
    null_horizon_rows = int(feature_stats[8])

    # Enforce the validation gate.
    status = "passed"
    failures: list[str] = []
    if prediction_duplicate_keys != 0:
        failures.append("prediction_duplicate_keys")
    if feature_duplicate_keys != 0:
        failures.append("feature_duplicate_keys")
    if unmatched_prediction_rows != 0:
        failures.append("unmatched_prediction_rows")
    horizon_alignment = "passed" if null_horizon_rows < joined_rows else "failed"
    if horizon_alignment != "passed":
        failures.append("horizon_alignment")
    if len(failures) > 0:
        status = "failed"

    # Write the validation payload.
    payload: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": BENCHMARK_ID,
        "prediction_rows": prediction_rows,
        "feature_rows": feature_rows,
        "joined_rows": joined_rows,
        "prediction_duplicate_keys": prediction_duplicate_keys,
        "feature_duplicate_keys": feature_duplicate_keys,
        "unmatched_prediction_rows": unmatched_prediction_rows,
        "unmatched_feature_rows": unmatched_feature_rows,
        "date_min": int(min(int(pred_stats[2]), int(feature_stats[3]))),
        "date_max": int(max(int(pred_stats[3]), int(feature_stats[4]))),
        "time_min": int(min(int(pred_stats[4]), int(feature_stats[5]))),
        "time_max": int(max(int(pred_stats[5]), int(feature_stats[6]))),
        "horizon_label": "ret_vwap_exec_10",
        "horizon_alignment": horizon_alignment,
        "null_prediction_rows": int(feature_stats[7]),
        "null_horizon_rows": null_horizon_rows,
        "status": status,
        "failures": failures,
    }
    write_yaml(BENCHMARK_ROOT / "evaluation" / "join_validation.yaml", payload)
    if status != "passed":
        raise RuntimeError(f"join validation failed: {failures}")
    return payload


def standardize_signal_metrics() -> pd.DataFrame:
    """Write the stable signal bucket metrics table."""
    # Load existing decile diagnostics.
    src = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "top_tail_metrics.csv")

    # Convert the existing schema into the stable contract.
    out = pd.DataFrame(
        {
            "signal_bucket": src["signal_decile"].astype(int),
            "row_count": src["mean_row_count"].astype(float),
            "mean_prediction": src["mean_prediction"].astype(float),
            "mean_return_bps": src["daily_return_bps"].astype(float),
            "t_stat": src["t_stat"].astype(float),
            "hit_rate": src["hit_rate"].astype(float),
            "mean_spread_bps": src["mean_spread_bps"].astype(float),
        }
    )
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv", index=False)
    return out


def standardize_liquidity_metrics() -> pd.DataFrame:
    """Write the stable liquidity bucket metrics table."""
    # Load existing liquidity diagnostics.
    src = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_signal_metrics.csv")

    # Convert the existing schema into the stable contract.
    out = pd.DataFrame(
        {
            "liq_bucket": src["liq_bucket"].astype(int),
            "signal_bucket": src["signal_decile"].astype(int),
            "row_count": src["row_count"].astype(int),
            "gross_return_bps": src["gross_return_bps"].astype(float),
            "entry_spread_cost_bps": src["mean_spread_bps"].astype(float),
            "entry_fee_cost_bps": 1.0,
            "entry_net_proxy_bps": src["entry_net_proxy_bps"].astype(float),
            "mean_signal_amount": pd.NA,
            "mean_spread_bps": src["mean_spread_bps"].astype(float),
        }
    )
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv", index=False)
    return out


def build_time_bucket_metrics() -> pd.DataFrame:
    """Write the stable time bucket attribution table."""
    # Load the q95_q80 bar-level baseline output.
    bar_path = BENCHMARK_ROOT / "backtest" / "variants" / "q95_q80" / "percentile_hysteresis_10min_bar.csv"
    bar = pd.read_csv(bar_path)

    # Compute net returns and bps-per-turnover fields.
    bar["net_return"] = bar["gross_return"] - bar["spread_cost"] - bar["fee_cost"]
    grouped = (
        bar.groupby("time", dropna=False)
        .agg(
            row_count=("date", "count"),
            gross_return=("gross_return", "mean"),
            turnover=("turnover", "mean"),
            spread_cost=("spread_cost", "mean"),
            fee_cost=("fee_cost", "mean"),
            net_return=("net_return", "mean"),
            active_names=("active_name_count", "mean"),
        )
        .reset_index()
        .sort_values("time")
    )

    # Convert returns into bps and turnover-normalized values.
    grouped["time_bucket"] = grouped["time"].astype(int).astype(str).str.zfill(6)
    grouped["gross_return_bps"] = grouped["gross_return"].astype(float) * 1e4
    grouped["spread_cost_bps"] = grouped["spread_cost"].astype(float) * 1e4
    grouped["fee_cost_bps"] = grouped["fee_cost"].astype(float) * 1e4
    grouped["net_return_bps"] = grouped["net_return"].astype(float) * 1e4
    grouped["gross_bps_per_turnover"] = grouped["gross_return_bps"] / grouped["turnover"]
    grouped["net_bps_per_turnover"] = grouped["net_return_bps"] / grouped["turnover"]
    out = grouped[
        [
            "time_bucket",
            "row_count",
            "gross_return_bps",
            "turnover",
            "spread_cost_bps",
            "fee_cost_bps",
            "net_return_bps",
            "gross_bps_per_turnover",
            "net_bps_per_turnover",
            "active_names",
        ]
    ]
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv", index=False)
    write_yaml(
        BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.yaml",
        {"schema_version": 1, "status": "computed", "metrics_csv": "evaluation/time_bucket_metrics.csv", "row_count": int(out.shape[0])},
    )
    return out


def build_extreme_value_metrics() -> dict[str, Any]:
    """Write extreme value diagnostics."""
    # Resolve enriched chunk files.
    enriched_manifest = BENCHMARK_ROOT / "predictions" / "enriched_test_predictions_manifest.yaml"
    enriched_paths = parquet_paths_from_manifest(enriched_manifest, "benchmark_chunk_files")
    enriched_sql_paths = duckdb_path_list(enriched_paths)

    # Aggregate extreme indicators in DuckDB.
    con = duckdb.connect()
    con.execute("set threads to 8")
    rows = con.execute(
        f"""
        with base as (
          select
            ret_vwap_exec_10,
            prediction,
            ntile(10) over (partition by date, time order by prediction) as signal_bucket,
            is_limit_up_all_day,
            is_limit_down_all_day,
            entry_open_is_up_limit or entry_vwap_is_up_limit or exit_open_is_up_limit or exit_vwap_is_up_limit as limit_up_execution,
            entry_open_is_down_limit or entry_vwap_is_down_limit or exit_open_is_down_limit or exit_vwap_is_down_limit as limit_down_execution,
            not fillable_open or not fillable_vwap as non_fillable,
            ret_vwap_exec_10 >= quantile_cont(ret_vwap_exec_10, 0.995) over () as extreme_positive_return,
            ret_vwap_exec_10 <= quantile_cont(ret_vwap_exec_10, 0.005) over () as extreme_negative_return
          from read_parquet({enriched_sql_paths})
          where prediction_available and ret_vwap_exec_10 is not null
        ),
        typed as (
          select 'all_day_limit_up' as extreme_type, * exclude(is_limit_down_all_day, limit_up_execution, limit_down_execution, non_fillable, extreme_positive_return, extreme_negative_return)
          from base where is_limit_up_all_day
          union all
          select 'all_day_limit_down' as extreme_type, * exclude(is_limit_up_all_day, limit_up_execution, limit_down_execution, non_fillable, extreme_positive_return, extreme_negative_return)
          from base where is_limit_down_all_day
          union all
          select 'execution_limit_up' as extreme_type, * exclude(is_limit_up_all_day, is_limit_down_all_day, limit_down_execution, non_fillable, extreme_positive_return, extreme_negative_return)
          from base where limit_up_execution
          union all
          select 'execution_limit_down' as extreme_type, * exclude(is_limit_up_all_day, is_limit_down_all_day, limit_up_execution, non_fillable, extreme_positive_return, extreme_negative_return)
          from base where limit_down_execution
          union all
          select 'non_fillable' as extreme_type, * exclude(is_limit_up_all_day, is_limit_down_all_day, limit_up_execution, limit_down_execution, extreme_positive_return, extreme_negative_return)
          from base where non_fillable
          union all
          select 'extreme_positive_return' as extreme_type, * exclude(is_limit_up_all_day, is_limit_down_all_day, limit_up_execution, limit_down_execution, non_fillable, extreme_negative_return)
          from base where extreme_positive_return
          union all
          select 'extreme_negative_return' as extreme_type, * exclude(is_limit_up_all_day, is_limit_down_all_day, limit_up_execution, limit_down_execution, non_fillable, extreme_positive_return)
          from base where extreme_negative_return
        )
        select
          extreme_type,
          count(*)::BIGINT as row_count,
          count(*)::DOUBLE / (select count(*) from base) as row_ratio,
          avg(case when signal_bucket = 10 then 1.0 else 0.0 end) as top_decile_share,
          avg(case when signal_bucket = 1 then 1.0 else 0.0 end) as bottom_decile_share,
          avg(ret_vwap_exec_10) * 1e4 as mean_return_bps
        from typed
        group by extreme_type
        order by extreme_type
        """
    ).fetchdf()
    con.close()

    # Persist both CSV and YAML forms for review.
    csv_path = BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.csv"
    rows.to_csv(csv_path, index=False)
    payload = {"schema_version": 1, "status": "computed", "metrics_csv": "evaluation/extreme_value_metrics.csv", "rows": rows.to_dict(orient="records")}
    write_yaml(BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.yaml", payload)
    return payload


def build_normalization_metrics() -> dict[str, Any]:
    """Write normalization diagnostics."""
    # Load frozen normalization and scalar files.
    norm = read_yaml(BENCHMARK_ROOT / "data" / "normalization_contract.yaml")
    scalars = pd.read_parquet(BENCHMARK_ROOT / "train" / "train_scalars.parquet")

    # Pull the final validation distribution scalar per tag.
    dist = scalars[scalars["tag"].astype(str).str.startswith("val/dist/")].copy()
    dist = dist.sort_values(["tag", "step"]).groupby("tag", as_index=False).tail(1)
    rows = []
    for _, row in dist.iterrows():
        rows.append({"metric_name": str(row["tag"]), "value": float(row["value"]), "split": "val", "note": "final TensorBoard scalar"})

    # Add normalization contract values.
    rows.append({"metric_name": "feature_transform", "value": str(norm["feature_transform"]["stock_norm"]["type"]), "split": "train", "note": "frozen contract"})
    rows.append({"metric_name": "label_transform", "value": str(norm["label_transform"]["type"]), "split": "train", "note": "frozen contract"})
    rows.append({"metric_name": "label_transform_scope", "value": str(norm["label_transform"].get("scope", "n/a")), "split": "train", "note": "frozen contract"})
    out = pd.DataFrame(rows)
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", index=False)
    payload = {"schema_version": 1, "status": "computed", "metrics_csv": "evaluation/normalization_metrics.csv", "normalization_contract": "data/normalization_contract.yaml"}
    write_yaml(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.yaml", payload)
    return payload


def standardize_trading_rule_metrics() -> pd.DataFrame:
    """Write the stable trading rule metrics table."""
    # Load the existing baseline summary.
    src = pd.read_csv(BENCHMARK_ROOT / "backtest" / "baseline_summary.csv")

    # Convert returns and costs into stable bps fields.
    out = pd.DataFrame(
        {
            "strategy_name": src["strategy_name"].astype(str),
            "open_quantile": src["open_quantile"].astype(float),
            "close_quantile": src["close_quantile"].astype(float),
            "gross_daily_return_bps": src["gross_daily_return"].astype(float) * 1e4,
            "net_daily_return_bps": src["net_daily_return"].astype(float) * 1e4,
            "gross_sharpe": src["gross_sharpe"].astype(float),
            "net_sharpe": src["net_sharpe"].astype(float),
            "max_drawdown": src["gross_max_drawdown"].astype(float),
            "daily_turnover": src["daily_turnover"].astype(float),
            "spread_cost_bps": src["daily_spread_cost"].astype(float) * 1e4,
            "fee_cost_bps": src["daily_fee_cost"].astype(float) * 1e4,
            "gross_bps_per_turnover": src["gross_daily_return"].astype(float) * 1e4 / src["daily_turnover"].astype(float),
            "net_bps_per_turnover": src["net_daily_return"].astype(float) * 1e4 / src["daily_turnover"].astype(float),
            "active_names": src["mean_active_name_count"].astype(float),
        }
    )
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv", index=False)
    return out


def build_evaluation_summary(
    join_validation: dict[str, Any],
    signal: pd.DataFrame,
    liquidity: pd.DataFrame,
    time_bucket: pd.DataFrame,
    trading: pd.DataFrame,
) -> dict[str, Any]:
    """Write the headline evaluation summary."""
    # Derive headline metrics from stable tables.
    top = signal.loc[signal["signal_bucket"].astype(int) == 10].iloc[0]
    bottom = signal.loc[signal["signal_bucket"].astype(int) == 1].iloc[0]
    high_liq_top = liquidity.loc[(liquidity["liq_bucket"].astype(int) == 1) & (liquidity["signal_bucket"].astype(int) == 10)].iloc[0]
    q95_q80 = trading.loc[trading["strategy_name"].astype(str) == "q95_q80"].iloc[0]
    best_time = time_bucket.sort_values("net_bps_per_turnover", ascending=False).iloc[0]
    worst_time = time_bucket.sort_values("net_bps_per_turnover", ascending=True).iloc[0]

    # Assemble the summary payload.
    payload: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": BENCHMARK_ID,
        "status": "complete" if join_validation["status"] == "passed" else "failed",
        "join_validation_status": join_validation["status"],
        "headline_metrics": {
            "top_decile_return_bps": float(top["mean_return_bps"]),
            "bottom_decile_return_bps": float(bottom["mean_return_bps"]),
            "top_minus_bottom_bps": float(top["mean_return_bps"] - bottom["mean_return_bps"]),
            "high_liq_top_decile_net_proxy_bps": float(high_liq_top["entry_net_proxy_bps"]),
            "q95_q80_gross_daily_return_bps": float(q95_q80["gross_daily_return_bps"]),
            "q95_q80_net_daily_return_bps": float(q95_q80["net_daily_return_bps"]),
            "q95_q80_net_bps_per_turnover": float(q95_q80["net_bps_per_turnover"]),
            "best_time_bucket_by_net_bps_per_turnover": str(best_time["time_bucket"]),
            "best_time_bucket_net_bps_per_turnover": float(best_time["net_bps_per_turnover"]),
            "worst_time_bucket_by_net_bps_per_turnover": str(worst_time["time_bucket"]),
            "worst_time_bucket_net_bps_per_turnover": float(worst_time["net_bps_per_turnover"]),
        },
        "artifacts": {
            "evaluation_input_manifest": "evaluation/evaluation_input_manifest.yaml",
            "join_validation": "evaluation/join_validation.yaml",
            "signal_bucket_metrics": "evaluation/signal_bucket_metrics.csv",
            "liquidity_bucket_metrics": "evaluation/liquidity_bucket_metrics.csv",
            "time_bucket_metrics": "evaluation/time_bucket_metrics.csv",
            "extreme_value_metrics": "evaluation/extreme_value_metrics.yaml",
            "normalization_metrics": "evaluation/normalization_metrics.yaml",
            "trading_rule_metrics": "evaluation/trading_rule_metrics.csv",
        },
    }
    write_yaml(BENCHMARK_ROOT / "evaluation" / "evaluation_summary.yaml", payload)
    return payload


def build_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    """Write the self-comparison against the parent baseline."""
    # Use identical baseline and experiment values.
    metrics = dict(summary["headline_metrics"])
    primary_value = float(metrics["q95_q80_net_bps_per_turnover"])
    payload = {
        "schema_version": 1,
        "parent_evaluation_id": BENCHMARK_ID,
        "experiment_id": BENCHMARK_ID,
        "hypothesis": "current frozen baseline reference",
        "changed_component": "none",
        "primary_metric": {
            "name": "q95_q80_net_bps_per_turnover",
            "baseline": primary_value,
            "experiment": primary_value,
            "absolute_change": 0.0,
            "relative_change": 0.0,
            "passed_threshold": True,
        },
        "secondary_metrics": {
            "top_decile_return_bps": float(metrics["top_decile_return_bps"]),
            "top_minus_bottom_bps": float(metrics["top_minus_bottom_bps"]),
            "high_liq_top_decile_net_proxy_bps": float(metrics["high_liq_top_decile_net_proxy_bps"]),
            "daily_turnover": float(pd.read_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv").query("strategy_name == 'q95_q80'").iloc[0]["daily_turnover"]),
        },
        "stability_checks": {
            "val_test_direction_consistent": "baseline_reference",
            "date_bucket_direction_consistent": "baseline_reference",
            "time_bucket_no_large_regression": "baseline_reference",
        },
        "decision": "baseline_reference",
    }
    write_yaml(BENCHMARK_ROOT / "evaluation" / "comparison_against_parent.yaml", payload)
    return payload


def render_csv_table(path: Path, n_rows: int) -> str:
    """Render a CSV preview table."""
    # Read a compact preview.
    df = pd.read_csv(path).head(int(n_rows))

    # Format cells as strings.
    rows = [[format_report_cell(str(col), value) for col, value in zip(list(df.columns), list(row), strict=True)] for row in df.astype(object).to_numpy().tolist()]
    return '<div class="table-wrap">' + render_table([str(col) for col in df.columns], rows) + "</div>"


def render_rows_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render an in-memory row table."""
    # Format every cell through the same report value formatter used by CSV tables.
    formatted_rows = [[format_report_cell(str(col), value) for col, value in zip(list(headers), list(row), strict=True)] for row in list(rows)]
    return '<div class="table-wrap">' + render_table(list(headers), formatted_rows) + "</div>"


def format_report_cell(column: str, value: Any) -> str:
    """Format one report table cell."""
    # Keep raw metric names in table cells for traceability.
    if str(column) == "metric":
        return str(value)
    return format_report_value(str(column), value)


def format_report_value(metric: str, value: Any) -> str:
    """Format one report value for human scanning."""
    # Preserve missing values as a short literal.
    if pd.isna(value):
        return "nan"

    # Keep non-numeric values readable and shorten absolute checkpoint paths.
    if not isinstance(value, (int, float, np.integer, np.floating)):
        text = str(value)
        if text.startswith("/data-cache/"):
            return Path(text).name
        return text

    # Format integer-like counts without decimal noise.
    name = str(metric).lower()
    number = float(value)
    if "count" in name or "rows" in name or name in {"iter", "step", "time", "time_bucket", "last_step", "prev_iter"}:
        return f"{number:,.0f}"

    # Format bps, turnover, IC and loss-like values with stable precision.
    if "bps" in name or "return" in name or "spread" in name or "cost" in name:
        return f"{number:.2f}"
    if "ic" in name or "ratio" in name or "turnover" in name or "hit_rate" in name:
        return f"{number:.4f}"
    if "mse" in name or "loss" in name or abs(number) < 0.0001:
        return f"{number:.3e}"
    return f"{number:.4f}"


def display_metric_name(raw_name: str) -> str:
    """Return the Chinese display name for one raw metric when available."""
    # Use the centralized field dictionary as the only display-name source.
    field = FIELD_DEFINITIONS.get(str(raw_name))
    if field is None:
        return str(raw_name)
    return str(field["display_name"])


def report_badge(label: str, status: str) -> str:
    """Render one status badge."""
    # Choose a small semantic class for visual scanning.
    class_name = {
        "good": "badge-good",
        "watch": "badge-watch",
        "bad": "badge-bad",
        "neutral": "badge-neutral",
    }[str(status)]
    return f'<span class="badge {class_name}">{html.escape(str(label))}</span>'


def render_summary_cards(cards: list[tuple[str, str, str]]) -> str:
    """Render compact summary cards for the report first screen."""
    # Render each card with escaped label, value and note.
    items: list[str] = []
    for key, value, note in list(cards):
        items.append(
            f"""
            <div class="summary-card">
              <div class="summary-card-key">{html.escape(str(key))}</div>
              <div class="summary-card-value">{html.escape(str(value))}</div>
              <div class="summary-card-note">{html.escape(str(note))}</div>
            </div>
            """
        )
    return '<div class="summary-grid">' + "\n".join(items) + "</div>"


def render_takeaways(items: list[str]) -> str:
    """Render concise takeaway bullets."""
    # Escape each takeaway while keeping the list visually distinct.
    rows = [f"<li>{html.escape(str(item))}</li>" for item in list(items)]
    return '<ul class="takeaways">' + "\n".join(rows) + "</ul>"


def render_metric_review_table(rows: list[tuple[str, Any, str, str]]) -> str:
    """Render a status-aware metric review table."""
    # Convert metric rows into trusted HTML cells.
    table_rows: list[list[str]] = []
    for metric, value, interpretation, status in list(rows):
        table_rows.append(
            [
                html.escape(str(metric)),
                html.escape(format_report_value(str(metric), value)),
                html.escape(str(interpretation)),
                report_badge(str(status), metric_status_class(str(status))),
            ]
        )
    return '<div class="table-wrap">' + render_html_table(["metric", "value", "interpretation", "status"], table_rows) + "</div>"


def metric_status_class(status: str) -> str:
    """Map report status text to badge class names."""
    # Keep status wording separate from CSS class names.
    if str(status) in {"passed", "selected", "positive", "stable", "baseline"}:
        return "good"
    if str(status) in {"watch", "mixed", "reference"}:
        return "watch"
    if str(status) in {"negative", "failed", "bad"}:
        return "bad"
    return "neutral"


def render_details_table(title: str, body: str) -> str:
    """Render an expandable detail block."""
    # Keep long appendix tables available without dominating the first read.
    return f'<details class="details"><summary>{html.escape(str(title))}</summary>{body}</details>'


def render_details_code(title: str, text: str) -> str:
    """Render an expandable raw text block."""
    # Reuse the report details styling for raw YAML payloads.
    return f'<details class="details"><summary>{html.escape(str(title))}</summary>{render_code_block(str(text))}</details>'


def render_field_notes(raw_fields: list[str], section_note: str) -> str:
    """Render visible Chinese explanations for selected fields."""
    # Start with the section-level reading guide.
    notes = [f'<div class="field-note">{html.escape(str(section_note))}</div>']

    # Render one detailed note per known raw field.
    for raw_field in list(raw_fields):
        field = FIELD_DEFINITIONS.get(str(raw_field))
        if field is None:
            notes.append(
                "<div class=\"field-note\">"
                f"<strong>{html.escape(str(raw_field))}</strong>: "
                "当前 raw field 暂未登记详细解释, 需要在后续版本补充字段定义。"
                "</div>"
            )
            continue
        notes.append(
            "<div class=\"field-note\">"
            f"<strong>{html.escape(str(field['display_name']))}</strong>: "
            f"{html.escape(str(field['explanation']))} "
            f"单位: {html.escape(str(field['unit']))}; "
            f"读数方向: {html.escape(str(field['direction']))}."
            "</div>"
        )
    return '<div class="field-notes"><div class="field-notes-title">字段解释</div>' + "\n".join(notes) + "</div>"


def field_name_appendix() -> str:
    """Render the raw-field appendix table."""
    # Build the final mapping from Chinese display names back to raw field names.
    rows = []
    for raw_name, field in FIELD_DEFINITIONS.items():
        rows.append(
            [
                str(field["display_name"]),
                str(raw_name),
                str(field["unit"]),
                str(field["direction"]),
                str(field["explanation"]),
            ]
        )
    return '<div class="table-wrap">' + render_table(["display_name", "raw_field_name", "unit", "direction", "explanation"], rows) + "</div>"


def ic_summary_table(train_ic: dict[str, Any], test_ic: dict[str, Any]) -> str:
    """Render train and test IC summaries as columns."""
    # Build one row per split from the Pearson IC YAML payloads.
    pooled = source_pooled_ic_values()
    rows = []
    for split, payload in [("train", train_ic), ("test", test_ic)]:
        rows.append(
            [
                split,
                format_report_value("count", payload["count"]),
                format_report_value("mean_ic", payload["mean"]),
                format_report_value("std_ic", payload["std"]),
                format_report_value("t_stat", payload["t_stat"]),
                format_report_value("positive_ratio", payload["positive_ratio"]),
                format_report_value("icir", payload["icir"]),
                format_report_value("pooled_ic", pooled[f"{split}_pooled_ic"]),
                format_report_value("pooled_rank_ic", float("nan")),
            ]
        )
    return '<div class="table-wrap">' + render_table(["split", "count", "mean_ic", "std_ic", "t_stat", "positive_ratio", "icir", "pooled_ic", "pooled_rank_ic"], rows) + "</div>"


def source_pooled_ic_values() -> dict[str, float]:
    """Read pooled Pearson IC values from the frozen source report."""
    # Extract the source training report text that already records train/test pooled IC.
    text = (SOURCE_RUN_ROOT / "train_report.html").read_text(encoding="utf-8")

    # Pull exact key/value pairs from the report HTML.
    values: dict[str, float] = {}
    for key in ["pooled_ic_train", "pooled_ic_test"]:
        pattern = rf'<div class="kv-key">{re.escape(key)}</div>\s*<div class="kv-value">([^<]+)</div>'
        match = re.search(pattern, text)
        values[key] = float(match.group(1).replace(",", ""))
    return {"train_pooled_ic": values["pooled_ic_train"], "test_pooled_ic": values["pooled_ic_test"]}


def prediction_target_scale_rows() -> list[tuple[str, float]]:
    """Build prediction/target scale check rows from normalization metrics."""
    # Convert the normalization metric rows into a lookup table.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv")
    metrics = {str(row["metric_name"]): float(row["value"]) for _, row in df[df["split"] == "val"].iterrows()}

    # Map TensorBoard scalar names to the report raw field names from outline.md.
    return [
        ("prediction_mean", metrics["val/dist/mean/pred"]),
        ("prediction_std", metrics["val/dist/std/pred"]),
        ("target_mean", metrics["val/dist/mean/target"]),
        ("target_std", metrics["val/dist/std/target"]),
        ("pred_std_over_target_std", metrics["val/dist/pred_std_over_target_std"]),
        ("prediction_p01", metrics["val/dist/p01/pred"]),
        ("prediction_p50", metrics["val/dist/p50/pred"]),
        ("prediction_p99", metrics["val/dist/p99/pred"]),
        ("target_p01", metrics["val/dist/p01/target"]),
        ("target_p50", metrics["val/dist/p50/target"]),
        ("target_p99", metrics["val/dist/p99/target"]),
    ]


def prediction_target_scale_table() -> str:
    """Render the prediction/target scale check table."""
    # Build a two-column raw field/value table.
    rows = [[field, value] for field, value in prediction_target_scale_rows()]
    return render_rows_table(["raw field name", "value"], rows)


def checkpoint_summary_table() -> str:
    """Render the compact checkpoint selector table."""
    # Read only the columns needed to explain the selection decision.
    df = pd.read_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv")
    columns = ["iter", "selected", "val/objective/mse", "val/quality/global_ic", "val/quality/rank_ic", "val/dist/pred_std_over_target_std", "metric_rank", "decision"]
    compact = df.loc[:, columns].sort_values(["selected", "metric_rank"], ascending=[False, True])

    # Format selected values for scan-friendly display.
    rows = []
    for _, row in compact.iterrows():
        rows.append([format_report_value(col, row[col]) for col in columns])
    return '<div class="table-wrap">' + render_table(columns, rows) + "</div>"


def experiment_comparison_placeholder(summary: dict[str, Any], test_ic: dict[str, Any]) -> str:
    """Render the current baseline row plus future experiment columns."""
    # Use the baseline as the first comparable experiment row.
    rows = [
        [
            BENCHMARK_ID,
            "2026-05-18",
            "GruMlpRegressor",
            "14",
            "ret_vwap_exec_10",
            str(BEST_CHECKPOINT_ITER),
            format_report_value("test_ic", test_ic["mean"]),
            "spread + fee",
            "baseline reference",
        ],
        ["future_experiment", "", "", "", "", "", "", "", ""],
    ]
    headers = ["experiment", "date", "model", "features", "label", "best_iter", "test_ic", "cost_model", "verdict"]
    return '<div class="table-wrap">' + render_table(headers, rows) + "</div>"


def comparison_delta_table(comparison: dict[str, Any]) -> str:
    """Render current-vs-parent comparison with explicit delta columns."""
    # Skip metrics that are already emphasized elsewhere in the report.
    excluded_metrics = {"q95_q80_net_bps_per_turnover", "top_minus_bottom_bps", "high_liq_top_decile_net_proxy_bps"}

    # Convert the nested comparison payload into a compact baseline comparison.
    primary = dict(comparison["primary_metric"])
    rows = []
    if str(primary["name"]) not in excluded_metrics:
        rows.append([
            str(primary["name"]),
            format_report_value(str(primary["name"]), primary["baseline"]),
            format_report_value(str(primary["name"]), primary["experiment"]),
            format_report_value(str(primary["name"]), primary["absolute_change"]),
            "same baseline",
        ])
    for key, value in dict(comparison["secondary_metrics"]).items():
        if str(key) in excluded_metrics:
            continue
        rows.append([str(key), format_report_value(str(key), value), format_report_value(str(key), value), "0.0000", "reference"])
    return '<div class="table-wrap">' + render_table(["metric", "baseline", "current", "delta", "note"], rows) + "</div>"


def configure_report_plot_style() -> None:
    """Configure matplotlib for compact report figures."""
    # Prefer common CJK fonts when available so Chinese labels render correctly.
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def signed_colors(values: list[float]) -> list[str]:
    """Return red/green colors by value sign."""
    # Use one stable palette across all report figures.
    return ["#c53030" if float(value) >= 0 else "#2f855a" for value in list(values)]


def ascii_plot_label(raw_name: str) -> str:
    """Return one ASCII-safe plot label."""
    # Avoid CJK glyph warnings on machines without Chinese fonts.
    labels = {
        "top_decile_return_bps": "top decile return",
        "bottom_decile_return_bps": "bottom decile return",
        "top_minus_bottom_bps": "top-bottom spread",
        "high_liq_top_decile_net_proxy_bps": "high-liq top net proxy",
        "q95_q80_gross_daily_return_bps": "q95/q80 gross daily",
        "q95_q80_net_daily_return_bps": "q95/q80 net daily",
        "q95_q80_net_bps_per_turnover": "q95/q80 net bps/turnover",
    }
    return labels.get(str(raw_name), str(raw_name))


def save_headline_metrics_figure(summary: dict[str, Any], out_path: Path) -> Path:
    """Plot the main headline metrics as a signed bar chart."""
    # Select the metrics that explain the baseline in one glance.
    metrics = dict(summary["headline_metrics"])
    raw_fields = [
        "top_decile_return_bps",
        "bottom_decile_return_bps",
        "top_minus_bottom_bps",
        "high_liq_top_decile_net_proxy_bps",
        "q95_q80_gross_daily_return_bps",
        "q95_q80_net_daily_return_bps",
        "q95_q80_net_bps_per_turnover",
    ]
    values = [float(metrics[field]) for field in raw_fields]
    labels = [ascii_plot_label(field) for field in raw_fields]

    # Draw a horizontal bar chart with a zero line for cost-aware interpretation.
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.barh(labels, values, color=signed_colors(values))
    ax.axvline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Headline metric sign check")
    ax.set_xlabel("bps or bps/turnover")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_ic_summary_figure(train_ic: dict[str, Any], test_ic: dict[str, Any], out_path: Path) -> Path:
    """Plot train/test IC mean and ICIR."""
    # Prepare side-by-side IC and stability values.
    splits = ["train", "test"]
    mean_values = [float(train_ic["mean"]), float(test_ic["mean"])]
    icir_values = [float(train_ic["icir"]), float(test_ic["icir"])]

    # Draw two small panels so level and stability are not mixed on one axis.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(splits, mean_values, color=["#2b6cb0", "#2f855a"])
    axes[0].set_title("Daily IC mean")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(splits, icir_values, color=["#2b6cb0", "#2f855a"])
    axes[1].set_title("Daily ICIR")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_checkpoint_selection_figure(out_path: Path) -> Path:
    """Plot checkpoint selection metrics by iteration."""
    # Load checkpoint selector output.
    df = pd.read_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv").sort_values("iter")
    selected = df[df["selected"].astype(bool)].iloc[0]

    # Plot MSE and IC metrics on separate axes because scales differ.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(df["iter"], df["val/objective/mse"], marker="o", color="#2b6cb0")
    axes[0].axvline(float(selected["iter"]), color="#c53030", linestyle="--", linewidth=1.0)
    axes[0].set_title("Validation MSE by checkpoint")
    axes[0].set_xlabel("iter")
    axes[0].grid(alpha=0.25)
    axes[1].plot(df["iter"], df["val/quality/global_ic"], marker="o", label="global IC", color="#2f855a")
    axes[1].plot(df["iter"], df["val/quality/rank_ic"], marker="o", label="rank IC", color="#805ad5")
    axes[1].axvline(float(selected["iter"]), color="#c53030", linestyle="--", linewidth=1.0)
    axes[1].set_title("Validation IC by checkpoint")
    axes[1].set_xlabel("iter")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_signal_quality_figure(out_path: Path) -> Path:
    """Plot signal bucket return and hit rate."""
    # Load signal bucket metrics.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv")

    # Plot return bars and hit-rate line to show monotonicity and direction together.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(df["signal_bucket"].astype(str), df["mean_return_bps"], color=signed_colors(df["mean_return_bps"].astype(float).tolist()))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Signal bucket return and hit rate")
    ax.set_xlabel("signal bucket")
    ax.set_ylabel("mean return bps")
    ax.grid(axis="y", alpha=0.25)
    twin = ax.twinx()
    twin.plot(df["signal_bucket"].astype(str), df["hit_rate"], marker="o", color="#2b6cb0")
    twin.set_ylabel("hit rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_trading_cost_figure(out_path: Path) -> Path:
    """Plot gross, net and cost drag for the q95/q80 rule."""
    # Select the current headline trading rule.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv")
    row = df.query("strategy_name == 'q95_q80'").iloc[0]
    values = [
        float(row["gross_daily_return_bps"]),
        -float(row["spread_cost_bps"]),
        -float(row["fee_cost_bps"]),
        float(row["net_daily_return_bps"]),
    ]
    labels = ["gross daily return", "spread cost", "fee cost", "net daily return"]

    # Draw the cost bridge as signed bars.
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values, color=signed_colors(values))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("q95/q80 trading cost bridge")
    ax.set_ylabel("bps")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_time_bucket_figure(out_path: Path) -> Path:
    """Plot time bucket net bps per turnover."""
    # Load time bucket metrics ordered by intraday bucket.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv")
    values = df["net_bps_per_turnover"].astype(float).tolist()

    # Draw positive and negative buckets in different colors.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(df["time_bucket"].astype(str), values, color=signed_colors(values))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Time bucket net bps per turnover")
    ax.set_xlabel("time bucket")
    ax.set_ylabel("net bps/turnover")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_liquidity_top_bucket_figure(out_path: Path) -> Path:
    """Plot top signal bucket net proxy by liquidity bucket."""
    # Use signal bucket 10 as the strongest long-side bucket.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv")
    top = df[df["signal_bucket"].astype(int) == 10].sort_values("liq_bucket")
    values = top["entry_net_proxy_bps"].astype(float).tolist()

    # Draw liquidity sensitivity of the top signal bucket.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(top["liq_bucket"].astype(str), values, color=signed_colors(values))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Top signal bucket by liquidity")
    ax.set_xlabel("liquidity bucket")
    ax.set_ylabel("entry net proxy bps")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_prediction_target_distribution_figure(out_path: Path) -> Path:
    """Plot prediction and target distribution checkpoints from normalization metrics."""
    # Extract validation distribution quantiles from the normalization diagnostics.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv")
    mapping = dict(zip(df["metric_name"].astype(str), df["value"], strict=False))
    quantiles = ["p01", "p50", "p99"]
    pred_values = [float(mapping[f"val/dist/{quantile}/pred"]) for quantile in quantiles]
    target_values = [float(mapping[f"val/dist/{quantile}/target"]) for quantile in quantiles]

    # Plot prediction and target quantiles on the same axis.
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(quantiles))
    width = 0.36
    ax.bar(positions - width / 2, pred_values, width=width, label="prediction", color="#2b6cb0")
    ax.bar(positions + width / 2, target_values, width=width, label="target", color="#805ad5")
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(quantiles)
    ax.set_title("Prediction vs target distribution")
    ax.set_ylabel("normalized value")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_prediction_target_scale_by_checkpoint_figure(out_path: Path) -> Path:
    """Plot prediction-to-target scale ratio by checkpoint."""
    # Read checkpoint selector diagnostics.
    df = pd.read_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv")
    selected = df[df["selected"].astype(bool)].iloc[0]

    # Plot the scale ratio across checkpoints.
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["iter"], df["val/dist/pred_std_over_target_std"], marker="o", color="#2b6cb0")
    ax.axvline(float(selected["iter"]), color="#c53030", linestyle="--", linewidth=1.0)
    ax.axhline(1.0, color="#334155", linestyle=":", linewidth=1.0)
    ax.set_title("Prediction/target scale by checkpoint")
    ax.set_xlabel("iter")
    ax.set_ylabel("pred std / target std")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_bootstrap_ci_figure(out_path: Path) -> Path:
    """Plot bootstrap confidence intervals."""
    # Load bootstrap interval metrics.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv")
    labels = [ascii_plot_label(str(metric)) for metric in df["metric"].astype(str)]
    means = df["mean"].astype(float).to_numpy()
    lower = means - df["ci_025"].astype(float).to_numpy()
    upper = df["ci_975"].astype(float).to_numpy() - means

    # Draw horizontal error bars so long metric names remain readable.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.errorbar(means, labels, xerr=[lower, upper], fmt="o", color="#2b6cb0", ecolor="#718096", capsize=4)
    ax.axvline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Bootstrap confidence intervals")
    ax.set_xlabel("metric value")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_month_stability_figure(out_path: Path) -> Path:
    """Plot monthly signal spread and trading efficiency."""
    # Load month stability metrics in chronological order.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv")
    labels = df["month"].astype(str).tolist()

    # Draw signal spread bars and net efficiency line together.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    values = df["top_minus_bottom_bps"].astype(float).tolist()
    ax.bar(labels, values, color=signed_colors(values))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_title("Monthly signal spread and net efficiency")
    ax.set_xlabel("month")
    ax.set_ylabel("top-minus-bottom bps")
    ax.tick_params(axis="x", rotation=45)
    twin = ax.twinx()
    twin.plot(labels, df["q95_q80_net_bps_per_turnover"].astype(float), marker="o", color="#2b6cb0")
    twin.set_ylabel("q95/q80 net bps/turnover")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_regime_stability_figure(out_path: Path) -> Path:
    """Plot gross and net daily bps by regime."""
    # Load regime stability metrics.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv")
    labels = df["regime"].astype(str).tolist()
    positions = np.arange(len(labels))
    width = 0.36

    # Compare gross and net daily bps by regime.
    fig, ax = plt.subplots(figsize=(8.5, 4))
    gross = df["gross_daily_bps"].astype(float).to_numpy()
    net = df["net_daily_bps"].astype(float).to_numpy()
    ax.bar(positions - width / 2, gross, width=width, label="gross", color=signed_colors(gross.tolist()))
    ax.bar(positions + width / 2, net, width=width, label="net", color=signed_colors(net.tolist()))
    ax.axhline(0.0, color="#334155", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Regime gross and net daily bps")
    ax.set_ylabel("bps")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_volatility_bucket_stability_figure(out_path: Path) -> Path:
    """Plot IC by volatility bucket."""
    # Load volatility bucket stability metrics.
    df = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv")
    labels = df["volatility_bucket"].astype(str).tolist()

    # Compare Pearson IC and rank IC across volatility buckets.
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(labels, df["mean_ic"].astype(float), marker="o", label="mean_ic", color="#2b6cb0")
    ax.plot(labels, df["mean_rank_ic"].astype(float), marker="o", label="mean_rank_ic", color="#805ad5")
    ax.set_title("IC by volatility bucket")
    ax.set_xlabel("volatility bucket")
    ax.set_ylabel("IC")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def build_report_figures(summary: dict[str, Any], train_ic: dict[str, Any], test_ic: dict[str, Any]) -> dict[str, Path]:
    """Build all report-specific figures."""
    # Ensure the target figure directory exists.
    configure_report_plot_style()
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Render every figure from the existing benchmark tables.
    figures = {
        "ic_summary": save_ic_summary_figure(train_ic, test_ic, REPORT_FIGURE_DIR / "ic_summary_bar.png"),
        "checkpoint": save_checkpoint_selection_figure(REPORT_FIGURE_DIR / "checkpoint_selection_metrics.png"),
        "signal_quality": save_signal_quality_figure(REPORT_FIGURE_DIR / "signal_bucket_return_hit_rate.png"),
        "trading_cost": save_trading_cost_figure(REPORT_FIGURE_DIR / "trading_cost_bridge.png"),
        "time_bucket": save_time_bucket_figure(REPORT_FIGURE_DIR / "time_bucket_net_bps_per_turnover.png"),
        "liquidity_top": save_liquidity_top_bucket_figure(REPORT_FIGURE_DIR / "liquidity_top_signal_bucket.png"),
        "pred_target_distribution": save_prediction_target_distribution_figure(REPORT_FIGURE_DIR / "prediction_target_distribution.png"),
        "pred_target_scale": save_prediction_target_scale_by_checkpoint_figure(REPORT_FIGURE_DIR / "prediction_target_scale_by_checkpoint.png"),
        "bootstrap_ci": save_bootstrap_ci_figure(REPORT_FIGURE_DIR / "bootstrap_confidence_intervals.png"),
        "month_stability": save_month_stability_figure(REPORT_FIGURE_DIR / "month_stability_metrics.png"),
        "regime_stability": save_regime_stability_figure(REPORT_FIGURE_DIR / "regime_stability_metrics.png"),
        "volatility_stability": save_volatility_bucket_stability_figure(REPORT_FIGURE_DIR / "volatility_bucket_stability_metrics.png"),
    }
    return figures


def copy_training_post_eval_artifacts() -> dict[str, str]:
    """Copy existing IC and diagnostic artifacts into the benchmark bundle."""
    # Define the source-to-destination artifact map.
    out_dir = BENCHMARK_ROOT / "evaluation" / "model_ic"
    figure_dir = BENCHMARK_ROOT / "evaluation" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifact_map = {
        SOURCE_RUN_ROOT / "daily_ic_summary_train.yaml": out_dir / "daily_ic_summary_train.yaml",
        SOURCE_RUN_ROOT / "daily_ic_summary_test.yaml": out_dir / "daily_ic_summary_test.yaml",
        SOURCE_RUN_ROOT / "intraday_ic.csv": out_dir / "intraday_ic.csv",
        SOURCE_RUN_ROOT / "vol_rolling_ic.csv": out_dir / "vol_rolling_ic.csv",
        SOURCE_RUN_ROOT / "vol_rolling_ic.yaml": out_dir / "vol_rolling_ic.yaml",
        SOURCE_RUN_ROOT / "price_rolling_ic.csv": out_dir / "price_rolling_ic.csv",
        SOURCE_RUN_ROOT / "price_rolling_ic.yaml": out_dir / "price_rolling_ic.yaml",
        SOURCE_RUN_ROOT / "test_evaluation_report" / "annual_ic.csv": out_dir / "annual_ic.csv",
        SOURCE_RUN_ROOT / "test_evaluation_report" / "test_prediction_rank_turnover.csv": out_dir / "test_prediction_rank_turnover.csv",
        SOURCE_RUN_ROOT / "test_evaluation_report" / "test_prediction_rank_turnover.yaml": out_dir / "test_prediction_rank_turnover.yaml",
        SOURCE_RUN_ROOT / "test_evaluation_report" / "test_residual_diagnostics.yaml": out_dir / "test_residual_diagnostics.yaml",
        SOURCE_RUN_ROOT / "intraday_ic.png": figure_dir / "intraday_ic.png",
        SOURCE_RUN_ROOT / "vol_rolling_ic.png": figure_dir / "vol_rolling_ic.png",
        SOURCE_RUN_ROOT / "price_rolling_ic.png": figure_dir / "price_rolling_ic.png",
        SOURCE_RUN_ROOT / "test_evaluation_report" / "test_pred_vs_target_rank.png": figure_dir / "test_pred_vs_target_rank.png",
        SOURCE_RUN_ROOT / "test_evaluation_report" / "test_prediction_rank_turnover.png": figure_dir / "test_prediction_rank_turnover.png",
        SOURCE_RUN_ROOT / "test_evaluation_report" / "test_residual_diagnostics.png": figure_dir / "test_residual_diagnostics.png",
        SOURCE_RUN_ROOT / "test_evaluation_report" / "test_vol_rolling_ic.png": figure_dir / "test_vol_rolling_ic.png",
        SOURCE_RUN_ROOT / "test_evaluation_report" / "test_price_rolling_ic.png": figure_dir / "test_price_rolling_ic.png",
    }

    # Copy every artifact and record relative paths.
    copied: dict[str, str] = {}
    for src, dst in artifact_map.items():
        copy_file(Path(src), Path(dst))
        copied[Path(dst).stem] = Path(dst).relative_to(BENCHMARK_ROOT).as_posix()
    write_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "model_ic_artifacts.yaml", {"schema_version": 1, "artifacts": copied})
    return copied


def build_training_diagnostics() -> dict[str, str]:
    """Build offline training diagnostics from frozen checkpoints and TensorBoard scalars."""
    # Prepare the output directory.
    out_dir = BENCHMARK_ROOT / "train" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute parameter and update norms from retained checkpoints.
    ckpt_manifest = read_yaml(BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml")
    ckpt_rows: list[dict[str, Any]] = []
    prev_state: dict[str, torch.Tensor] | None = None
    prev_iter: int | None = None
    for record in sorted(list(ckpt_manifest["available_checkpoints"]), key=lambda item: int(item["iter"])):
        path = Path(str(record["path"]))
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = {str(key): value.detach().float().cpu() for key, value in dict(ckpt["model"]).items()}
        param_sq = float(sum(float((tensor * tensor).sum().item()) for tensor in state.values()))
        max_abs = float(max(float(tensor.abs().max().item()) for tensor in state.values()))
        update_norm = float("nan")
        update_norm_over_param_norm = float("nan")
        if prev_state is not None:
            diff_sq = float(sum(float(((state[key] - prev_state[key]) ** 2).sum().item()) for key in state.keys()))
            update_norm = float(diff_sq ** 0.5)
            update_norm_over_param_norm = float(update_norm / (param_sq ** 0.5))
        ckpt_rows.append(
            {
                "iter": int(record["iter"]),
                "checkpoint_path": path.as_posix(),
                "retained": bool(record["retained"]),
                "parameter_l2_norm": float(param_sq ** 0.5),
                "parameter_max_abs": max_abs,
                "update_l2_norm_from_prev_checkpoint": update_norm,
                "update_norm_over_param_norm": update_norm_over_param_norm,
                "prev_iter": prev_iter,
            }
        )
        prev_state = state
        prev_iter = int(record["iter"])
    norm_df = pd.DataFrame(ckpt_rows)
    norm_df.to_csv(out_dir / "checkpoint_parameter_update_norms.csv", index=False)

    # Record that historical gradient norm was not stored in this frozen run.
    write_yaml(
        out_dir / "gradient_norm_availability.yaml",
        {
            "schema_version": 1,
            "status": "not_available_from_frozen_run",
            "reason": "gradient tensors were not persisted during training; future runs should log train/grad/global_norm online.",
            "available_proxy": "train/diagnostics/checkpoint_parameter_update_norms.csv",
        },
    )

    # Build train-vs-val gap from TensorBoard scalars.
    scalars = pd.read_parquet(BENCHMARK_ROOT / "train" / "train_scalars.parquet")
    train_loss = scalars.loc[scalars["tag"].astype(str) == "train/objective/loss_mean", ["step", "value"]].rename(columns={"value": "train_loss_mean"}).sort_values("step")
    val_mse = scalars.loc[scalars["tag"].astype(str) == "val/objective/mse", ["step", "value"]].rename(columns={"value": "val_mse"}).sort_values("step")
    gap = pd.merge_asof(val_mse, train_loss, on="step", direction="backward")
    gap["val_minus_train"] = gap["val_mse"] - gap["train_loss_mean"]
    gap["val_over_train"] = gap["val_mse"] / gap["train_loss_mean"]
    gap.to_csv(out_dir / "train_val_gap.csv", index=False)

    # Summarize LR, AMP, and timing tags from final logged points.
    perf_rows: list[dict[str, Any]] = []
    for tag in ["train/optim/lr", "train/optim/amp_scale", "train/time/data_ms", "train/time/model_ms", "train/time/iter_ms", "train/objective/loss_mean"]:
        sub = scalars.loc[scalars["tag"].astype(str) == tag].sort_values("step")
        if sub.empty:
            continue
        tail = sub.tail(100)
        perf_rows.append(
            {
                "tag": tag,
                "last_step": int(sub.iloc[-1]["step"]),
                "last_value": float(sub.iloc[-1]["value"]),
                "tail100_mean": float(tail["value"].mean()),
                "tail100_std": float(tail["value"].std(ddof=0)),
                "min_value": float(sub["value"].min()),
                "max_value": float(sub["value"].max()),
            }
        )
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(out_dir / "train_runtime_scalar_summary.csv", index=False)
    write_yaml(out_dir / "train_runtime_scalar_summary.yaml", {"schema_version": 1, "rows": perf_rows})

    # Build checkpoint selector table with explicit promote/reject reasons.
    ckpt_metrics = pd.read_csv(BENCHMARK_ROOT / "train" / "checkpoint_metrics.csv")
    best_mse = float(ckpt_metrics["val/objective/mse"].min())
    selector = ckpt_metrics.copy()
    selector["selection_metric"] = "val/objective/mse"
    selector["metric_rank"] = selector["val/objective/mse"].rank(method="min", ascending=True).astype(int)
    selector["decision"] = np.where(selector["val/objective/mse"].astype(float) == best_mse, "promote", "reject")
    selector["reason"] = np.where(selector["decision"] == "promote", "minimum validation MSE among retained candidate checkpoints", "validation MSE is higher than selected checkpoint")
    selector.to_csv(out_dir / "checkpoint_selector_table.csv", index=False)

    return {
        "checkpoint_parameter_update_norms": "train/diagnostics/checkpoint_parameter_update_norms.csv",
        "gradient_norm_availability": "train/diagnostics/gradient_norm_availability.yaml",
        "train_val_gap": "train/diagnostics/train_val_gap.csv",
        "train_runtime_scalar_summary": "train/diagnostics/train_runtime_scalar_summary.csv",
        "checkpoint_selector_table": "train/diagnostics/checkpoint_selector_table.csv",
    }


def bootstrap_ci(values: np.ndarray, *, seed: int) -> dict[str, float]:
    """Compute one deterministic bootstrap confidence interval."""
    # Draw bootstrap means with a fixed seed.
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(values, dtype=float)
    samples = rng.choice(arr, size=(2000, int(arr.size)), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci_025": float(np.quantile(samples, 0.025)),
        "ci_500": float(np.quantile(samples, 0.500)),
        "ci_975": float(np.quantile(samples, 0.975)),
        "sample_count": int(arr.size),
        "bootstrap_samples": 2000,
    }


def build_bootstrap_confidence_intervals() -> dict[str, Any]:
    """Build bootstrap confidence intervals for key daily metrics."""
    # Load daily signal and trading metrics.
    signal_daily = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "model_signal_monotonicity_daily.csv")
    trading_daily = pd.read_csv(BENCHMARK_ROOT / "backtest" / "variants" / "q95_q80" / "percentile_hysteresis_daily.csv")
    trading_daily["net_bps_per_turnover"] = trading_daily["net_return"].astype(float) * 1e4 / trading_daily["turnover"].astype(float)

    # Compute deterministic bootstrap intervals.
    rows = [
        {"metric": "top_minus_bottom_bps", **bootstrap_ci(signal_daily["top_minus_bottom"].to_numpy(dtype=float) * 1e4, seed=7)},
        {"metric": "q95_q80_net_bps_per_turnover", **bootstrap_ci(trading_daily["net_bps_per_turnover"].to_numpy(dtype=float), seed=11)},
        {"metric": "q95_q80_net_daily_return_bps", **bootstrap_ci(trading_daily["net_return"].to_numpy(dtype=float) * 1e4, seed=13)},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv", index=False)
    payload = {"schema_version": 1, "metrics_csv": "evaluation/bootstrap_confidence_intervals.csv", "rows": rows}
    write_yaml(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.yaml", payload)
    return payload


def build_stability_diagnostics() -> dict[str, str]:
    """Build month, regime, and volatility stability diagnostics."""
    # Build monthly signal and trading stability.
    signal_daily = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "model_signal_monotonicity_daily.csv")
    trading_daily = pd.read_csv(BENCHMARK_ROOT / "backtest" / "variants" / "q95_q80" / "percentile_hysteresis_daily.csv")
    signal_daily["month"] = (signal_daily["date"].astype(int) // 100).astype(int)
    trading_daily["month"] = (trading_daily["date"].astype(int) // 100).astype(int)
    trading_daily["net_bps_per_turnover"] = trading_daily["net_return"].astype(float) * 1e4 / trading_daily["turnover"].astype(float)
    monthly = signal_daily.groupby("month").agg(
        top_minus_bottom_bps=("top_minus_bottom", lambda col: float(np.mean(col) * 1e4)),
        positive_top_minus_bottom_ratio=("top_minus_bottom", lambda col: float(np.mean(np.asarray(col) > 0.0))),
    ).reset_index()
    monthly_trading = trading_daily.groupby("month").agg(
        q95_q80_net_daily_bps=("net_return", lambda col: float(np.mean(col) * 1e4)),
        q95_q80_net_bps_per_turnover=("net_bps_per_turnover", "mean"),
        daily_turnover=("turnover", "mean"),
    ).reset_index()
    monthly = monthly.merge(monthly_trading, on="month", how="outer")
    monthly.to_csv(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv", index=False)

    # Build realized-regime stability from daily absolute gross return terciles.
    trading_daily["regime"] = pd.qcut(trading_daily["gross_return"].abs(), q=3, labels=["low_abs_gross", "mid_abs_gross", "high_abs_gross"])
    regime = trading_daily.groupby("regime", observed=True).agg(
        day_count=("date", "count"),
        gross_daily_bps=("gross_return", lambda col: float(np.mean(col) * 1e4)),
        net_daily_bps=("net_return", lambda col: float(np.mean(col) * 1e4)),
        net_bps_per_turnover=("net_bps_per_turnover", "mean"),
        turnover=("turnover", "mean"),
    ).reset_index()
    regime.to_csv(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv", index=False)

    # Build volatility bucket stability from existing rolling IC curve.
    vol = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "model_ic" / "vol_rolling_ic.csv")
    vol["volatility_bucket"] = pd.qcut(vol["group_center_rank"], q=5, labels=["vol_q1", "vol_q2", "vol_q3", "vol_q4", "vol_q5"])
    vol_bucket = vol.groupby("volatility_bucket", observed=True).agg(
        row_count=("count", "sum"),
        mean_ic=("mean_ic", "mean"),
        mean_rank_ic=("mean_rank_ic", "mean"),
        center_rank_min=("group_center_rank", "min"),
        center_rank_max=("group_center_rank", "max"),
    ).reset_index()
    vol_bucket.to_csv(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv", index=False)
    return {
        "month_stability_metrics": "evaluation/month_stability_metrics.csv",
        "regime_stability_metrics": "evaluation/regime_stability_metrics.csv",
        "volatility_bucket_stability_metrics": "evaluation/volatility_bucket_stability_metrics.csv",
    }


def build_label_availability_coverage() -> dict[str, Any]:
    """Explain label availability coverage for joined prediction rows."""
    # Resolve enriched chunk files.
    enriched_manifest = BENCHMARK_ROOT / "predictions" / "enriched_test_predictions_manifest.yaml"
    enriched_paths = parquet_paths_from_manifest(enriched_manifest, "benchmark_chunk_files")
    enriched_sql_paths = duckdb_path_list(enriched_paths)

    # Aggregate availability indicators.
    con = duckdb.connect()
    con.execute("set threads to 8")
    row = con.execute(
        f"""
        select
          count(*) filter (where prediction_available)::BIGINT as joined_rows,
          count(*) filter (where prediction_available and ret_vwap_exec_10 is null)::BIGINT as null_horizon_rows,
          count(*) filter (where prediction_available and ret_vwap_exec_10 is not null)::BIGINT as usable_label_rows,
          count(*) filter (where prediction_available and ret_vwap_exec_10 is null and not fillable_vwap)::BIGINT as null_horizon_not_fillable_vwap,
          count(*) filter (where prediction_available and ret_vwap_exec_10 is null and not fillable_open)::BIGINT as null_horizon_not_fillable_open,
          count(*) filter (where prediction_available and ret_vwap_exec_10 is null and is_limit_up_all_day)::BIGINT as null_horizon_all_day_limit_up,
          count(*) filter (where prediction_available and ret_vwap_exec_10 is null and is_limit_down_all_day)::BIGINT as null_horizon_all_day_limit_down,
          count(*) filter (where prediction_available and ret_vwap_exec_10 is null and (entry_vwap_is_up_limit or entry_vwap_is_down_limit or exit_vwap_is_up_limit or exit_vwap_is_down_limit))::BIGINT as null_horizon_entry_or_exit_vwap_limit
        from read_parquet({enriched_sql_paths})
        """
    ).fetchone()
    con.close()

    # Write row-wise coverage attribution.
    joined = int(row[0])
    names = [
        "joined_rows",
        "null_horizon_rows",
        "usable_label_rows",
        "null_horizon_not_fillable_vwap",
        "null_horizon_not_fillable_open",
        "null_horizon_all_day_limit_up",
        "null_horizon_all_day_limit_down",
        "null_horizon_entry_or_exit_vwap_limit",
    ]
    rows = [{"category": name, "row_count": int(value), "row_ratio_of_joined": float(int(value) / joined)} for name, value in zip(names, row)]
    out = pd.DataFrame(rows)
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "label_availability_coverage.csv", index=False)
    payload = {"schema_version": 1, "metrics_csv": "evaluation/label_availability_coverage.csv", "rows": rows}
    write_yaml(BENCHMARK_ROOT / "evaluation" / "label_availability_coverage.yaml", payload)
    return payload


def build_turnover_and_capacity_diagnostics() -> dict[str, str]:
    """Build turnover decomposition and capacity sensitivity tables."""
    # Approximate turnover decomposition from the bar-level q95_q80 output.
    bar = pd.read_csv(BENCHMARK_ROOT / "backtest" / "variants" / "q95_q80" / "percentile_hysteresis_10min_bar.csv")
    min_time = int(bar["time"].min())
    max_time = int(bar["time"].max())
    bar["turnover_component"] = np.where(bar["time"].astype(int) == min_time, "initial_open_proxy", np.where(bar["time"].astype(int) == max_time, "final_close_proxy", "intraday_rebalance_proxy"))
    turnover = bar.groupby("turnover_component").agg(
        row_count=("date", "count"),
        mean_turnover=("turnover", "mean"),
        mean_spread_cost_bps=("spread_cost", lambda col: float(np.mean(col) * 1e4)),
        mean_fee_cost_bps=("fee_cost", lambda col: float(np.mean(col) * 1e4)),
        mean_gross_return_bps=("gross_return", lambda col: float(np.mean(col) * 1e4)),
    ).reset_index()
    turnover.to_csv(BENCHMARK_ROOT / "evaluation" / "turnover_decomposition.csv", index=False)
    write_yaml(
        BENCHMARK_ROOT / "evaluation" / "turnover_decomposition.yaml",
        {
            "schema_version": 1,
            "status": "computed_from_bar_turnover_proxy",
            "note": "Existing bar output has aggregate turnover only; open/close/change components are approximated by first, last, and middle bars.",
            "metrics_csv": "evaluation/turnover_decomposition.csv",
        },
    )

    # Copy the existing cost-aware capacity sensitivity summary when available.
    src = Path("/data-cache/nn/trade_plan_experiments/0516_cost_aware_entry/cost_aware_entry_experiment_summary.csv")
    capacity = pd.read_csv(src)
    keep_cols = [
        "strategy_name",
        "sizing_method",
        "turnover_budget",
        "no_trade_band",
        "entry_cost_penalty",
        "max_liq_bucket",
        "gross_daily_return",
        "daily_turnover",
        "net_10m_daily_return",
        "capacity_10bps",
        "capacity_20bps",
    ]
    capacity[keep_cols].to_csv(BENCHMARK_ROOT / "evaluation" / "capacity_sensitivity_metrics.csv", index=False)
    write_yaml(
        BENCHMARK_ROOT / "evaluation" / "capacity_sensitivity_metrics.yaml",
        {
            "schema_version": 1,
            "status": "copied_from_existing_cost_aware_entry_experiment",
            "source": src.as_posix(),
            "metrics_csv": "evaluation/capacity_sensitivity_metrics.csv",
        },
    )
    return {
        "turnover_decomposition": "evaluation/turnover_decomposition.csv",
        "capacity_sensitivity_metrics": "evaluation/capacity_sensitivity_metrics.csv",
    }


def build_short_side_diagnostics() -> dict[str, Any]:
    """Build avoid-bad-stock diagnostics from bottom signal buckets."""
    # Load signal bucket and liquidity tables.
    signal = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv")
    liquidity = pd.read_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv")
    bottom = signal.loc[signal["signal_bucket"].astype(int) == 1].iloc[0]
    top = signal.loc[signal["signal_bucket"].astype(int) == 10].iloc[0]
    high_liq_bottom = liquidity.loc[(liquidity["liq_bucket"].astype(int) == 1) & (liquidity["signal_bucket"].astype(int) == 1)].iloc[0]
    high_liq_top = liquidity.loc[(liquidity["liq_bucket"].astype(int) == 1) & (liquidity["signal_bucket"].astype(int) == 10)].iloc[0]

    # Write a compact diagnostic table.
    rows = [
        {
            "diagnostic": "bottom_decile_avoidance_alpha_bps",
            "value": float(-bottom["mean_return_bps"]),
            "note": "Positive value means avoiding bottom decile can help long-only construction.",
        },
        {
            "diagnostic": "top_decile_long_alpha_bps",
            "value": float(top["mean_return_bps"]),
            "note": "Direct long-only top-decile gross signal.",
        },
        {
            "diagnostic": "high_liq_bottom_avoidance_net_proxy_bps",
            "value": float(-high_liq_bottom["entry_net_proxy_bps"]),
            "note": "Positive value means high-liquidity bad-stock avoidance is potentially valuable.",
        },
        {
            "diagnostic": "high_liq_top_net_proxy_bps",
            "value": float(high_liq_top["entry_net_proxy_bps"]),
            "note": "High-liquidity top-decile net proxy.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(BENCHMARK_ROOT / "evaluation" / "short_side_avoid_bad_stock_metrics.csv", index=False)
    payload = {"schema_version": 1, "metrics_csv": "evaluation/short_side_avoid_bad_stock_metrics.csv", "rows": rows}
    write_yaml(BENCHMARK_ROOT / "evaluation" / "short_side_avoid_bad_stock_metrics.yaml", payload)
    return payload


def basic_info_rows() -> dict[str, list[tuple[str, str]]]:
    """Build basic data, model, and method rows for reports."""
    # Load frozen contracts.
    meta = read_yaml(BENCHMARK_ROOT / "data" / "npz_meta.yaml")
    norm = read_yaml(BENCHMARK_ROOT / "data" / "normalization_contract.yaml")
    model = read_yaml(BENCHMARK_ROOT / "model" / "effective_model_summary.yaml")
    train_cfg = read_yaml(BENCHMARK_ROOT / "train" / "train_config.yaml")
    ckpt = read_yaml(BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml")

    # Summarize data ranges and storage.
    dates = dict(meta["dates"])
    groups = dict(meta["storage"]["groups"])
    data_rows = [
        ("raw_date_range", f"{meta['prep_config']['start_trade_date']} -> {meta['prep_config']['end_trade_date']}"),
        ("train_dates", f"{dates['train'][0]} -> {dates['train'][-1]} ({len(dates['train'])} days)"),
        ("val_dates", f"{dates['val'][0]} -> {dates['val'][-1]} ({len(dates['val'])} days)"),
        ("test_dates", f"{dates['test'][0]} -> {dates['test'][-1]} ({len(dates['test'])} days)"),
        ("train_rows", f"{int(groups['train']['rows']):,}"),
        ("val_rows", f"{int(groups['val']['rows']):,}"),
        ("test_rows", f"{int(groups['test']['rows']):,}"),
        ("feature_dim", str(int(groups["train"]["feature_dim"]))),
        ("features", ", ".join(list(meta["feature_names"]))),
        ("label", str(norm["label"])),
        ("feature_normalization", f"{norm['feature_transform']['stock_norm']['type']} / scope={norm['feature_transform']['stock_norm']['scope']}"),
        ("label_normalization", f"{norm['label_transform']['type']} / scope={norm['label_transform']['scope']}"),
    ]

    # Summarize model architecture and initialization.
    model_rows = [
        ("model_class", str(model["model_class"])),
        ("input_tensor", str(model["input_tensor"])),
        ("gru_input_size", str(model["gru"]["input_size"])),
        ("gru_hidden_size", str(model["gru"]["hidden_size"])),
        ("gru_num_layers", str(model["gru"]["num_layers"])),
        ("gru_bidirectional", str(model["gru"]["bidirectional"])),
        ("mlp_effective_hidden_width", str(model["mlp"]["effective_hidden_width"])),
        ("trainable_parameters", f"{int(model['trainable_parameters']):,}"),
        ("parameter_initialization", "PyTorch default initialization for nn.GRU and nn.Linear; no custom init in prediction_nn2/model.py."),
        ("best_checkpoint", "model/best_checkpoint.pt"),
        ("state_dict_summary", "model/state_dict_summary.yaml"),
    ]

    # Summarize training and evaluation methods.
    method_rows = [
        ("optimizer", str(train_cfg["optimizer"])),
        ("criterion", str(train_cfg["criterion"])),
        ("final_iter", str(train_cfg["stage_manifest"]["final_iter"])),
        ("best_iter", str(train_cfg["stage_manifest"]["best_it"])),
        ("checkpoint_selection", f"{train_cfg['checkpoint_selection']['rule']} {train_cfg['checkpoint_selection']['metric']}"),
        ("checkpoint_saved_to", str(ckpt["best_checkpoint"]["path"])),
        ("prediction_manifest", "predictions/inference_test_manifest.yaml"),
        ("post_train_processing", "eval_train/eval_test/inference_test manifests, IC reports, signal diagnostics, portfolio backtest."),
        ("backtest_method", "long_only_10min_percentile_hysteresis, spread + fee cost model."),
        ("evaluation_contract", "join validation, IC, signal/liquidity/time/extreme/normalization/trading metrics, comparison against parent."),
    ]
    return {"data": data_rows, "model": model_rows, "method": method_rows}


def markdown_table_from_csv(path: Path, n_rows: int) -> str:
    """Render a small CSV preview as markdown."""
    # Load and truncate the table.
    df = pd.read_csv(Path(path)).head(int(n_rows))

    # Build a compact markdown table.
    headers = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(value) for value in list(row)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def full_report_markdown(summary: dict[str, Any]) -> str:
    """Build the full benchmark evaluation report markdown."""
    # Load core manifests and metric summaries.
    tb_manifest = read_yaml(BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml")
    ckpt_manifest = read_yaml(BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml")
    join_validation = read_yaml(BENCHMARK_ROOT / "evaluation" / "join_validation.yaml")
    train_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_train.yaml")
    test_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_test.yaml")
    rank_turnover = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "test_prediction_rank_turnover.yaml")
    residual = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "test_residual_diagnostics.yaml")
    vol_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "vol_rolling_ic.yaml")
    price_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "price_rolling_ic.yaml")
    comparison = read_yaml(BENCHMARK_ROOT / "evaluation" / "comparison_against_parent.yaml")
    basic = basic_info_rows()

    # Derive retained training artifact details.
    event_files = list(tb_manifest["event_files"])
    tags = list(tb_manifest["tags"])
    retained = [
        "TensorBoard event files and exported scalar parquet.",
        "Checkpoint metrics for validation checkpoint selection.",
        "Best checkpoint metadata and checkpoint file.",
        "Train stage manifest and train config summary.",
        "Train loss curve exported from TensorBoard.",
        "Train/test/inference prediction manifests.",
    ]

    # Compose the report body.
    metrics = dict(summary["headline_metrics"])
    lines = [
        "# Prediction-NN-2 Baseline 完整 Evaluation 报告",
        "",
        "## 结论摘要",
        "",
        f"- benchmark id: `{BENCHMARK_ID}`.",
        f"- replay status: `{read_yaml(BENCHMARK_ROOT / 'replay.yaml')['status']}`.",
        f"- top-minus-bottom: `{float(metrics['top_minus_bottom_bps']):.3f}` bps.",
        f"- high-liq top decile net proxy: `{float(metrics['high_liq_top_decile_net_proxy_bps']):.3f}` bps.",
        f"- q95_q80 net daily return: `{float(metrics['q95_q80_net_daily_return_bps']):.3f}` bps.",
        f"- q95_q80 net bps per turnover: `{float(metrics['q95_q80_net_bps_per_turnover']):.3f}`.",
        f"- best time bucket by net bps/turnover: `{metrics['best_time_bucket_by_net_bps_per_turnover']}`.",
        f"- worst time bucket by net bps/turnover: `{metrics['worst_time_bucket_by_net_bps_per_turnover']}`.",
        "",
        "## Key Takeaways",
        "",
        f"- test daily Pearson IC mean 为 `{float(test_ic['pearson_ic']['mean']):.4f}`, ICIR 为 `{float(test_ic['pearson_ic']['icir']):.2f}`, 统计信号稳定。",
        f"- signal top-minus-bottom 为 `{float(metrics['top_minus_bottom_bps']):.2f}` bps, bottom decile 明显弱于 top decile。",
        f"- q95_q80 gross daily return 为 `{float(metrics['q95_q80_gross_daily_return_bps']):.2f}` bps, 但 net daily return 为 `{float(metrics['q95_q80_net_daily_return_bps']):.2f}` bps, cost 是主要约束。",
        f"- best time bucket 是 `{metrics['best_time_bucket_by_net_bps_per_turnover']}`, worst time bucket 是 `{metrics['worst_time_bucket_by_net_bps_per_turnover']}`。",
        "- 后续实验应复用本报告的 comparison schema, 每次追加 experiment row 和 parent delta。",
        "",
        "## Experiment Comparison Template",
        "",
        "| experiment | date | model | features | label | best_iter | test_ic | top-bottom_bps | net_bps/turnover | cost_model | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| {BENCHMARK_ID} | 2026-05-18 | GruMlpRegressor | 14 | ret_vwap_exec_10 | {BEST_CHECKPOINT_ITER} | {float(test_ic['pearson_ic']['mean']):.4f} | {float(metrics['top_minus_bottom_bps']):.2f} | {float(metrics['q95_q80_net_bps_per_turnover']):.4f} | spread + fee | baseline reference |",
        "| future_experiment |  |  |  |  |  |  |  |  |  |  |",
        "",
        "## 已覆盖的 Evaluation",
        "",
        "- checkpoint selector, parameter norm, update norm, train-val gap, runtime scalar summary。",
        "- join validation, IC, signal/liquidity/time/extreme/normalization diagnostics。",
        "- bootstrap CI, month/regime/volatility stability, label availability, turnover, capacity, short-side diagnostics。",
        "- trading metrics and comparison against parent baseline。",
        "",
        "## 训练中保留的数据",
        "",
    ]
    lines.extend([f"- {item}" for item in retained])
    lines.extend(
        [
            "",
            "## 基础信息",
            "",
            "### 数据与 Normalization",
            "",
            "\n".join([f"- {key}: `{value}`." for key, value in basic["data"]]),
            "",
            "### 模型与参数",
            "",
            "\n".join([f"- {key}: `{value}`." for key, value in basic["model"]]),
            "",
            "### 训练、后处理与回测方法",
            "",
            "\n".join([f"- {key}: `{value}`." for key, value in basic["method"]]),
            "",
            "## TensorBoard 与训练监控",
            "",
            f"- TensorBoard source dir: `{tb_manifest['source_dir']}`.",
            f"- exported scalar parquet: `train/train_scalars.parquet`.",
            f"- scalar rows: `{tb_manifest['scalar_rows']}`.",
            f"- event file count: `{len(event_files)}`.",
            f"- best checkpoint iter: `{ckpt_manifest['best_checkpoint_iter']}`.",
            f"- checkpoint selection metric: `{ckpt_manifest['best_selection_metric']}`.",
            "",
            "TensorBoard scalar tags:",
            "",
        ]
    )
    lines.extend([f"- `{tag}`" for tag in tags])
    lines.extend(
        [
            "",
            "Checkpoint metrics preview:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "train" / "checkpoint_metrics.csv", 8),
            "",
            "Checkpoint selector table:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv", 8),
            "",
            "Parameter / update norm diagnostics:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_parameter_update_norms.csv", 8),
            "",
            "Train vs val gap:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_val_gap.csv", 8),
            "",
            "Runtime scalar summary:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_runtime_scalar_summary.csv", 12),
            "",
            "## Join Validation",
            "",
            "```yaml",
            yaml.safe_dump(join_validation, sort_keys=False, allow_unicode=True).strip(),
            "```",
            "",
            "## IC Evaluation",
            "",
            f"- train daily Pearson IC mean: `{float(train_ic['pearson_ic']['mean']):.6f}`, ICIR: `{float(train_ic['pearson_ic']['icir']):.3f}`, positive ratio: `{float(train_ic['pearson_ic']['positive_ratio']):.3f}`.",
            f"- test daily Pearson IC mean: `{float(test_ic['pearson_ic']['mean']):.6f}`, ICIR: `{float(test_ic['pearson_ic']['icir']):.3f}`, positive ratio: `{float(test_ic['pearson_ic']['positive_ratio']):.3f}`.",
            f"- annual test Pearson IC: see `evaluation/model_ic/annual_ic.csv`.",
            "",
            "Intraday IC preview:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "model_ic" / "intraday_ic.csv", 8),
            "",
            "## Rolling IC 与 Rank Turnover",
            "",
            f"- volatility rolling IC max: `{float(vol_ic['pearson_ic_max']['value']):.6f}` at rank `{float(vol_ic['pearson_ic_max']['group_center_rank']):.3f}`.",
            f"- volatility rolling IC min: `{float(vol_ic['pearson_ic_min']['value']):.6f}` at rank `{float(vol_ic['pearson_ic_min']['group_center_rank']):.3f}`.",
            f"- price rolling IC max: `{float(price_ic['pearson_ic_max']['value']):.6f}` at rank `{float(price_ic['pearson_ic_max']['group_center_rank']):.3f}`.",
            f"- price rolling IC min: `{float(price_ic['pearson_ic_min']['value']):.6f}` at rank `{float(price_ic['pearson_ic_min']['group_center_rank']):.3f}`.",
            f"- mean rank turnover: `{float(rank_turnover['mean_rank_turnover']):.6f}`.",
            f"- highest rank turnover time: `{rank_turnover['highest_turnover_time']}`.",
            f"- lowest rank turnover time: `{rank_turnover['lowest_turnover_time']}`.",
            "",
            "## Residual Diagnostics",
            "",
            f"- residual mean: `{float(residual['residual_mean']):.8f}`.",
            f"- residual std: `{float(residual['residual_std']):.8f}`.",
            f"- residual skew: `{float(residual['residual_skew']):.3f}`.",
            f"- residual kurtosis: `{float(residual['residual_kurtosis']):.3f}`.",
            f"- RMSE: `{float(residual['rmse']):.8f}`.",
            "",
            "## Signal Bucket Evaluation",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv", 12),
            "",
            "## Liquidity Evaluation",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv", 18),
            "",
            "## Time Bucket Attribution",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv", 20),
            "",
            "## Extreme Value Diagnostics",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.csv", 20),
            "",
            "## Normalization Diagnostics",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", 30),
            "",
            "## Bootstrap Confidence Intervals",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv", 10),
            "",
            "## Stability Diagnostics",
            "",
            "Month stability:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv", 20),
            "",
            "Regime stability:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv", 10),
            "",
            "Volatility bucket stability:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv", 10),
            "",
            "## Label Availability Coverage",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "label_availability_coverage.csv", 12),
            "",
            "## Turnover, Capacity, Short-side Diagnostics",
            "",
            "Turnover decomposition:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "turnover_decomposition.csv", 10),
            "",
            "Capacity sensitivity:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "capacity_sensitivity_metrics.csv", 12),
            "",
            "Short-side / avoid-bad-stock diagnostics:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "short_side_avoid_bad_stock_metrics.csv", 10),
            "",
            "## Trading 与 Backtest",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv", 8),
            "",
            "Alpha per turnover:",
            "",
            markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "alpha_per_turnover.csv", 8),
            "",
            "## Comparison Against Parent",
            "",
            "```yaml",
            yaml.safe_dump(comparison, sort_keys=False, allow_unicode=True).strip(),
            "```",
            "",
            "## 可复用代码",
            "",
            "本报告由 `portfolio_backtest/scripts/run_baseline_evaluation.py` 生成。下次实验应复用该脚本中的 contract、metrics 和 report 函数，只替换 `BENCHMARK_ROOT`、`SOURCE_RUN_ROOT`、`SOURCE_SIGNAL_ROOT`、`SOURCE_TRADING_ROOT` 和 `BENCHMARK_ID` 对应的新实验路径。",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_kv_table(rows: list[tuple[str, Any]]) -> str:
    """Render key-value rows as a markdown table."""
    # Convert every value through the report formatter for consistent scanning.
    lines = ["| raw field name | value |", "| --- | --- |"]
    for key, value in list(rows):
        lines.append(f"| `{key}` | `{format_report_value(str(key), value)}` |")
    return "\n".join(lines)


def markdown_rows_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render in-memory rows as a markdown table."""
    # Build the markdown header from raw field names.
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    # Format every row through the report formatter.
    for row in list(rows):
        values = [format_report_cell(str(col), value) for col, value in zip(list(headers), list(row), strict=True)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def markdown_prediction_target_scale_table() -> str:
    """Render prediction/target scale check as markdown."""
    # Build a two-column raw field/value table.
    return markdown_rows_table(["raw field name", "value"], [[field, value] for field, value in prediction_target_scale_rows()])


def markdown_field_notes(raw_fields: list[str], section_note: str) -> str:
    """Render field explanations as markdown bullets."""
    # Start with the section-level note before individual field definitions.
    lines = ["字段解释:", "", f"- {section_note}"]

    # Emit one raw-field definition per requested field.
    for raw_field in list(raw_fields):
        field = FIELD_DEFINITIONS.get(str(raw_field))
        if field is None:
            lines.append(f"- `{raw_field}`: 当前 raw field 暂未登记详细解释。")
            continue
        lines.append(f"- `{raw_field}`: {field['explanation']} 单位: {field['unit']}; 读数方向: {field['direction']}.")
    return "\n".join(lines)


def markdown_comparison_delta_table(comparison: dict[str, Any]) -> str:
    """Render baseline delta comparison as a markdown table."""
    # Skip metrics that the comparison section should no longer display.
    excluded_metrics = {"q95_q80_net_bps_per_turnover", "top_minus_bottom_bps", "high_liq_top_decile_net_proxy_bps"}

    # Convert the primary metric first when it is still eligible.
    primary = dict(comparison["primary_metric"])
    rows = []
    if str(primary["name"]) not in excluded_metrics:
        rows.append([
            str(primary["name"]),
            format_report_value(str(primary["name"]), primary["baseline"]),
            format_report_value(str(primary["name"]), primary["experiment"]),
            format_report_value(str(primary["name"]), primary["absolute_change"]),
            "same baseline",
        ])

    # Append secondary metrics using the baseline self-comparison convention.
    for key, value in dict(comparison["secondary_metrics"]).items():
        if str(key) in excluded_metrics:
            continue
        rows.append([str(key), format_report_value(str(key), value), format_report_value(str(key), value), "0.0000", "reference"])

    # Render the compact markdown table.
    lines = ["| metric | baseline | current | delta | note |", "| --- | --- | --- | --- | --- |"]
    for metric, baseline, current, delta, note in rows:
        lines.append(f"| `{metric}` | `{baseline}` | `{current}` | `{delta}` | {note} |")
    return "\n".join(lines)


def outline_aligned_report_markdown(summary: dict[str, Any]) -> str:
    """Build the benchmark report markdown with the same outline as HTML."""
    # Load the same artifacts used by the HTML report.
    basic = basic_info_rows()
    tb_manifest = read_yaml(BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml")
    join_validation = read_yaml(BENCHMARK_ROOT / "evaluation" / "join_validation.yaml")
    comparison = read_yaml(BENCHMARK_ROOT / "evaluation" / "comparison_against_parent.yaml")
    train_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_train.yaml")["pearson_ic"]
    test_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_test.yaml")["pearson_ic"]
    pooled_ic = source_pooled_ic_values()

    # Assemble data pipeline rows.
    data_basic_fields = ["train_dates", "val_dates", "test_dates", "feature_dim", "features", "label", "feature_normalization", "label_normalization"]
    join_basic_fields = ["prediction_rows", "feature_rows"]
    basic_data = dict(basic["data"])
    data_basic_rows = [(field, basic_data[field]) for field in data_basic_fields] + [(field, join_validation[field]) for field in join_basic_fields]
    normalization_contract_rows = [row for row in data_basic_rows if row[0] in {"feature_normalization", "label_normalization", "label", "feature_dim", "features"}]

    # Compose markdown in the outline order.
    lines = [
        "# Prediction-NN-2 Baseline 完整 Evaluation 报告",
        "",
        "## 1. Executive Summary",
        "",
        markdown_kv_table(
            [
                ("benchmark", BENCHMARK_ID),
                ("selected_checkpoint", BEST_CHECKPOINT_ITER),
                ("test_ic", test_ic["mean"]),
                ("train_ic", train_ic["mean"]),
                ("test_pooled_ic", pooled_ic["test_pooled_ic"]),
                ("train_pooled_ic", pooled_ic["train_pooled_ic"]),
                ("model_class", "GruMlpRegressor"),
            ]
        ),
        "",
        markdown_field_notes(["benchmark", "selected_checkpoint", "test_ic", "train_ic", "test_pooled_ic", "train_pooled_ic", "model_class"], "Executive Summary 展示当前 benchmark 的核心统计信号和模型上下文。"),
        "",
        "## 2. Data Pipeline",
        "",
        "### 2.1. Data Basics",
        "",
        markdown_kv_table(data_basic_rows),
        "",
        markdown_field_notes([key for key, _ in data_basic_rows], "Data Basics 展示本次实验的数据范围、feature/label 设定和 prediction/feature 数据规模。"),
        "",
        "### 2.2. Data Normalization",
        "",
        "Normalization contract:",
        "",
        markdown_kv_table(normalization_contract_rows),
        "",
        "Normalization diagnostics:",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", 30),
        "",
        "Prediction/target scale check:",
        "",
        markdown_prediction_target_scale_table(),
        "",
        "Figure:",
        "",
        "- `evaluation/figures/test_residual_diagnostics.png`: Residual distribution diagnostics.",
        "",
        markdown_field_notes(["feature_normalization", "label_normalization", "label", "feature_dim", "features", "metric_name", "value", "split", "note", "prediction_mean", "prediction_std", "target_mean", "target_std", "pred_std_over_target_std", "prediction_p01", "prediction_p50", "prediction_p99", "target_p01", "target_p50", "target_p99"], "Data Normalization 解释 feature/label normalization contract, 并检查 prediction/target scale 是否合理。"),
        "",
        "## 3. Experiment Comparison",
        "",
        "### 3.1. Experiment Rows",
        "",
        "| experiment | date | model | features | label | best_iter | test_ic | cost_model | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| {BENCHMARK_ID} | 2026-05-18 | GruMlpRegressor | 14 | ret_vwap_exec_10 | {BEST_CHECKPOINT_ITER} | {format_report_value('test_ic', test_ic['mean'])} | spread + fee | baseline reference |",
        "| future_experiment |  |  |  |  |  |  |  |  |",
        "",
        markdown_field_notes(["experiment", "date", "model", "features", "label", "best_iter", "test_ic", "cost_model", "verdict"], "Experiment Rows 预留后续实验对比位置。"),
        "",
        "### 3.2. Metric Delta Against Baseline",
        "",
        markdown_comparison_delta_table(comparison),
        "",
        "<details><summary>raw comparison YAML</summary>",
        "",
        "```yaml",
        yaml.safe_dump(comparison, sort_keys=False, allow_unicode=True).strip(),
        "```",
        "",
        "</details>",
        "",
        markdown_field_notes(["metric", "baseline", "current", "delta", "note", "daily_turnover"], "Metric Delta Against Baseline 比较当前 experiment 与 baseline。"),
        "",
        "## 4. Evaluation",
        "",
        "### 4.1. IC Summary",
        "",
        markdown_rows_table(
            ["split", "count", "mean_ic", "std_ic", "t_stat", "positive_ratio", "icir", "pooled_ic", "pooled_rank_ic"],
            [
                ["train", train_ic["count"], train_ic["mean"], train_ic["std"], train_ic["t_stat"], train_ic["positive_ratio"], train_ic["icir"], pooled_ic["train_pooled_ic"], float("nan")],
                ["test", test_ic["count"], test_ic["mean"], test_ic["std"], test_ic["t_stat"], test_ic["positive_ratio"], test_ic["icir"], pooled_ic["test_pooled_ic"], float("nan")],
            ],
        ),
        "",
        markdown_field_notes(["split", "count", "mean_ic", "std_ic", "t_stat", "positive_ratio", "icir", "pooled_ic", "pooled_rank_ic"], "IC Summary 用来判断模型样本外统计信号是否稳定。"),
        "",
        "### 4.2. Signal Quality",
        "",
        "#### Bucket Construction Principle",
        "",
        "每个 evaluation 截面内按 `prediction` 从低到高排序, 再切成 `signal_bucket`. `signal_bucket=1` 代表最低 prediction 分组, `signal_bucket=10` 代表最高 prediction 分组。读表时应检查 `mean_prediction` 是否随 bucket 单调上升, 以及 `mean_return_bps` 和 `hit_rate` 是否随 bucket 改善。",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv", 12),
        "",
        markdown_field_notes(["signal_bucket", "row_count", "mean_prediction", "mean_return_bps", "t_stat", "hit_rate", "mean_spread_bps"], "Signal Quality 检查 prediction 排序后收益是否随 bucket 单调变强。"),
        "",
        "### 4.3. Trading And Cost",
        "",
        "#### Cost Model",
        "",
        "本报告使用 spread + fee 成本模型: spread 成本用于近似吃单或跨越 bid-ask 的交易摩擦, fee 成本用于近似显性交易费用。选择该模型是因为它足够保守且可解释, 可以把统计 signal 的 gross 收益和真实执行后的 net 收益区分开, 避免只看 IC 或 bucket return 时高估可交易性。",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv", 8),
        "",
        markdown_field_notes(["strategy_name", "open_quantile", "close_quantile", "gross_daily_return_bps", "net_daily_return_bps", "gross_sharpe", "net_sharpe", "max_drawdown", "daily_turnover", "spread_cost_bps", "fee_cost_bps", "gross_bps_per_turnover", "net_bps_per_turnover", "active_names"], "Trading And Cost 判断统计 signal 转成交易后是否还能覆盖 spread 和 fee。"),
        "",
        "### 4.4. Time And Liquidity Attribution",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv", 20),
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv", 18),
        "",
        markdown_field_notes(["time_bucket", "gross_return_bps", "turnover", "net_return_bps", "liq_bucket", "entry_spread_cost_bps", "entry_fee_cost_bps", "entry_net_proxy_bps", "mean_signal_amount"], "Time 与 liquidity attribution 用来定位哪些时段和流动性区域贡献收益。"),
        "",
        "### 4.5. Robustness Diagnostics",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv", 10),
        "",
        "Figure: `reports/figures/bootstrap_confidence_intervals.png`.",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv", 20),
        "",
        "Figure: `reports/figures/month_stability_metrics.png`.",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv", 10),
        "",
        "Figure: `reports/figures/regime_stability_metrics.png`.",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv", 10),
        "",
        "Figure: `reports/figures/volatility_bucket_stability_metrics.png`.",
        "",
        "Figure: `evaluation/figures/vol_rolling_ic.png`.",
        "",
        "Figure: `evaluation/figures/price_rolling_ic.png`.",
        "",
        markdown_field_notes(["metric", "mean", "std", "ci_025", "ci_500", "ci_975", "sample_count", "bootstrap_samples", "month", "positive_top_minus_bottom_ratio", "regime", "volatility_bucket", "mean_rank_ic"], "Robustness Diagnostics 检查核心指标在 bootstrap、月份、market regime 和 volatility bucket 下是否稳定。"),
        "",
        "## 5. Train Metrics",
        "",
        "### 5.1. Checkpoint Selection",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv", 8),
        "",
        markdown_field_notes(["iter", "selected", "val/objective/mse", "val/quality/global_ic", "val/quality/rank_ic", "val/dist/pred_std_over_target_std", "metric_rank", "decision"], "Checkpoint Selection 解释为什么选中 best checkpoint。"),
        "",
        "### 5.2. Model And Method",
        "",
        markdown_kv_table(basic["model"] + basic["method"]),
        "",
        markdown_field_notes([key for key, _ in basic["model"] + basic["method"]], "Model And Method 记录模型结构、训练目标、checkpoint 选择规则和 evaluation contract。"),
        "",
        "### 5.3. Training Monitoring",
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_val_gap.csv", 8),
        "",
        markdown_table_from_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_runtime_scalar_summary.csv", 12),
        "",
        f"- TensorBoard source dir: `{tb_manifest['source_dir']}`.",
        f"- scalar rows: `{tb_manifest['scalar_rows']}`.",
        "",
        markdown_field_notes(["step", "val_mse", "train_loss_mean", "val_minus_train", "val_over_train", "tag", "last_step", "last_value", "tail100_mean", "tail100_std", "min_value", "max_value"], "Training Monitoring 检查 train/validation loss gap、runtime scalar 和参数更新范数。"),
        "",
        "## 6. Appendix Diagnostics",
        "",
        "- `intraday IC table`: `evaluation/model_ic/intraday_ic.csv`.",
        "- `extreme value metrics`: `evaluation/extreme_value_metrics.csv`.",
        "- `normalization metrics`: `evaluation/normalization_metrics.csv`.",
        "- `label availability coverage`: `evaluation/label_availability_coverage.csv`.",
        "- `turnover decomposition`: `evaluation/turnover_decomposition.csv`.",
        "- `capacity sensitivity`: `evaluation/capacity_sensitivity_metrics.csv`.",
        "- `short-side diagnostics`: `evaluation/short_side_avoid_bad_stock_metrics.csv`.",
    ]
    return "\n".join(lines) + "\n"


def write_full_benchmark_report(summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write the full benchmark report into benchmark and repo report dirs."""
    # Build and persist the markdown report.
    markdown = outline_aligned_report_markdown(summary)
    report_md = BENCHMARK_ROOT / "reports" / "full_evaluation_report.md"
    report_md.write_text(markdown, encoding="utf-8")

    # Render a self-contained HTML report with native sections and tables.
    basic = basic_info_rows()
    tb_manifest = read_yaml(BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml")
    join_validation = read_yaml(BENCHMARK_ROOT / "evaluation" / "join_validation.yaml")
    comparison = read_yaml(BENCHMARK_ROOT / "evaluation" / "comparison_against_parent.yaml")
    train_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_train.yaml")["pearson_ic"]
    test_ic = read_yaml(BENCHMARK_ROOT / "evaluation" / "model_ic" / "daily_ic_summary_test.yaml")["pearson_ic"]
    pooled_ic = source_pooled_ic_values()
    figures = build_report_figures(summary, dict(train_ic), dict(test_ic))

    # Derive the first-screen summary values.
    summary_cards = [
        ("benchmark", BENCHMARK_ID, "current frozen baseline"),
        ("selected_checkpoint", str(BEST_CHECKPOINT_ITER), "minimum validation MSE"),
        ("test_ic", format_report_value("test_ic", test_ic["mean"]), f"ICIR {format_report_value('icir', test_ic['icir'])}"),
        ("train_ic", format_report_value("train_ic", train_ic["mean"]), f"ICIR {format_report_value('icir', train_ic['icir'])}"),
        ("test_pooled_ic", format_report_value("test_pooled_ic", pooled_ic["test_pooled_ic"]), "pooled Pearson IC"),
        ("train_pooled_ic", format_report_value("train_pooled_ic", pooled_ic["train_pooled_ic"]), "pooled Pearson IC"),
        ("model_class", "GruMlpRegressor", "model architecture"),
    ]

    # Write concise investment-style takeaways.
    takeaways = [
        f"模型 IC 在 test split 上为 {format_report_value('test_ic', test_ic['mean'])}, daily ICIR 为 {format_report_value('icir', test_ic['icir'])}, 统计信号稳定。",
        f"当前报告使用 iter {BEST_CHECKPOINT_ITER} 作为 selected checkpoint, 选择规则是 minimum validation MSE。",
        "数据范围、normalization 与 prediction/feature 规模在 Data Pipeline 中展开, signal 与 trading 指标在 Evaluation 中展开。",
        "本报告已预留 Experiment Comparison 行, 后续实验可以直接追加到同一 schema。",
    ]

    # Build the data pipeline rows from data contracts and join validation.
    data_basic_fields = ["train_dates", "val_dates", "test_dates", "feature_dim", "features", "label", "feature_normalization", "label_normalization"]
    join_basic_fields = ["prediction_rows", "feature_rows"]
    basic_data = dict(basic["data"])
    data_basic_rows = [(field, basic_data[field]) for field in data_basic_fields] + [(field, format_report_value(field, join_validation[field])) for field in join_basic_fields]
    normalization_contract_rows = [row for row in data_basic_rows if row[0] in {"feature_normalization", "label_normalization", "label", "feature_dim", "features"}]

    # Render sections from high-level conclusion to appendices.
    data_pipeline_body = (
        render_subsection(
            "2.1. Data Basics",
            render_value_rows(data_basic_rows)
            + render_field_notes(data_basic_fields + join_basic_fields, "Data Basics 展示本次实验的数据范围、feature/label 设定和 prediction/feature 数据规模。"),
        )
        + render_subsection(
            "2.2. Data Normalization",
            render_block_title("normalization contract")
            + render_value_rows(normalization_contract_rows)
            + render_block_title("normalization_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", 30)
            + render_block_title("prediction/target scale check")
            + prediction_target_scale_table()
            + render_embedded_figure("Residual Diagnostics", BENCHMARK_ROOT / "evaluation" / "figures" / "test_residual_diagnostics.png", "Residual distribution diagnostics.")
            + render_field_notes(["feature_normalization", "label_normalization", "label", "feature_dim", "features", "metric_name", "value", "split", "note", "prediction_mean", "prediction_std", "target_mean", "target_std", "pred_std_over_target_std", "prediction_p01", "prediction_p50", "prediction_p99", "target_p01", "target_p50", "target_p99"], "Data Normalization 解释 feature/label normalization contract, 并检查 prediction/target scale 是否合理。"),
        )
    )
    experiment_body = (
        render_subsection(
            "3.1. Experiment Rows",
            experiment_comparison_placeholder(summary, dict(test_ic))
            + render_field_notes(["experiment", "date", "model", "features", "label", "best_iter", "test_ic", "cost_model", "verdict"], "Experiment Rows 预留后续实验对比位置。后续新实验应使用同一 schema 追加一行。"),
        )
        + render_subsection(
            "3.2. Metric Delta Against Baseline",
            comparison_delta_table(comparison)
            + render_details_code("raw comparison YAML", yaml.safe_dump(comparison, sort_keys=False, allow_unicode=True).strip())
            + render_field_notes(["metric", "baseline", "current", "delta", "note", "test_ic", "mean_ic", "std_ic", "t_stat", "ic_positive_ratio", "icir", "top_decile_return_bps", "daily_turnover"], "Metric Delta Against Baseline 比较当前 experiment 与 baseline。当前报告是 baseline self-comparison, delta 为 0。")
        )
    )
    evaluation_body = (
        render_subsection(
            "4.1. IC Summary",
            ic_summary_table(dict(train_ic), dict(test_ic))
            + render_embedded_figure("IC Summary Bar", figures["ic_summary"], "Train/test mean_ic 与 icir 对比。")
            + render_embedded_figure("Intraday IC", BENCHMARK_ROOT / "evaluation" / "figures" / "intraday_ic.png", "Train/test intraday IC.")
            + render_field_notes(["split", "count", "mean_ic", "std_ic", "t_stat", "positive_ratio", "icir", "pooled_ic", "pooled_rank_ic"], "IC Summary 用来判断模型样本外统计信号是否稳定。test IC 越高越好, ICIR 越高说明 daily IC 的波动更小。"),
        )
        + render_subsection(
            "4.2. Signal Quality",
            render_block_title("Bucket Construction Principle")
            + render_field_notes([], "每个 evaluation 截面内按 prediction 从低到高排序, 再切成 signal_bucket。signal_bucket=1 代表最低 prediction 分组, signal_bucket=10 代表最高 prediction 分组。读表时应检查 mean_prediction 是否随 bucket 单调上升, 以及 mean_return_bps 和 hit_rate 是否随 bucket 改善。")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv", 12)
            + render_embedded_figure("Signal Bucket Return Hit Rate", figures["signal_quality"], "每个 signal_bucket 的 mean_return_bps 和 hit_rate。")
            + render_embedded_figure("Signal Decile Return", BENCHMARK_ROOT / "evaluation" / "figures" / "model_signal_decile_return.png", "Signal decile return.")
            + render_embedded_figure("Signal Top Minus Bottom", BENCHMARK_ROOT / "evaluation" / "figures" / "model_signal_top_minus_bottom.png", "Daily top-minus-bottom signal spread.")
            + render_field_notes(["signal_bucket", "row_count", "mean_prediction", "mean_return_bps", "t_stat", "hit_rate", "mean_spread_bps", "top_minus_bottom_bps"], "Signal Quality 检查 prediction 排序后收益是否随 bucket 单调变强。当前 bottom decile 很弱, top-bottom spread 为正, 说明 signal 有明显排序能力。"),
        )
        + render_subsection(
            "4.3. Trading And Cost",
            render_block_title("Cost Model")
            + render_field_notes([], "本报告使用 spread + fee 成本模型: spread 成本用于近似吃单或跨越 bid-ask 的交易摩擦, fee 成本用于近似显性交易费用。选择该模型是因为它足够保守且可解释, 可以把统计 signal 的 gross 收益和真实执行后的 net 收益区分开, 避免只看 IC 或 bucket return 时高估可交易性。")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv", 8)
            + render_embedded_figure("Trading Cost Bridge", figures["trading_cost"], "q95/q80 策略从 gross 到 net 的成本拆解。")
            + render_embedded_figure("Trading Comparison", BENCHMARK_ROOT / "backtest" / "figures" / "percentile_hysteresis_baseline_comparison.png", "Trading baseline comparison.")
            + render_embedded_figure("Net Wealth Curve", BENCHMARK_ROOT / "backtest" / "figures" / "percentile_hysteresis_baseline_net_wealth_curve.png", "Net wealth curve.")
            + render_field_notes(["strategy_name", "open_quantile", "close_quantile", "gross_daily_return_bps", "net_daily_return_bps", "gross_sharpe", "net_sharpe", "max_drawdown", "daily_turnover", "spread_cost_bps", "fee_cost_bps", "gross_bps_per_turnover", "net_bps_per_turnover", "active_names"], "Trading And Cost 判断统计 signal 转成交易后是否还能覆盖 spread 和 fee。当前 gross 为正但 net 为负, 说明交易规则或成本控制仍需改进。"),
        )
        + render_subsection(
            "4.4. Time And Liquidity Attribution",
            render_block_title("time_bucket_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv", 20)
            + render_embedded_figure("Time Bucket Net Bps Per Turnover", figures["time_bucket"], "按 time_bucket 展示 net_bps_per_turnover。")
            + render_block_title("liquidity_bucket_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv", 18)
            + render_embedded_figure("Liquidity Top Signal Bucket", figures["liquidity_top"], "Top signal_bucket 在不同 liq_bucket 下的 entry_net_proxy_bps。")
            + render_embedded_figure("Liquidity Heatmap", BENCHMARK_ROOT / "evaluation" / "figures" / "model_signal_liquidity_heatmap.png", "Liquidity x signal diagnostics.")
            + render_field_notes(["time_bucket", "row_count", "gross_return_bps", "turnover", "spread_cost_bps", "fee_cost_bps", "net_return_bps", "gross_bps_per_turnover", "net_bps_per_turnover", "active_names", "liq_bucket", "signal_bucket", "entry_spread_cost_bps", "entry_fee_cost_bps", "entry_net_proxy_bps", "mean_signal_amount", "mean_spread_bps"], "Time 与 liquidity attribution 用来定位哪些时段和流动性区域贡献收益, 哪些区域需要过滤或降低 turnover。"),
        )
        + render_subsection(
            "4.5. Robustness Diagnostics",
            render_block_title("bootstrap_confidence_intervals.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv", 10)
            + render_embedded_figure("Bootstrap Confidence Intervals", figures["bootstrap_ci"], "Bootstrap 区间展示核心指标的不确定性。")
            + render_block_title("month_stability_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv", 20)
            + render_embedded_figure("Month Stability Metrics", figures["month_stability"], "按月份展示 top_minus_bottom_bps 和 q95/q80 net bps per turnover。")
            + render_block_title("regime_stability_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv", 10)
            + render_embedded_figure("Regime Stability Metrics", figures["regime_stability"], "按 market regime 对比 gross_daily_bps 与 net_daily_bps。")
            + render_block_title("volatility_bucket_stability_metrics.csv")
            + render_csv_table(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv", 10)
            + render_embedded_figure("Volatility Bucket Stability Metrics", figures["volatility_stability"], "按 volatility_bucket 对比 mean_ic 与 mean_rank_ic。")
            + render_embedded_figure("Vol Rolling IC", BENCHMARK_ROOT / "evaluation" / "figures" / "vol_rolling_ic.png", "IC by volatility rank.")
            + render_embedded_figure("Price Rolling IC", BENCHMARK_ROOT / "evaluation" / "figures" / "price_rolling_ic.png", "IC by price rank.")
            + render_field_notes(["metric", "mean", "std", "ci_025", "ci_500", "ci_975", "sample_count", "bootstrap_samples", "month", "positive_top_minus_bottom_ratio", "q95_q80_net_daily_bps", "q95_q80_net_bps_per_turnover", "daily_turnover", "regime", "day_count", "gross_daily_bps", "net_daily_bps", "net_bps_per_turnover", "turnover", "volatility_bucket", "row_count", "mean_ic", "mean_rank_ic", "center_rank_min", "center_rank_max"], "Robustness Diagnostics 检查核心指标在 bootstrap、月份、market regime 和 volatility bucket 下是否稳定。区间跨 0 的指标需要谨慎。"),
        )
    )
    train_body = (
        render_subsection(
            "5.1. Checkpoint Selection",
            checkpoint_summary_table()
            + render_embedded_figure("Checkpoint Selection Metrics", figures["checkpoint"], "val/objective/mse 和 IC 随 iter 的变化, 红线为 selected checkpoint。")
            + render_field_notes(["iter", "selected", "val/objective/mse", "val/quality/global_ic", "val/quality/rank_ic", "val/dist/pred_std_over_target_std", "metric_rank", "decision"], "Checkpoint Selection 解释为什么选中 iter 140000。selector 以 validation MSE 为主, 同时保留 IC 与 prediction scale 作为 sanity check。")
            + render_details_table("full checkpoint selector", render_csv_table(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv", 8)),
        )
        + render_subsection(
            "5.2. Model And Method",
            render_block_title("Model")
            + render_value_rows(basic["model"])
            + render_block_title("Method")
            + render_value_rows(basic["method"])
            + render_field_notes([row[0] for row in basic["model"] + basic["method"]], "Model And Method 记录模型结构、训练目标、checkpoint 选择规则和 evaluation contract, 用于复现实验。"),
        )
        + render_subsection(
            "5.3. Training Monitoring",
            render_block_title("train_val_gap.csv")
            + render_csv_table(BENCHMARK_ROOT / "train" / "diagnostics" / "train_val_gap.csv", 8)
            + render_block_title("train_runtime_scalar_summary.csv")
            + render_csv_table(BENCHMARK_ROOT / "train" / "diagnostics" / "train_runtime_scalar_summary.csv", 12)
            + render_embedded_figure("Train Loss", BENCHMARK_ROOT / "train" / "train_loss_curve.png", "TensorBoard loss curve.")
            + render_details_table("parameter and update norm", render_csv_table(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_parameter_update_norms.csv", 8))
            + render_details_table("TensorBoard storage", render_value_rows([("source_dir", str(tb_manifest["source_dir"])), ("scalar_parquet", "train/train_scalars.parquet"), ("scalar_rows", str(tb_manifest["scalar_rows"])), ("tags", ", ".join(list(tb_manifest["tags"])))]))
            + render_field_notes(["step", "val_mse", "train_loss_mean", "val_minus_train", "val_over_train", "tag", "last_step", "last_value", "tail100_mean", "tail100_std", "min_value", "max_value"], "Training Monitoring 检查 train/validation loss gap、runtime scalar 和参数更新范数, 用于发现训练不稳定或过拟合迹象。"),
        )
    )
    appendix_body = (
        render_details_table("intraday IC table", render_csv_table(BENCHMARK_ROOT / "evaluation" / "model_ic" / "intraday_ic.csv", 12))
        + render_details_table("extreme value metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.csv", 20))
        + render_details_table("normalization metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", 30))
        + render_details_table("label availability coverage", render_csv_table(BENCHMARK_ROOT / "evaluation" / "label_availability_coverage.csv", 12))
        + render_details_table("turnover decomposition", render_csv_table(BENCHMARK_ROOT / "evaluation" / "turnover_decomposition.csv", 10))
        + render_details_table("capacity sensitivity", render_csv_table(BENCHMARK_ROOT / "evaluation" / "capacity_sensitivity_metrics.csv", 12))
        + render_details_table("short-side diagnostics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "short_side_avoid_bad_stock_metrics.csv", 10))
        + render_field_notes([], "Appendix Diagnostics 保留长表和辅助诊断, 用于追查主线指标背后的数据质量、normalization、turnover、capacity 和 short-side 行为。")
    )
    sections = [
        render_section("1. Executive Summary", render_summary_cards(summary_cards) + render_takeaways(takeaways)),
        render_section("2. Data Pipeline", data_pipeline_body),
        render_section("3. Experiment Comparison", experiment_body),
        render_section("4. Evaluation", evaluation_body),
        render_section("5. Train Metrics", train_body),
        render_section("6. Appendix Diagnostics", appendix_body),
    ]
    html = build_page("Prediction-NN-2 Baseline 完整 Evaluation 报告", "训练中监测、IC、signal diagnostics、normalization、extreme values 与 backtest 汇总。", sections)
    report_html = BENCHMARK_ROOT / "reports" / "full_evaluation_report.html"
    report_html.write_text(html, encoding="utf-8")

    # Copy the lightweight public reports into repo report/benchmarks.
    copy_file(report_md, REPO_BENCHMARK_REPORT_DIR / "current_baseline_full_evaluation_report.md")
    copy_file(report_html, REPO_BENCHMARK_REPORT_DIR / "current_baseline_full_evaluation_report.html")
    return report_md, report_html


def write_train_monitoring_html() -> Path:
    """Write the training monitoring HTML report."""
    # Load train artifacts.
    tb_manifest = read_yaml(BENCHMARK_ROOT / "train" / "tensorboard_manifest.yaml")
    checkpoint_manifest = read_yaml(BENCHMARK_ROOT / "train" / "checkpoint_manifest.yaml")

    # Render training monitoring sections.
    sections = [
        render_section("TensorBoard Manifest", render_code_block(yaml.safe_dump(tb_manifest, sort_keys=False, allow_unicode=True))),
        render_section("Checkpoint Manifest", render_code_block(yaml.safe_dump(checkpoint_manifest, sort_keys=False, allow_unicode=True))),
        render_section("Checkpoint Metrics", render_csv_table(BENCHMARK_ROOT / "train" / "checkpoint_metrics.csv", 12)),
        render_section("Train Loss Curve", render_embedded_figure("Train Loss Curve", BENCHMARK_ROOT / "train" / "train_loss_curve.png", "TensorBoard train/objective/loss_mean exported as PNG.")),
    ]
    html = build_page("Baseline Train Monitoring", "TensorBoard and checkpoint monitoring for the frozen baseline.", sections)
    out = BENCHMARK_ROOT / "reports" / "train_monitoring.html"
    out.write_text(html, encoding="utf-8")
    return out


def write_model_signal_evaluation_html(summary: dict[str, Any]) -> Path:
    """Write the model signal evaluation HTML report."""
    # Render model signal sections.
    sections = [
        render_section("Headline Metrics", render_value_rows([(key, str(value)) for key, value in dict(summary["headline_metrics"]).items()])),
        render_section("Join Validation", render_code_block((BENCHMARK_ROOT / "evaluation" / "join_validation.yaml").read_text(encoding="utf-8"))),
        render_section("Signal Bucket Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv", 12)),
        render_section("Liquidity Bucket Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv", 18)),
        render_section("Time Bucket Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv", 30)),
        render_section("Extreme Value Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.csv", 20)),
        render_section("Normalization Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.csv", 30)),
        render_section("Signal Decile Figure", render_embedded_figure("Signal Decile Return", BENCHMARK_ROOT / "evaluation" / "figures" / "model_signal_decile_return.png", "Existing signal decile return diagnostics.")),
        render_section("Liquidity Figure", render_embedded_figure("Liquidity Heatmap", BENCHMARK_ROOT / "evaluation" / "figures" / "model_signal_liquidity_heatmap.png", "Existing liquidity x signal diagnostics.")),
    ]
    html = build_page("Baseline Model Signal Evaluation", "Post-training statistical evaluation for the frozen baseline.", sections)
    out = BENCHMARK_ROOT / "reports" / "model_signal_evaluation.html"
    out.write_text(html, encoding="utf-8")
    return out


def write_trading_evaluation_html() -> Path:
    """Write the trading evaluation HTML report."""
    # Render trading sections.
    sections = [
        render_section("Trading Rule Metrics", render_csv_table(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv", 12)),
        render_section("Alpha Per Turnover", render_csv_table(BENCHMARK_ROOT / "evaluation" / "alpha_per_turnover.csv", 12)),
        render_section("Trading Baseline Report", render_code_block((BENCHMARK_ROOT / "reports" / "trading_baseline_report.md").read_text(encoding="utf-8"))),
        render_section("Strategy Comparison", render_embedded_figure("Strategy Comparison", BENCHMARK_ROOT / "backtest" / "figures" / "percentile_hysteresis_baseline_comparison.png", "Existing percentile hysteresis comparison.")),
        render_section("Net Wealth Curve", render_embedded_figure("Net Wealth Curve", BENCHMARK_ROOT / "backtest" / "figures" / "percentile_hysteresis_baseline_net_wealth_curve.png", "Existing net wealth curve.")),
    ]
    html = build_page("Baseline Trading Evaluation", "Trading-rule evaluation for the frozen baseline.", sections)
    out = BENCHMARK_ROOT / "reports" / "trading_evaluation.html"
    out.write_text(html, encoding="utf-8")
    return out


def write_evaluation_card_html(summary: dict[str, Any], comparison: dict[str, Any]) -> Path:
    """Write the top-level evaluation card HTML report."""
    # Render the top-level report card.
    sections = [
        render_section("Headline Metrics", render_value_rows([(key, str(value)) for key, value in dict(summary["headline_metrics"]).items()])),
        render_section("Evaluation Summary", render_code_block(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))),
        render_section("Comparison Against Parent", render_code_block(yaml.safe_dump(comparison, sort_keys=False, allow_unicode=True))),
        render_section("Train Monitoring", render_code_block("reports/train_monitoring.html")),
        render_section("Model Signal Evaluation", render_code_block("reports/model_signal_evaluation.html")),
        render_section("Trading Evaluation", render_code_block("reports/trading_evaluation.html")),
    ]
    html = build_page("Baseline Evaluation Card", "Complete baseline evaluation entry point.", sections)
    out = BENCHMARK_ROOT / "reports" / "evaluation_card.html"
    out.write_text(html, encoding="utf-8")
    copy_file(out, REPO_BENCHMARK_REPORT_DIR / "current_baseline_evaluation_card.html")
    return out


def update_benchmark_and_replay(summary: dict[str, Any]) -> dict[str, Any]:
    """Update benchmark and replay manifests with complete evaluation outputs."""
    # Update benchmark.yaml with complete evaluation pointers.
    benchmark = read_yaml(BENCHMARK_ROOT / "benchmark.yaml")
    benchmark["evaluation"] = {
        "evaluation_input_manifest": "evaluation/evaluation_input_manifest.yaml",
        "join_validation": "evaluation/join_validation.yaml",
        "evaluation_summary": "evaluation/evaluation_summary.yaml",
        "signal_bucket_metrics": "evaluation/signal_bucket_metrics.csv",
        "liquidity_bucket_metrics": "evaluation/liquidity_bucket_metrics.csv",
        "time_bucket_metrics": "evaluation/time_bucket_metrics.csv",
        "extreme_value_metrics": "evaluation/extreme_value_metrics.yaml",
        "normalization_metrics": "evaluation/normalization_metrics.yaml",
        "trading_rule_metrics": "evaluation/trading_rule_metrics.csv",
        "comparison_against_parent": "evaluation/comparison_against_parent.yaml",
        "bootstrap_confidence_intervals": "evaluation/bootstrap_confidence_intervals.yaml",
        "month_stability_metrics": "evaluation/month_stability_metrics.csv",
        "regime_stability_metrics": "evaluation/regime_stability_metrics.csv",
        "volatility_bucket_stability_metrics": "evaluation/volatility_bucket_stability_metrics.csv",
        "label_availability_coverage": "evaluation/label_availability_coverage.yaml",
        "turnover_decomposition": "evaluation/turnover_decomposition.yaml",
        "capacity_sensitivity_metrics": "evaluation/capacity_sensitivity_metrics.yaml",
        "short_side_avoid_bad_stock_metrics": "evaluation/short_side_avoid_bad_stock_metrics.yaml",
    }
    benchmark["reports"]["train_monitoring_html"] = "reports/train_monitoring.html"
    benchmark["reports"]["model_signal_evaluation_html"] = "reports/model_signal_evaluation.html"
    benchmark["reports"]["trading_evaluation_html"] = "reports/trading_evaluation.html"
    benchmark["reports"]["evaluation_card_html"] = "reports/evaluation_card.html"
    benchmark["reports"]["full_evaluation_report_md"] = "reports/full_evaluation_report.md"
    benchmark["reports"]["full_evaluation_report_html"] = "reports/full_evaluation_report.html"
    benchmark["known_gaps"] = ["validation prediction manifest is not materialized in the current run"]
    benchmark["primary_metrics"].update(dict(summary["headline_metrics"]))
    write_yaml(BENCHMARK_ROOT / "benchmark.yaml", benchmark)

    # Build the complete replay gate.
    checks = {
        "evaluation_input_manifest_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "evaluation_input_manifest.yaml"),
        "join_validation_passed": read_yaml(BENCHMARK_ROOT / "evaluation" / "join_validation.yaml")["status"] == "passed",
        "evaluation_summary_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "evaluation_summary.yaml"),
        "signal_bucket_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "signal_bucket_metrics.csv"),
        "liquidity_bucket_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "liquidity_bucket_metrics.csv"),
        "time_bucket_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "time_bucket_metrics.csv"),
        "extreme_value_metrics_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "extreme_value_metrics.yaml"),
        "normalization_metrics_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "normalization_metrics.yaml"),
        "trading_rule_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "trading_rule_metrics.csv"),
        "comparison_against_parent_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "comparison_against_parent.yaml"),
        "bootstrap_confidence_intervals_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "bootstrap_confidence_intervals.yaml"),
        "month_stability_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "month_stability_metrics.csv"),
        "regime_stability_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "regime_stability_metrics.csv"),
        "volatility_bucket_stability_metrics_readable": readable_csv(BENCHMARK_ROOT / "evaluation" / "volatility_bucket_stability_metrics.csv"),
        "label_availability_coverage_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "label_availability_coverage.yaml"),
        "turnover_decomposition_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "turnover_decomposition.yaml"),
        "capacity_sensitivity_metrics_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "capacity_sensitivity_metrics.yaml"),
        "short_side_avoid_bad_stock_metrics_readable": readable_yaml(BENCHMARK_ROOT / "evaluation" / "short_side_avoid_bad_stock_metrics.yaml"),
        "checkpoint_selector_table_readable": readable_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_selector_table.csv"),
        "checkpoint_parameter_update_norms_readable": readable_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "checkpoint_parameter_update_norms.csv"),
        "train_val_gap_readable": readable_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_val_gap.csv"),
        "train_runtime_scalar_summary_readable": readable_csv(BENCHMARK_ROOT / "train" / "diagnostics" / "train_runtime_scalar_summary.csv"),
        "train_monitoring_html_exists": (BENCHMARK_ROOT / "reports" / "train_monitoring.html").exists(),
        "model_signal_evaluation_html_exists": (BENCHMARK_ROOT / "reports" / "model_signal_evaluation.html").exists(),
        "trading_evaluation_html_exists": (BENCHMARK_ROOT / "reports" / "trading_evaluation.html").exists(),
        "evaluation_card_html_exists": (BENCHMARK_ROOT / "reports" / "evaluation_card.html").exists(),
        "full_evaluation_report_md_exists": (BENCHMARK_ROOT / "reports" / "full_evaluation_report.md").exists(),
        "full_evaluation_report_html_exists": (BENCHMARK_ROOT / "reports" / "full_evaluation_report.html").exists(),
    }
    payload = {
        "schema_version": 2,
        "benchmark_id": BENCHMARK_ID,
        "status": "passed" if all(bool(value) for value in checks.values()) else "failed",
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
    }
    write_yaml(BENCHMARK_ROOT / "replay.yaml", payload)
    return payload


def readable_yaml(path: Path) -> bool:
    """Return whether one YAML file can be parsed."""
    # Parse the YAML file.
    try:
        read_yaml(Path(path))
        return True
    except Exception:
        return False


def readable_csv(path: Path) -> bool:
    """Return whether one CSV file can be read."""
    # Read a small CSV preview.
    try:
        pd.read_csv(Path(path), nrows=5)
        return True
    except Exception:
        return False


def run_baseline_evaluation() -> Path:
    """Run the complete baseline evaluation build."""
    # Create the target directories.
    (BENCHMARK_ROOT / "evaluation").mkdir(parents=True, exist_ok=True)
    (BENCHMARK_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    REPO_BENCHMARK_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Build contract and metrics artifacts.
    build_evaluation_input_manifest()
    join_validation = run_join_validation()
    signal = standardize_signal_metrics()
    liquidity = standardize_liquidity_metrics()
    time_bucket = build_time_bucket_metrics()
    build_extreme_value_metrics()
    build_normalization_metrics()
    copy_training_post_eval_artifacts()
    build_training_diagnostics()
    build_bootstrap_confidence_intervals()
    build_stability_diagnostics()
    build_label_availability_coverage()
    build_turnover_and_capacity_diagnostics()
    build_short_side_diagnostics()
    trading = standardize_trading_rule_metrics()
    summary = build_evaluation_summary(join_validation, signal, liquidity, time_bucket, trading)
    comparison = build_comparison(summary)

    # Render reports and update replay.
    write_train_monitoring_html()
    write_model_signal_evaluation_html(summary)
    write_trading_evaluation_html()
    card = write_evaluation_card_html(summary, comparison)
    write_full_benchmark_report(summary)
    replay = update_benchmark_and_replay(summary)
    if replay["status"] != "passed":
        raise RuntimeError(f"replay failed: {replay}")
    return card


if __name__ == "__main__":
    print(run_baseline_evaluation().as_posix())
