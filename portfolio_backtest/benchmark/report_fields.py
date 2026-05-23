"""Field definitions and report value formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
        "val_mse_raw": {"display_name": "val_mse_raw", "unit": "raw-label MSE", "direction": "lower is better", "explanation": "raw label space 下的 validation MSE。"},
        "val_mse_normalized": {"display_name": "val_mse_normalized", "unit": "normalized-label MSE", "direction": "lower is better", "explanation": "转换到 normalized label space 后的 validation MSE, 用于和 train loss 比较。"},
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

    # Keep booleans as true/false instead of numeric floats.
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))

    # Keep non-numeric values readable and shorten absolute checkpoint paths.
    if not isinstance(value, (int, float, np.integer, np.floating)):
        text = str(value)
        if text.startswith("/data-cache/"):
            return Path(text).name
        return text

    # Format integer-like counts without decimal noise.
    name = str(metric).lower()
    number = float(value)
    if "count" in name or "rows" in name or "iter" in name or name in {"step", "time", "time_bucket", "last_step", "prev_iter", "selected_checkpoint"}:
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
