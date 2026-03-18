"""Compute multi-dimensional IC diagnostics from qmodel evaluator feather outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class EvalConfig:
    """Define evaluation IO and rolling-group knobs."""

    stock1m_dir: Path
    window_size: int
    step_size: int
    horizon_minutes: int


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation for two 1D arrays."""
    # Keep only finite rows and require at least two samples.
    m = np.isfinite(x) & np.isfinite(y)
    x2 = x[m].astype(float, copy=False)
    y2 = y[m].astype(float, copy=False)
    if int(x2.shape[0]) < 2:
        return float("nan")
    return float(np.corrcoef(x2, y2)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation via rank Pearson."""
    # Keep only finite rows and require at least two samples.
    m = np.isfinite(x) & np.isfinite(y)
    x2 = x[m].astype(float, copy=False)
    y2 = y[m].astype(float, copy=False)
    if int(x2.shape[0]) < 2:
        return float("nan")
    xr = pd.Series(x2).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y2).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(xr, yr)[0, 1])


def load_eval_predictions(shard_path: Path) -> pd.DataFrame:
    """Load a rank0 feather shard written by qmodel evaluator."""
    # Read the feather file and validate required columns.
    df = pd.read_feather(Path(shard_path))
    need = {"prediction", "target", "StockCode", "DateTime", "date", "time"}
    if not need.issubset(set(df.columns)):
        raise RuntimeError(f"Missing columns in shard: {sorted(need - set(df.columns))}")
    return df


def pooled_ic(df: pd.DataFrame) -> dict[str, float]:
    """Compute pooled IC across all (stock,time) samples."""
    # Compute Pearson and Spearman across the full dataframe.
    pred = df["prediction"].to_numpy(dtype=float)
    tgt = df["target"].to_numpy(dtype=float)
    return {"pearson_ic": _pearson(pred, tgt), "rank_ic": _spearman(pred, tgt), "count": int(np.isfinite(pred).sum())}


def intraday_time_series_ic(df: pd.DataFrame, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute intraday minute-of-day aggregated cross-sectional IC curve."""
    # Compute per-minute cross-sectional IC for each day.
    tmp = df[["date", "time", "prediction", "target"]].copy()
    rows: list[dict[str, object]] = []
    for (d, t), g in tmp.groupby(["date", "time"], sort=True):
        # Compute correlations on the cross-section at this timestamp.
        p = g["prediction"].to_numpy(dtype=float)
        y = g["target"].to_numpy(dtype=float)
        rows.append({"date": int(d), "time": int(t), "ic": _pearson(p, y), "rank_ic": _spearman(p, y), "n": int(np.isfinite(p).sum())})
    cs = pd.DataFrame(rows)

    # Aggregate across dates by minute-of-day.
    agg = cs.groupby("time", sort=True).agg(
        mean_ic=("ic", "mean"),
        std_ic=("ic", "std"),
        mean_rank_ic=("rank_ic", "mean"),
        std_rank_ic=("rank_ic", "std"),
        count=("ic", "count"),
    )
    agg = agg.reset_index().sort_values("time", kind="stable").reset_index(drop=True)
    agg.to_csv(out_csv, index=False)

    # Plot the intraday mean IC curve on an HH:MM axis.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    xs = agg["time"].to_numpy(dtype=int)
    labels = [f"{int(t)//10000:02d}:{(int(t)%10000)//100:02d}" for t in xs]
    ax.plot(np.arange(len(labels)), agg["mean_ic"].to_numpy(dtype=float), label="Pearson IC", linewidth=1.8)
    ax.plot(np.arange(len(labels)), agg["mean_rank_ic"].to_numpy(dtype=float), label="Rank IC (Spearman)", linewidth=1.8)
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title("Intraday IC curve (mean across dates)")
    ax.set_xlabel("time (minute bars; lunch break absent)")
    ax.set_ylabel("mean IC")
    tick_pos = np.linspace(0, max(len(labels) - 1, 1), 10).round().astype(int)
    ax.set_xticks(tick_pos, [labels[i] for i in tick_pos], rotation=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return agg


def _load_price_panel_for_dates(config: EvalConfig, dates: list[int]) -> pd.DataFrame:
    """Load Close series for the requested trade dates from stock1m."""
    # Read Close and DateTime for each date and concatenate into one table.
    parts: list[pd.DataFrame] = []
    for d in list(dates):
        # Resolve file path by year folder convention.
        year = int(d) // 10000
        path = Path(config.stock1m_dir) / str(year) / f"{int(d)}.feather"
        day = pd.read_feather(path, columns=["StockCode", "DateTime", "Close", "Date"])
        day = day.sort_values(["StockCode", "DateTime"], kind="stable").reset_index(drop=True)
        parts.append(day)
    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


def _forward_vol_label(day: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    """Compute per-row forward volatility label using next-horizon 1m returns std."""
    # Build log close and 1m returns per stock with NaN for invalid prices.
    close = day["Close"].to_numpy(dtype=float)
    m = np.isfinite(close) & (close > 0.0)
    log_close = np.full_like(close, np.nan, dtype=float)
    log_close[m] = np.log(close[m])
    day = day.copy()
    day["log_close"] = log_close
    day["r1"] = day.groupby("StockCode", sort=False)["log_close"].diff(1)

    # Compute forward std of r1[t+1:t+h] using groupby+rolling vectorization.
    h = int(horizon_minutes)
    g = day.groupby("StockCode", sort=False)["r1"]
    r_next = g.shift(-1)
    vol = (
        r_next.groupby(day["StockCode"], sort=False)
        .rolling(window=h, min_periods=2)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
        .shift(-(h - 1))
    )
    return vol.astype(np.float32, copy=False).rename("volatility_label")


def attach_labels(pred_df: pd.DataFrame, config: EvalConfig) -> pd.DataFrame:
    """Attach volatility_label and price_label to the prediction dataframe."""
    # Resolve trade_date list from pred_df by converting yymmdd to yyyymmdd.
    yymmdd = pred_df["date"].astype(int).unique().tolist()
    yyyymmdd = [20000000 + int(d) for d in yymmdd]

    # Load price panels and compute volatility labels per date.
    panel = _load_price_panel_for_dates(config, yyyymmdd)
    panel["trade_date"] = panel["Date"].astype(int)
    panel["price_label"] = panel["Close"].astype(float)
    panel["volatility_label"] = _forward_vol_label(panel, int(config.horizon_minutes)).astype(float)

    # Merge labels onto prediction rows using (StockCode, DateTime) keys.
    key_cols = ["StockCode", "DateTime"]
    merged = pred_df.merge(panel[key_cols + ["price_label", "volatility_label"]], on=key_cols, how="left", validate="many_to_one")
    return merged


def _rolling_group_ic(
    df: pd.DataFrame,
    label_col: str,
    window_size: int,
    step_size: int,
) -> pd.DataFrame:
    """Compute rolling-window IC over cross-sections sorted by a label."""
    # Compute per-timestamp grouped IC rows.
    rows: list[dict[str, object]] = []
    for (d, t), g in df.groupby(["date", "time"], sort=True):
        # Sort by the grouping label and drop missing rows.
        gg = g[["prediction", "target", label_col]].dropna(subset=["prediction", "target", label_col]).sort_values(label_col, kind="stable")
        n = int(gg.shape[0])
        if n < int(window_size):
            continue

        # Slide a rolling window along sorted rows and compute IC per window.
        pred = gg["prediction"].to_numpy(dtype=float)
        tgt = gg["target"].to_numpy(dtype=float)
        for st in range(0, n - int(window_size) + 1, int(step_size)):
            # Define window center rank as percentile for cross-date comparability.
            center = float(st + int(window_size) * 0.5)
            center_rank = float(center / float(n))
            p = pred[st : st + int(window_size)]
            y = tgt[st : st + int(window_size)]
            rows.append(
                {
                    "date": int(d),
                    "time": int(t),
                    "group_center_rank": float(center_rank),
                    "ic": _pearson(p, y),
                    "rank_ic": _spearman(p, y),
                    "n": int(window_size),
                }
            )
    out = pd.DataFrame(rows)

    # Aggregate across all timestamps by binned center rank.
    if out.shape[0] == 0:
        return out
    out["rank_bin"] = out["group_center_rank"].round(3)
    agg = out.groupby("rank_bin", sort=True).agg(
        group_center_rank=("group_center_rank", "mean"),
        mean_ic=("ic", "mean"),
        std_ic=("ic", "std"),
        mean_rank_ic=("rank_ic", "mean"),
        std_rank_ic=("rank_ic", "std"),
        count=("ic", "count"),
    )
    return agg.reset_index(drop=True).sort_values("group_center_rank", kind="stable").reset_index(drop=True)


def _plot_group_curve(df: pd.DataFrame, title: str, out_png: Path) -> None:
    """Plot mean IC curves against group_center_rank."""
    # Render a simple 2-line plot for Pearson and Rank IC.
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df["group_center_rank"].to_numpy(dtype=float), df["mean_ic"].to_numpy(dtype=float), label="Pearson IC", linewidth=1.8)
    ax.plot(df["group_center_rank"].to_numpy(dtype=float), df["mean_rank_ic"].to_numpy(dtype=float), label="Rank IC", linewidth=1.8)
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("group_center_rank (percentile)")
    ax.set_ylabel("mean IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _empty_group_schema() -> pd.DataFrame:
    """Return an empty rolling-group IC dataframe with a stable schema."""
    # Define a stable column order so downstream report rendering is predictable.
    cols = ["group_center_rank", "mean_ic", "std_ic", "mean_rank_ic", "std_rank_ic", "count"]
    out = pd.DataFrame({c: pd.Series([], dtype=float) for c in cols})
    out["count"] = out["count"].astype(int)
    return out


def _plot_empty_group_curve(title: str, out_png: Path) -> None:
    """Write a placeholder plot when rolling groups are empty."""
    # Render a simple figure with an explanatory text to avoid downstream missing files.
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.text(0.5, 0.5, "Empty rolling groups: n < window_size", ha="center", va="center", fontsize=12)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def volatility_rolling_ic(pred_df: pd.DataFrame, config: EvalConfig, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute volatility rolling-window IC and persist CSV/plot."""
    # Compute rolling-window IC assuming labels are already attached.
    agg = _rolling_group_ic(pred_df, "volatility_label", int(config.window_size), int(config.step_size))
    if agg.shape[0] == 0:
        warnings.warn("Empty volatility rolling IC: valid stock count < window_size.", RuntimeWarning)
        agg = _empty_group_schema()
        agg.to_csv(out_csv, index=False)
        _plot_empty_group_curve("Volatility rolling IC", out_png)
        return agg

    # Persist the aggregated curve and emit the plot.
    agg.to_csv(out_csv, index=False)
    _plot_group_curve(agg, "Volatility rolling IC", out_png)
    return agg


def price_rolling_ic(pred_df: pd.DataFrame, config: EvalConfig, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute price rolling-window IC and persist CSV/plot."""
    # Compute rolling-window IC assuming labels are already attached.
    agg = _rolling_group_ic(pred_df, "price_label", int(config.window_size), int(config.step_size))
    if agg.shape[0] == 0:
        warnings.warn("Empty price rolling IC: valid stock count < window_size.", RuntimeWarning)
        agg = _empty_group_schema()
        agg.to_csv(out_csv, index=False)
        _plot_empty_group_curve("Price rolling IC", out_png)
        return agg

    # Persist the aggregated curve and emit the plot.
    agg.to_csv(out_csv, index=False)
    _plot_group_curve(agg, "Price rolling IC", out_png)
    return agg


def score_ret_rank_plot(pred_df: pd.DataFrame, out_png: Path) -> pd.DataFrame:
    """Plot predicted-score rank bins against realized target return and win-rate."""
    # Build rank-percentile bins by prediction within each timestamp cross-section.
    tmp = pred_df[["date", "time", "prediction", "target"]].dropna(subset=["prediction", "target"]).copy()
    tmp["pred_rank_pct"] = tmp.groupby(["date", "time"], sort=False)["prediction"].rank(method="average", pct=True)

    # Bin rank into deciles and aggregate realized return and sign win rate.
    tmp["decile"] = np.minimum((tmp["pred_rank_pct"] * 10.0).astype(int), 9)
    agg = tmp.groupby("decile", sort=True).agg(
        mean_target=("target", "mean"),
        win_rate=("target", lambda x: float((np.asarray(x, dtype=float) > 0.0).mean())),
        count=("target", "size"),
    )
    agg = agg.reset_index()

    # Plot mean return and win-rate on dual axes.
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    xs = agg["decile"].to_numpy(dtype=int)
    ax1.plot(xs, agg["mean_target"].to_numpy(dtype=float), color="#4c72b0", linewidth=2.0, label="mean target")
    ax2.plot(xs, agg["win_rate"].to_numpy(dtype=float), color="#dd8452", linewidth=2.0, label="win rate")
    ax1.set_xlabel("prediction rank decile (0=low, 9=high)")
    ax1.set_ylabel("mean target")
    ax2.set_ylabel("win rate (target>0)")
    ax1.set_title("Prediction vs target: rank curve")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return agg
