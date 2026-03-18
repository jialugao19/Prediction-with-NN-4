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

    # Merge per-day labels in a loop to keep peak memory bounded on multi-year spans.
    parts: list[pd.DataFrame] = []
    for d_yymmdd, d_yyyymmdd in zip(yymmdd, yyyymmdd):
        # Select prediction rows for one trade date to minimize merge payload.
        day_pred = pred_df.loc[pred_df["date"].astype(int) == int(d_yymmdd)].copy()

        # Load price panel for this date and compute same-day labels.
        panel = _load_price_panel_for_dates(config, [int(d_yyyymmdd)])
        panel["price_label"] = panel["Close"].astype(float)
        panel["volatility_label"] = _forward_vol_label(panel, int(config.horizon_minutes)).astype(float)

        # Merge labels onto prediction rows using (StockCode, DateTime) keys.
        key_cols = ["StockCode", "DateTime"]
        day_merged = day_pred.merge(panel[key_cols + ["price_label", "volatility_label"]], on=key_cols, how="left", validate="many_to_one")
        parts.append(day_merged)

    # Concatenate day merges back into one dataframe in stable original order.
    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


def annual_pooled_ic(df: pd.DataFrame, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute pooled IC per calendar year and persist a CSV plus bar plot."""
    # Compute year integer from yymmdd date and keep only finite prediction/target rows.
    tmp = df[["date", "prediction", "target"]].dropna(subset=["prediction", "target"]).copy()
    tmp["year"] = (2000 + (tmp["date"].astype(int) // 10000)).astype(int)

    # Aggregate pooled Pearson/Spearman IC per year.
    rows: list[dict[str, object]] = []
    for y, g in tmp.groupby("year", sort=True):
        # Compute correlations using the shared correlation helpers.
        pred = g["prediction"].to_numpy(dtype=float)
        tgt = g["target"].to_numpy(dtype=float)
        rows.append({"year": int(y), "pearson_ic": _pearson(pred, tgt), "rank_ic": _spearman(pred, tgt), "count": int(np.isfinite(pred).sum())})
    out = pd.DataFrame(rows).sort_values("year", kind="stable").reset_index(drop=True)
    out.to_csv(out_csv, index=False)

    # Plot yearly IC bars for Pearson and Rank IC.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    xs = out["year"].to_numpy(dtype=int)
    ax.bar(xs - 0.15, out["pearson_ic"].to_numpy(dtype=float), width=0.3, label="Pearson IC")
    ax.bar(xs + 0.15, out["rank_ic"].to_numpy(dtype=float), width=0.3, label="Rank IC (Spearman)")
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_title("Annual pooled IC (prediction vs target)")
    ax.set_xlabel("year")
    ax.set_ylabel("IC")
    ax.set_xticks(xs, [str(int(x)) for x in xs])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out


def _rolling_group_ic(
    df: pd.DataFrame,
    label_col: str,
    window_size: int,
    step_size: int,
) -> pd.DataFrame:
    """Compute rolling-window IC over cross-sections sorted by a label."""
    # Define helpers to compute Pearson correlation from windowed prefix sums.
    def _corr_from_sums(sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float, w: int) -> float:
        """Compute Pearson correlation from raw sums on a fixed window."""
        # Compute covariance and variances with stable float math.
        wf = float(w)
        cov = float(sum_xy - (sum_x * sum_y) / wf)
        var_x = float(sum_x2 - (sum_x * sum_x) / wf)
        var_y = float(sum_y2 - (sum_y * sum_y) / wf)
        if not (np.isfinite(cov) and np.isfinite(var_x) and np.isfinite(var_y)):
            return float("nan")
        if float(var_x) <= 0.0 or float(var_y) <= 0.0:
            return float("nan")
        return float(cov / float(np.sqrt(var_x * var_y)))

    # Accumulate aggregated moments per rank bin to avoid materializing millions of window rows.
    bins: dict[float, dict[str, float]] = {}
    w = int(window_size)
    step = int(step_size)
    for (_d, _t), g in df.groupby(["date", "time"], sort=True):
        # Sort by the grouping label and drop missing rows.
        gg = g[["prediction", "target", label_col]].dropna(subset=["prediction", "target", label_col]).sort_values(label_col, kind="stable")
        n = int(gg.shape[0])
        if int(n) < int(w):
            continue

        # Extract prediction/target arrays and cast once for stable prefix-sum accumulation.
        pred = gg["prediction"].to_numpy(dtype=np.float64, copy=False)
        tgt = gg["target"].to_numpy(dtype=np.float64, copy=False)

        # Build prefix sums for Pearson IC computation on raw values.
        ps = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pred, dtype=np.float64)])
        ts = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tgt, dtype=np.float64)])
        p2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pred * pred, dtype=np.float64)])
        t2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tgt * tgt, dtype=np.float64)])
        pts = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pred * tgt, dtype=np.float64)])

        # Build global ranks once and reuse for the rolling-window rank IC approximation.
        pr = stats.rankdata(pred, method="average").astype(np.float64, copy=False)
        tr = stats.rankdata(tgt, method="average").astype(np.float64, copy=False)
        prs = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pr, dtype=np.float64)])
        trs = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tr, dtype=np.float64)])
        pr2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pr * pr, dtype=np.float64)])
        tr2s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(tr * tr, dtype=np.float64)])
        prts = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(pr * tr, dtype=np.float64)])

        # Slide windows along sorted rows and accumulate moments per rank bin.
        for st in range(0, int(n - w + 1), int(step)):
            # Compute the window's center rank percentile for cross-date binning.
            center = float(st + w * 0.5)
            center_rank = float(center / float(n))
            rank_bin = float(round(center_rank, 3))

            # Compute Pearson IC from prefix sums on raw prediction/target.
            ed = int(st + w)
            sum_p = float(ps[ed] - ps[st])
            sum_t = float(ts[ed] - ts[st])
            sum_p2 = float(p2s[ed] - p2s[st])
            sum_t2 = float(t2s[ed] - t2s[st])
            sum_pt = float(pts[ed] - pts[st])
            ic = _corr_from_sums(sum_p, sum_t, sum_p2, sum_t2, sum_pt, w)

            # Compute rank IC from prefix sums on global ranks restricted to the same window.
            sum_pr = float(prs[ed] - prs[st])
            sum_tr = float(trs[ed] - trs[st])
            sum_pr2 = float(pr2s[ed] - pr2s[st])
            sum_tr2 = float(tr2s[ed] - tr2s[st])
            sum_prt = float(prts[ed] - prts[st])
            rank_ic = _corr_from_sums(sum_pr, sum_tr, sum_pr2, sum_tr2, sum_prt, w)

            # Initialize accumulator buckets lazily per observed rank bin.
            if rank_bin not in bins:
                bins[rank_bin] = {
                    "sum_center_rank": 0.0,
                    "sum_ic": 0.0,
                    "sum_ic2": 0.0,
                    "n_ic": 0.0,
                    "sum_rank_ic": 0.0,
                    "sum_rank_ic2": 0.0,
                    "n_rank_ic": 0.0,
                    "count": 0.0,
                }
            acc = bins[rank_bin]

            # Accumulate first and second moments for mean/std computation.
            acc["sum_center_rank"] += float(center_rank)
            if np.isfinite(ic):
                acc["sum_ic"] += float(ic)
                acc["sum_ic2"] += float(ic * ic)
                acc["n_ic"] += 1.0
            if np.isfinite(rank_ic):
                acc["sum_rank_ic"] += float(rank_ic)
                acc["sum_rank_ic2"] += float(rank_ic * rank_ic)
                acc["n_rank_ic"] += 1.0
            acc["count"] += 1.0

    # Return empty output early when no bins were accumulated.
    if len(bins) == 0:
        return pd.DataFrame([])

    # Convert aggregated moments into the stable curve dataframe schema.
    rows: list[dict[str, object]] = []
    for rank_bin in sorted(bins.keys()):
        # Convert sums into mean/std while guarding against negative variance from float drift.
        acc = bins[rank_bin]
        c = float(acc["count"])
        mean_center = float(acc["sum_center_rank"] / c)
        # Compute mean/std for IC metrics using finite-only counts to match pandas semantics.
        n_ic = float(acc["n_ic"])
        n_rank_ic = float(acc["n_rank_ic"])
        mean_ic = float(acc["sum_ic"] / n_ic) if n_ic > 0.0 else float("nan")
        var_ic = float(acc["sum_ic2"] / n_ic - mean_ic * mean_ic) if n_ic > 1.0 else float("nan")
        mean_rank_ic = float(acc["sum_rank_ic"] / n_rank_ic) if n_rank_ic > 0.0 else float("nan")
        var_rank_ic = float(acc["sum_rank_ic2"] / n_rank_ic - mean_rank_ic * mean_rank_ic) if n_rank_ic > 1.0 else float("nan")
        rows.append(
            {
                "group_center_rank": float(mean_center),
                "mean_ic": float(mean_ic),
                "std_ic": float(np.sqrt(max(var_ic, 0.0))) if np.isfinite(var_ic) else float("nan"),
                "mean_rank_ic": float(mean_rank_ic),
                "std_rank_ic": float(np.sqrt(max(var_rank_ic, 0.0))) if np.isfinite(var_rank_ic) else float("nan"),
                "count": int(c),
            }
        )
    out = pd.DataFrame(rows).sort_values("group_center_rank", kind="stable").reset_index(drop=True)
    return out


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
