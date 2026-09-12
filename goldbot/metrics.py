"""Performance statistics. No scipy dependency."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def probabilistic_sharpe(r: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Bailey & López de Prado PSR: P(true per-bar SR > benchmark) given skew/kurtosis and sample size."""
    r = r.dropna()
    n = len(r)
    if n < 30 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    skew, kurt = r.skew(), r.kurt() + 3.0
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return float("nan")
    return norm_cdf((sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom))


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def summarize(equity: pd.Series, bars_per_year: int, bars_per_day: int,
              trades: int = 0, total_cost: float = 0.0, total_swap: float = 0.0,
              lots: pd.Series | None = None) -> dict:
    r = equity.pct_change().dropna()
    n = len(r)
    years = n / bars_per_year if bars_per_year else float("nan")
    e0, e1 = float(equity.iloc[0]), float(equity.iloc[-1])
    total_ret = e1 / e0 - 1.0
    cagr = (e1 / e0) ** (1.0 / years) - 1.0 if years > 0 and e1 > 0 else -1.0
    std = r.std()
    vol = std * math.sqrt(bars_per_year) if std > 0 else 0.0
    sharpe = r.mean() / std * math.sqrt(bars_per_year) if std > 0 else 0.0
    dstd = r[r < 0].std()
    sortino = r.mean() / dstd * math.sqrt(bars_per_year) if dstd and dstd > 0 else float("nan")
    dd = drawdown(equity)
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")
    trading_days = n / bars_per_day if bars_per_day else float("nan")
    out = {
        "bars": int(n),
        "years": round(years, 2),
        "start_equity": round(e0, 2),
        "end_equity": round(e1, 2),
        "total_return": round(total_ret, 4),
        "cagr": round(cagr, 4),
        "ann_vol": round(vol, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3) if not math.isnan(sortino) else None,
        "max_drawdown": round(mdd, 4),
        "calmar": round(calmar, 3) if not math.isnan(calmar) else None,
        "psr_gt_0": round(probabilistic_sharpe(r), 3) if n >= 30 else None,
        "trades": int(trades),
        "total_cost": round(total_cost, 2),
        "total_swap": round(total_swap, 2),
        "pnl_per_trading_day": round((e1 - e0) / trading_days, 2) if trading_days else None,
        "worst_bar": round(float(r.min()), 4) if n else None,
        "best_bar": round(float(r.max()), 4) if n else None,
    }
    if lots is not None and len(lots):
        out["time_in_market"] = round(float((lots != 0).mean()), 3)
        out["avg_abs_lots"] = round(float(lots.abs()[lots != 0].mean()), 4) if (lots != 0).any() else 0.0
    return out


def block_bootstrap(r: np.ndarray, horizon: int, sims: int = 2000, block: int = 20, seed: int = 0) -> dict:
    """Stationary-ish block bootstrap of bar returns -> distribution of horizon return and max drawdown."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) <= block + 1:
        return {}
    rng = np.random.default_rng(seed)
    nblocks = int(math.ceil(horizon / block))
    starts = rng.integers(0, len(r) - block, size=(sims, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(sims, -1)[:, :horizon]
    paths = np.cumprod(1.0 + r[idx], axis=1)
    ret = paths[:, -1] - 1.0
    mdd = (paths / np.maximum.accumulate(paths, axis=1) - 1.0).min(axis=1)
    q = lambda a, p: round(float(np.percentile(a, p)), 4)
    return {
        "horizon_bars": horizon, "sims": sims, "block": block,
        "return_p05": q(ret, 5), "return_p50": q(ret, 50), "return_p95": q(ret, 95),
        "maxdd_p05": q(mdd, 5), "maxdd_p50": q(mdd, 50), "maxdd_p95": q(mdd, 95),
        "prob_loss": round(float((ret < 0).mean()), 3),
        "prob_dd_gt_20pct": round(float((mdd < -0.20).mean()), 3),
    }
