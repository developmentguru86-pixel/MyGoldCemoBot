"""Performance statistics. No scipy dependency."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, |rel err| < 1.2e-9)."""
    if p <= 0.0 or p >= 1.0:
        return float("nan")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def deflated_sharpe(r: pd.Series, trial_sharpes_bar: list[float]) -> float | None:
    """Bailey & López de Prado (2014): PSR against the Sharpe one expects from the best of N trials
    by luck alone, given the variance of the trial Sharpes. Per-bar units throughout."""
    trials = [x for x in trial_sharpes_bar if x == x]
    n_tr = len(trials)
    if n_tr < 2:
        return probabilistic_sharpe(r, 0.0)
    v = float(np.var(trials))
    if v <= 0:
        return probabilistic_sharpe(r, 0.0)
    gamma = 0.5772156649015329
    sr_star = math.sqrt(v) * ((1 - gamma) * norm_ppf(1 - 1.0 / n_tr) + gamma * norm_ppf(1 - 1.0 / (n_tr * math.e)))
    return probabilistic_sharpe(r, sr_star)


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
    if e0 <= 0:
        raise ValueError("summarize: starting equity must be > 0")
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
