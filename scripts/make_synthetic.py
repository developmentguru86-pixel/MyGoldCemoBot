"""SYNTHETIC XAU/USD H4 bars for pipeline smoke tests ONLY.
GARCH(1,1) volatility + 2-state Markov drift. Results on this data say NOTHING about
real performance. Use scripts/fetch_history.py for real bars."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot.data import save_csv  # noqa: E402


def make(years: float, seed: int, bars_per_day: int = 6, p0: float = 3500.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-01-01", periods=int(years * 260), tz="UTC")
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=4 * k) for d in days for k in range(bars_per_day)])
    n = len(idx)
    bpy = bars_per_day * 260
    # GARCH(1,1) on bar returns, unconditional annual vol ~15%
    omega, alpha, beta = (0.15 ** 2 / bpy) * 0.05, 0.08, 0.87
    sig2 = np.empty(n); eps = np.empty(n)
    sig2[0] = 0.15 ** 2 / bpy
    # regime drift: annualised +25% / -15%, persistence 0.995
    mu = np.empty(n); state = 0
    drift = np.array([0.25, -0.15]) / bpy
    for t in range(n):
        if rng.random() > 0.995:
            state = 1 - state
        mu[t] = drift[state]
        if t > 0:
            sig2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * rng.standard_t(df=5) / np.sqrt(5 / 3)
    r = mu + eps
    close = p0 * np.cumprod(1 + r)
    open_ = np.r_[p0, close[:-1]]
    wick = np.abs(rng.normal(0, 1, n)) * np.sqrt(sig2) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    hour = idx.hour
    spread = 0.25 + 0.05 * rng.random(n) + np.where((hour == 20) | (hour == 0), 0.6, 0.0)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "tick_volume": rng.integers(500, 5000, n), "spread": spread}, index=idx)
    df.index.name = "time"
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/SYNTHETIC_H4.csv")
    a = ap.parse_args()
    df = make(a.years, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, a.out)
    print(f"wrote {len(df)} SYNTHETIC bars to {a.out} ({df.index[0].date()} .. {df.index[-1].date()})")
