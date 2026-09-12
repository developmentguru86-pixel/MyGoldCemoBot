"""Time-series momentum ensemble with volatility targeting and a fractional-Kelly throttle.

Every feature at bar t uses information up to and including bar t's close.
exposure[t] is the position (notional / equity) to hold over bar t+1.
No look-ahead by construction (tests/test_strategy.py asserts this).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def compute_exposure(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    s = cfg.strategy
    close = df["close"]
    ret = close.pct_change()

    # realised vol, EWMA, per bar and annualised (floored to avoid blow-ups in dead markets)
    vol_bar = ret.ewm(span=s.vol_span, min_periods=s.vol_span).std()
    vol_ann = (vol_bar * np.sqrt(cfg.bars_per_year)).clip(lower=s.vol_floor)

    # momentum votes: sign of vol-normalised L-bar return, averaged across lookbacks -> [-1, 1]
    votes = []
    for L in s.lookbacks:
        mom = close / close.shift(L) - 1.0
        z = mom / (vol_bar * np.sqrt(L))
        votes.append(np.sign(z))
    signal = pd.concat(votes, axis=1).mean(axis=1)

    exposure_raw = (signal * s.target_vol / vol_ann).clip(-s.max_leverage, s.max_leverage)

    # fractional Kelly on the un-throttled strategy's own realised returns:
    # f* = mu / sigma^2 is the optimal multiplier on a unit-exposure strategy.
    if s.kelly.enabled:
        unit_ret = exposure_raw.shift(1) * ret
        mp = max(2, s.kelly.window // 2)
        mu = unit_ret.rolling(s.kelly.window, min_periods=mp).mean()
        var = unit_ret.rolling(s.kelly.window, min_periods=mp).var()
        f_star = mu / var.replace(0.0, np.nan)
        kelly_mult = (s.kelly.fraction * f_star).clip(lower=s.kelly.min_mult, upper=1.0).fillna(1.0)
    else:
        kelly_mult = pd.Series(1.0, index=df.index)

    exposure = (exposure_raw * kelly_mult).fillna(0.0)

    return pd.DataFrame({
        "close": close,
        "ret": ret,
        "vol_ann": vol_ann,
        "signal": signal,
        "exposure_raw": exposure_raw,
        "kelly_mult": kelly_mult,
        "exposure": exposure,
    })


def bars_needed(cfg: Config) -> int:
    """Warm-up length the live loop must fetch so the last exposure value is fully formed."""
    s = cfg.strategy
    n = max(s.lookbacks) + 3 * s.vol_span + 20
    if s.kelly.enabled:
        n += s.kelly.window
    return n
