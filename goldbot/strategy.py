"""Time-series momentum ensemble with volatility targeting, entry hysteresis, regime filter
and a cost-aware fractional-Kelly throttle.

Every feature at bar t uses information up to and including bar t's close.
exposure[t] is the position (notional / equity) to hold over bar t+1.
No look-ahead by construction (tests assert this).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _hysteresis(signal: np.ndarray, entry_min: float, exit_min: float, min_hold: int) -> np.ndarray:
    """Turn a continuous signal into a held direction with entry/exit thresholds and a minimum hold."""
    n = len(signal)
    held = np.zeros(n)
    pos, last_entry = 0.0, -10**9
    for t in range(n):
        s = signal[t]
        if np.isnan(s):
            held[t] = 0.0
            continue
        if pos == 0.0:
            if abs(s) >= max(entry_min, 1e-12) and (t - last_entry) >= min_hold:
                pos, last_entry = np.sign(s), t
        else:
            # keep while the signal still points the same way and stays above the exit threshold
            same_side = np.sign(s) == pos
            if not same_side and (t - last_entry) >= min_hold:
                if abs(s) >= max(entry_min, 1e-12):   # flip straight into the other side
                    pos, last_entry = np.sign(s), t
                else:
                    pos = 0.0
            elif same_side and abs(s) < exit_min:
                pos = 0.0
        held[t] = pos
    return held


def compute_exposure(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    s = cfg.strategy
    close = df["close"]
    ret = close.pct_change()

    vol_bar = ret.ewm(span=s.vol_span, min_periods=s.vol_span).std()
    vol_ann = (vol_bar * np.sqrt(cfg.bars_per_year)).clip(lower=s.vol_floor)

    # momentum votes (sign of vol-normalised L-bar return) and their average strength
    votes, zs = [], []
    for L in s.lookbacks:
        mom = close / close.shift(L) - 1.0
        z = mom / (vol_bar * np.sqrt(L))
        votes.append(np.sign(z))
        zs.append(z.abs())
    signal = pd.concat(votes, axis=1).mean(axis=1)          # in [-1, 1]
    z_strength = pd.concat(zs, axis=1).mean(axis=1)

    # --- direction with hysteresis / minimum hold; strength gate
    if s.entry_min_signal > 0 or s.exit_min_signal > 0 or s.min_hold_bars > 0:
        gated = signal.where(z_strength >= s.z_min) if s.z_min > 0 else signal
        direction = pd.Series(_hysteresis(gated.to_numpy(dtype=float), s.entry_min_signal, s.exit_min_signal,
                                          s.min_hold_bars), index=df.index)
        base = direction * np.where(direction != 0, np.maximum(signal.abs(), 1.0 / len(s.lookbacks)), 0.0)
        base = pd.Series(base, index=df.index)
    else:
        base = signal.where(z_strength >= s.z_min, 0.0) if s.z_min > 0 else signal

    # --- regime filter
    regime_scale = pd.Series(1.0, index=df.index)
    r = s.regime
    if r.er_window > 0:
        net = (close - close.shift(r.er_window)).abs()
        path = close.diff().abs().rolling(r.er_window).sum()
        er = (net / path.replace(0.0, np.nan)).fillna(0.0)
        regime_scale = regime_scale.where(er >= r.er_min, 0.0)
    if r.vol_pct_window > 0 and r.vol_pct_max < 1.0:
        pct = vol_ann.rolling(r.vol_pct_window, min_periods=r.vol_pct_window // 2).rank(pct=True)
        regime_scale = regime_scale.where(~(pct > r.vol_pct_max), regime_scale * r.high_vol_scale)

    # --- confidence tiers: size by the strength of the vol-normalised momentum, not just its sign
    conf_scale = pd.Series(1.0, index=df.index)
    if s.confidence and s.confidence_tiers:
        tiers = sorted(s.confidence_tiers)
        mult = np.zeros(len(df))
        zs_np = z_strength.to_numpy(dtype=float)
        for thr, m in tiers:
            mult = np.where(zs_np >= thr, m, mult)
        conf_scale = pd.Series(mult, index=df.index)

    exposure_raw = (base * regime_scale * conf_scale * s.target_vol / vol_ann).clip(-s.max_leverage, s.max_leverage)
    if s.direction == "long":
        exposure_raw = exposure_raw.clip(lower=0.0)
    elif s.direction == "short":
        exposure_raw = exposure_raw.clip(upper=0.0)

    # --- fractional Kelly on the un-throttled strategy's own returns, net of its cost drag
    if s.kelly.enabled:
        unit_ret = exposure_raw.shift(1) * ret
        mp = max(2, s.kelly.window // 2)
        mu = unit_ret.rolling(s.kelly.window, min_periods=mp).mean()
        var = unit_ret.rolling(s.kelly.window, min_periods=mp).var()
        if s.kelly.cost_aware:
            c = cfg.costs
            cost_side = (c.spread / 2.0 + (close * c.slippage_pct if c.slippage_pct > 0 else c.slippage)) / close + c.fee_pct
            turnover = exposure_raw.diff().abs()
            drag = (turnover * cost_side).rolling(s.kelly.window, min_periods=mp).mean() * s.kelly.cost_margin
            mu = mu - drag
        f_star = mu / var.replace(0.0, np.nan)
        kelly_mult = (s.kelly.fraction * f_star).clip(lower=s.kelly.min_mult, upper=1.0).fillna(1.0)
    else:
        kelly_mult = pd.Series(1.0, index=df.index)

    exposure = (exposure_raw * kelly_mult).fillna(0.0)
    return pd.DataFrame({
        "close": close, "ret": ret, "vol_ann": vol_ann, "signal": signal, "z_strength": z_strength,
        "regime_scale": regime_scale, "exposure_raw": exposure_raw, "kelly_mult": kelly_mult, "exposure": exposure,
    })


def bars_needed(cfg: Config) -> int:
    """Warm-up length the live loop must fetch so the last exposure value is fully formed."""
    s = cfg.strategy
    warm = 3 * s.vol_span + 20
    n = max(s.lookbacks) + warm + (s.kelly.window if s.kelly.enabled else 0)
    # regime windows overlap the same warm-up, they are not additive
    n = max(n, warm + s.regime.vol_pct_window, s.regime.er_window + warm, s.min_hold_bars + warm)
    return n
