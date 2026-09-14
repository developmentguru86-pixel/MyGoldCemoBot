"""Layer 2 — precision entry engine.

Layer 1 (the existing quant model) decides WHAT and WHETHER (long/flat per asset).
Layer 2 decides WHEN: it waits for a specific price-action sequence at a higher-timeframe level.

Sequence for a long (mirrored for shorts):
  1. LEVEL    a pivot low that held for `right` bars on each side (higher-timeframe support)
  2. SWEEP    a bar trades BELOW the level (stop hunt) but closes back ABOVE it
  3. RECLAIM  price closes above the level within `reclaim_bars` (may be the sweep bar itself)
  4. BOS      price closes above the highest high since the sweep -> structure break, entry
  Stop: below the sweep low (plus a buffer). Risk per trade is a fixed % of equity, so the
  position size follows from the stop distance, not from a fixed notional.

Everything is evaluated bar by bar with no forward information: a setup uses only bars up to the
entry bar, and the outcome is resolved from later bars. Pivots are confirmed `right` bars after the
fact, so a level only becomes usable once it is confirmed (that delay is enforced).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EntryCfg:
    pivot_left: int = 10           # bars either side that define a swing pivot
    pivot_right: int = 5           # confirmation delay: a level is only usable after this many bars
    level_window: int = 250        # how far back to look for active levels
    level_tol: float = 0.003       # how close to the level the sweep must come (fraction of price)
    sweep_max_bars: int = 3        # sweep -> reclaim window
    bos_max_bars: int = 10         # reclaim -> structure break window
    stop_buffer: float = 0.002     # stop placed this far below the sweep low
    target_r: float = 2.0          # take profit at this multiple of risk
    time_stop_bars: int = 60       # give up if neither barrier is hit
    risk_per_trade: float = 0.025  # fraction of equity risked per trade
    max_concurrent: int = 3        # portfolio-level cap on simultaneous open trades
    min_stop_frac: float = 0.005   # ignore setups with an implausibly tight stop
    max_stop_frac: float = 0.15    # ... or an absurdly wide one


def pivot_lows(df: pd.DataFrame, left: int, right: int) -> np.ndarray:
    """Index positions of confirmed swing lows. Confirmation happens `right` bars later."""
    low = df["low"].to_numpy(dtype=float)
    n = len(low)
    out = []
    for i in range(left, n - right):
        w = low[i - left:i + right + 1]
        if low[i] == w.min() and (w == low[i]).sum() == 1:
            out.append(i)
    return np.array(out, dtype=int)


def find_setups(df: pd.DataFrame, cfg: EntryCfg, allowed: np.ndarray | None = None) -> list[dict]:
    """Scan for LEVEL -> SWEEP -> RECLAIM -> BOS sequences. `allowed[t]` gates entries by layer 1."""
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    n = len(c)
    piv = pivot_lows(df, cfg.pivot_left, cfg.pivot_right)
    # a pivot at index p is only KNOWN from p + right onwards
    known_from = {int(p): int(p) + cfg.pivot_right for p in piv}
    piv_sorted = sorted(known_from)
    setups: list[dict] = []
    t = cfg.pivot_left + cfg.pivot_right
    while t < n - 1:
        if allowed is not None and not allowed[t]:
            t += 1
            continue
        # active levels: confirmed, within the lookback window, and below current price
        levels = [p for p in piv_sorted if known_from[p] <= t and t - p <= cfg.level_window and l[p] < c[t]]
        if not levels:
            t += 1
            continue
        lvl_idx = max(levels, key=lambda p: l[p])    # nearest support below price
        level = l[lvl_idx]
        # 1) SWEEP: this bar dips below the level but closes back above it
        swept = l[t] < level * (1 - 0.0) and l[t] <= level * (1 + cfg.level_tol) and c[t] > level
        if not swept:
            t += 1
            continue
        sweep_low = l[t]
        # 2) RECLAIM within the window (the sweep bar closing above already counts)
        reclaim = None
        for k in range(t, min(n, t + cfg.sweep_max_bars + 1)):
            if c[k] > level:
                reclaim = k
                break
        if reclaim is None:
            t += 1
            continue
        # 3) BOS: close above the highest high since the sweep
        entry = None
        for k in range(reclaim + 1, min(n, reclaim + cfg.bos_max_bars + 1)):
            swing_high = h[t:k].max()
            if c[k] > swing_high:
                entry = k
                break
            if l[k] < sweep_low:       # setup invalidated before it triggered
                break
        if entry is None:
            t += 1
            continue
        stop = sweep_low * (1 - cfg.stop_buffer)
        risk_frac = (c[entry] - stop) / c[entry]
        if not (cfg.min_stop_frac <= risk_frac <= cfg.max_stop_frac):
            t += 1
            continue
        setups.append({"level_idx": lvl_idx, "sweep_idx": t, "reclaim_idx": reclaim, "entry_idx": entry,
                       "entry": c[entry], "stop": stop, "risk_frac": risk_frac, "time": df.index[entry]})
        t = entry + 1                  # no overlapping setups on the same asset
    return setups


def resolve(df: pd.DataFrame, s: dict, cfg: EntryCfg) -> dict:
    """Walk forward from the entry bar: stop, target or time stop, whichever comes first.
    Intrabar ambiguity is resolved conservatively — if a bar spans both, the stop is assumed first."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    n = len(c)
    e, stop = s["entry"], s["stop"]
    risk = e - stop
    target = e + cfg.target_r * risk
    for k in range(s["entry_idx"] + 1, min(n, s["entry_idx"] + cfg.time_stop_bars + 1)):
        if l[k] <= stop:
            return {**s, "exit_idx": k, "exit": stop, "r": -1.0, "reason": "stop"}
        if h[k] >= target:
            return {**s, "exit_idx": k, "exit": target, "r": cfg.target_r, "reason": "target"}
    k = min(n - 1, s["entry_idx"] + cfg.time_stop_bars)
    return {**s, "exit_idx": k, "exit": c[k], "r": (c[k] - e) / risk, "reason": "time"}


def run_asset(df: pd.DataFrame, cfg: EntryCfg, allowed: np.ndarray | None = None,
              cost_frac: float = 0.0) -> list[dict]:
    trades = []
    for s in find_setups(df, cfg, allowed):
        tr = resolve(df, s, cfg)
        # costs in R units: round-trip cost as a fraction of price, divided by the risk fraction
        tr["r_net"] = tr["r"] - cost_frac / s["risk_frac"]
        trades.append(tr)
    return trades


def equity_curve(trades: list[dict], cfg: EntryCfg, start: float = 5000.0) -> tuple[pd.Series, list[dict]]:
    """Compound the account trade by trade, risking a fixed fraction of CURRENT equity each time."""
    tr = sorted(trades, key=lambda x: x["time"])
    eq = start
    pts, out = [], []
    for t in tr:
        pnl = eq * cfg.risk_per_trade * t["r_net"]
        eq += pnl
        out.append({**t, "pnl": pnl, "equity": eq})
        pts.append((t["time"], eq))
        if eq <= 0:
            break
    if not pts:
        return pd.Series(dtype=float), out
    s = pd.Series([p[1] for p in pts], index=pd.DatetimeIndex([p[0] for p in pts]))
    return s, out


def metrics(trades: list[dict], cfg: EntryCfg, start: float = 5000.0, bars_per_year: int = 365,
            trading_days: float | None = None) -> dict:
    if not trades:
        return {"trades": 0}
    r = np.array([t["r_net"] for t in trades], dtype=float)
    wins, losses = r[r > 0], r[r <= 0]
    eq, det = equity_curve(trades, cfg, start)
    # longest losing streak
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    span_days = (trades[-1]["time"] - trades[0]["time"]).days or 1
    td = trading_days if trading_days else span_days
    ret = eq.pct_change().dropna() if len(eq) > 1 else pd.Series(dtype=float)
    per_year = len(trades) / (span_days / 365.25)
    sharpe = float(ret.mean() / ret.std() * math.sqrt(per_year)) if len(ret) > 2 and ret.std() > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    return {
        "trades": len(trades),
        "trades_per_year": round(per_year, 1),
        "winrate": round(float((r > 0).mean()), 3),
        "avg_win_R": round(float(wins.mean()), 3) if len(wins) else None,
        "avg_loss_R": round(float(losses.mean()), 3) if len(losses) else None,
        "expectancy_R": round(float(r.mean()), 3),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        "sharpe": round(sharpe, 3) if sharpe == sharpe else None,
        "max_drawdown": round(dd, 4),
        "max_losing_streak": int(best),
        "end_equity": round(float(eq.iloc[-1]), 2) if len(eq) else start,
        "eur_per_trading_day": round((float(eq.iloc[-1]) - start) / td, 2) if len(eq) else 0.0,
        "eur_per_trade": round(float(np.mean([d["pnl"] for d in det])), 2) if det else 0.0,
    }


def ruin_probabilities(trades: list[dict], cfg: EntryCfg, sims: int = 5000, horizon: int | None = None,
                       seed: int = 0) -> dict:
    """Bootstrap the trade sequence: probability of losing more than 20 % / 40 % of the account."""
    if not trades:
        return {}
    r = np.array([t["r_net"] for t in trades], dtype=float)
    span_days = (trades[-1]["time"] - trades[0]["time"]).days or 1
    n = horizon or max(10, int(round(len(trades) / (span_days / 365.25))))   # one year of trades
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(sims, n))
    paths = np.cumprod(1 + cfg.risk_per_trade * r[idx], axis=1)
    trough = paths.min(axis=1)
    return {"horizon_trades": int(n),
            "p_loss": round(float((paths[:, -1] < 1).mean()), 3),
            "p_dd_gt_20pct": round(float((trough < 0.8).mean()), 3),
            "p_dd_gt_40pct": round(float((trough < 0.6).mean()), 3),
            "median_year": round(float(np.median(paths[:, -1]) - 1), 4),
            "p05_year": round(float(np.percentile(paths[:, -1], 5) - 1), 4),
            "p95_year": round(float(np.percentile(paths[:, -1], 95) - 1), 4)}
