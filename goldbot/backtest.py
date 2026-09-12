"""Sequential backtest engine.

Sequential (not vectorised) on purpose: lot rounding depends on the equity path,
which on a 5k account is the dominant sizing effect. Same sizing/cost/risk code
as the live loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .costs import swap_cost, trade_cost
from .metrics import summarize
from .risk import RiskManager
from .sizing import exposure_to_lots, lots_to_exposure
from .strategy import compute_exposure

TRADE_COLS = ["time", "delta_lots", "lots_after", "price", "spread", "cost", "reason"]


@dataclass
class BacktestResult:
    equity: pd.Series
    lots: pd.Series
    exposure_target: pd.Series
    trades: pd.DataFrame
    total_cost: float
    total_swap: float
    halted_at: pd.Timestamp | None
    metrics: dict
    params: dict = field(default_factory=dict)


def should_trade(tgt_lots: float, cur_lots: float, tgt_exp: float, cur_exp: float, threshold: float) -> bool:
    """Single rebalance rule shared by backtest and live."""
    if tgt_lots == cur_lots:
        return False
    if tgt_lots == 0.0 or np.sign(tgt_lots) != np.sign(cur_lots):
        return True
    return abs(tgt_exp - cur_exp) >= threshold


def simulate(df: pd.DataFrame, exposure: pd.Series, cfg: Config,
             start_equity: float | None = None) -> BacktestResult:
    c, s, ct = cfg.costs, cfg.strategy, cfg.contract
    idx = df.index
    n = len(idx)
    close = df["close"].to_numpy(dtype=float)
    if "spread" in df.columns:
        spread = df["spread"].to_numpy(dtype=float)
        spread = np.where(np.isfinite(spread) & (spread > 0), spread, c.spread)
    else:
        spread = np.full(n, c.spread)
    target = exposure.reindex(idx).fillna(0.0).to_numpy(dtype=float)

    eq = float(cfg.starting_equity if start_equity is None else start_equity)
    rm = RiskManager(cfg.risk, eq, idx[0])
    lots = 0.0
    equity = np.empty(n)
    lots_hist = np.zeros(n)
    equity[0] = eq
    trades: list[tuple] = []
    total_cost = total_swap = 0.0
    halted_at = None

    for t in range(1, n):
        px = close[t]
        eq += lots * ct.size * (px - close[t - 1])
        sw = swap_cost(lots, px, ct.size, c.swap_long_annual, c.swap_short_annual, cfg.bars_per_year)
        eq -= sw
        total_swap += sw
        if eq <= 0:  # ruin
            equity[t:] = 0.0
            halted_at = idx[t]
            break

        may_hold, reason = rm.step(eq, idx[t])
        tgt_exp = target[t] if may_hold else 0.0
        tgt_lots = exposure_to_lots(tgt_exp, eq, px, ct, s.max_leverage)
        cur_exp = lots_to_exposure(lots, eq, px, ct)

        if should_trade(tgt_lots, lots, tgt_exp, cur_exp, s.rebalance_threshold):
            delta = tgt_lots - lots
            cost = trade_cost(delta, spread[t], c.slippage, ct.size, c.commission_per_lot, px, c.fee_pct)
            eq -= cost
            total_cost += cost
            lots = tgt_lots
            trades.append((idx[t], delta, lots, px, spread[t], cost, reason or "signal"))

        if rm.state.halted and halted_at is None:
            halted_at = idx[t]
        equity[t] = eq
        lots_hist[t] = lots

    eq_s = pd.Series(equity, index=idx, name="equity")
    lots_s = pd.Series(lots_hist, index=idx, name="lots")
    trades_df = pd.DataFrame(trades, columns=TRADE_COLS)
    m = summarize(eq_s, cfg.bars_per_year, cfg.bars_per_day,
                  trades=len(trades_df), total_cost=total_cost, total_swap=total_swap, lots=lots_s)
    m["halted_at"] = str(halted_at) if halted_at is not None else None
    return BacktestResult(eq_s, lots_s, pd.Series(target, index=idx, name="exposure_target"),
                          trades_df, total_cost, total_swap, halted_at, m)


def run_backtest(df: pd.DataFrame, cfg: Config) -> BacktestResult:
    feats = compute_exposure(df, cfg)
    res = simulate(df, feats["exposure"], cfg)
    res.params = {"lookbacks": cfg.strategy.lookbacks, "target_vol": cfg.strategy.target_vol,
                  "vol_span": cfg.strategy.vol_span}
    return res


def default_grid() -> list[dict]:
    grid = []
    for lb in ([15, 45, 90], [30, 90, 180], [60, 180, 360]):
        for tv in (0.08, 0.12, 0.16):
            grid.append({"lookbacks": lb, "target_vol": tv})
    return grid


@dataclass
class WalkForwardResult:
    equity: pd.Series
    windows: list[dict]
    metrics: dict
    trades: int


def walk_forward(df: pd.DataFrame, cfg: Config, grid: list[dict] | None = None,
                 train_bars: int | None = None, test_bars: int | None = None,
                 min_trades: int = 10) -> WalkForwardResult:
    """Rolling walk-forward: pick params by in-sample Sharpe, apply to the next unseen window,
    chain the out-of-sample equity (equity level carries over, position is re-established)."""
    grid = grid or default_grid()
    bpy = cfg.bars_per_year
    train_bars = train_bars or 2 * bpy
    test_bars = test_bars or bpy // 2
    if len(df) < train_bars + test_bars:
        raise ValueError(f"need >= {train_bars + test_bars} bars for walk-forward, have {len(df)}")

    expos = [compute_exposure(df, cfg.with_strategy(**p))["exposure"] for p in grid]
    windows: list[dict] = []
    parts: list[pd.Series] = []
    eq_start = cfg.starting_equity
    n_trades = 0
    tot_cost = tot_swap = 0.0
    start = train_bars
    while start + test_bars <= len(df):
        tr, te = slice(start - train_bars, start), slice(start, start + test_bars)
        best_i, best_sc = 0, -np.inf
        for i in range(len(grid)):
            m = simulate(df.iloc[tr], expos[i].iloc[tr], cfg).metrics
            sc = m["sharpe"] if m["trades"] >= min_trades else -np.inf
            if sc > best_sc:
                best_i, best_sc = i, sc
        res = simulate(df.iloc[te], expos[best_i].iloc[te], cfg, start_equity=eq_start)
        windows.append({
            "test_start": str(df.index[te.start]), "test_end": str(df.index[te.stop - 1]),
            "params": grid[best_i], "train_sharpe": None if best_sc == -np.inf else round(float(best_sc), 3),
            "test_sharpe": res.metrics["sharpe"], "test_return": res.metrics["total_return"],
            "test_maxdd": res.metrics["max_drawdown"], "test_trades": res.metrics["trades"],
            "halted_at": res.metrics["halted_at"],
        })
        parts.append(res.equity if not parts else res.equity.iloc[1:])
        eq_start = float(res.equity.iloc[-1])
        n_trades += res.metrics["trades"]
        tot_cost += res.total_cost
        tot_swap += res.total_swap
        start += test_bars
        if eq_start <= 0:
            break

    equity = pd.concat(parts)
    m = summarize(equity, bpy, cfg.bars_per_day, trades=n_trades, total_cost=tot_cost, total_swap=tot_swap)
    return WalkForwardResult(equity, windows, m, n_trades)
