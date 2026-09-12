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
from .metrics import deflated_sharpe, summarize
from .risk import RiskManager
from .sizing import exposure_to_lots, lots_to_exposure
from .strategy import bars_needed, compute_exposure

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
    """Walk-forward / purged CV select among these in-sample. Axes: speed (lookbacks), risk (target vol),
    turnover (rebalance band), selectivity (unanimous entry), regime (efficiency-ratio trend filter)."""
    grid = []
    for lb in ([30, 90, 180], [60, 180, 360], [90, 270, 540]):
        for tv in (0.08, 0.12):
            for thr in (0.15, 0.35):
                for entry in (0.0, 1.0):
                    for regime in ({"er_window": 0, "er_min": 0.0}, {"er_window": 60, "er_min": 0.25}):
                        grid.append({"lookbacks": lb, "target_vol": tv, "rebalance_threshold": thr,
                                     "entry_min_signal": entry, "regime": regime})
    return grid


def grid_label(p: dict) -> str:
    r = p.get("regime") or {}
    return (f"lb={p['lookbacks']} tv={p['target_vol']} thr={p.get('rebalance_threshold')} "
            f"entry={p.get('entry_min_signal', 0)} er={'on' if r.get('er_window') else 'off'}")


ROBUST_W = {"worst": 0.5, "stab": 0.3, "mdd": 2.0, "turnover": 0.1}


def robust_score(sharpes: list[float], maxdds: list[float], trades_per_year: float, w: dict = ROBUST_W) -> float:
    """Selection objective: median segment Sharpe, penalised for the worst segment, instability across
    segments, drawdown and turnover. A parameter set that shines in two segments and fails in four
    loses against one that is merely decent everywhere."""
    if not sharpes:
        return -np.inf
    med, worst, stab = float(np.median(sharpes)), float(min(sharpes)), float(np.std(sharpes))
    return (med - w["worst"] * max(0.0, -worst) - w["stab"] * stab
            - w["mdd"] * float(np.mean([abs(m) for m in maxdds])) - w["turnover"] * trades_per_year / 100.0)


def select_params(df: pd.DataFrame, expos: list[pd.Series], cfg: Config, k: int = 4,
                  min_trades: int = 5, selector: str = "robust") -> tuple[int, float, list[float]]:
    """Choose the grid index on a training slice. 'robust': score across k contiguous segments;
    'sharpe': classic aggregate Sharpe. Returns (index, score, per-bar Sharpe of every trial for DSR)."""
    n = len(df)
    bpy = cfg.bars_per_year
    segs = [(i * n // k, (i + 1) * n // k) for i in range(k)] if selector == "robust" else [(0, n)]
    best_i, best_sc, trials = 0, -np.inf, []
    for gi, ex in enumerate(expos):
        shs, mdds, trades = [], [], 0
        for (x, y) in segs:
            m = simulate(df.iloc[x:y], ex.iloc[x:y], cfg).metrics
            trades += m["trades"]
            if m["trades"] >= min_trades:
                shs.append(m["sharpe"]); mdds.append(m["max_drawdown"])
        full_sr = float(np.mean(shs)) if shs else float("nan")
        trials.append(full_sr / np.sqrt(bpy) if full_sr == full_sr else float("nan"))
        if selector == "robust":
            sc = robust_score(shs, mdds, trades / (n / bpy)) if len(shs) >= max(2, k // 2) else -np.inf
        else:
            sc = shs[0] if shs else -np.inf
        if sc > best_sc:
            best_i, best_sc = gi, sc
    return best_i, best_sc, trials


@dataclass
class WalkForwardResult:
    equity: pd.Series
    windows: list[dict]
    metrics: dict
    trades: int


def walk_forward(df: pd.DataFrame, cfg: Config, grid: list[dict] | None = None,
                 train_bars: int | None = None, test_bars: int | None = None,
                 min_trades: int = 10, selector: str = "robust") -> WalkForwardResult:
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
    all_trials: list[float] = []
    start = train_bars
    while start + test_bars <= len(df):
        tr, te = slice(start - train_bars, start), slice(start, start + test_bars)
        best_i, best_sc, trials = select_params(df.iloc[tr], [e.iloc[tr] for e in expos], cfg,
                                                min_trades=max(3, min_trades // 3), selector=selector)
        all_trials += trials
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
    dsr = deflated_sharpe(equity.pct_change().dropna(), all_trials)
    m["dsr"] = round(float(dsr), 3) if dsr == dsr else None
    m["selector"] = selector
    m["n_trials"] = len(grid)
    return WalkForwardResult(equity, windows, m, n_trades)


@dataclass
class PurgedCVResult:
    folds: list[dict]
    summary: dict


def purged_cv(df: pd.DataFrame, cfg: Config, grid: list[dict] | None = None, k: int = 6,
              embargo_bars: int | None = None, min_trades: int = 10) -> PurgedCVResult:
    """Purged K-fold CV with embargo (López de Prado): for each fold, parameters are chosen on the
    OTHER folds with an embargo of at least the feature memory around the test fold, then evaluated
    on the fold. Gives K roughly independent out-of-sample segments instead of one chained curve."""
    grid = grid or default_grid()
    n = len(df)
    if embargo_bars is None:
        embargo_bars = max(bars_needed(cfg.with_strategy(**p)) for p in grid)
    expos = [compute_exposure(df, cfg.with_strategy(**p))["exposure"] for p in grid]
    bounds = [(i * n // k, (i + 1) * n // k) for i in range(k)]
    min_seg = max(cfg.bars_per_year // 4, 200)
    folds: list[dict] = []
    for fi, (a, b) in enumerate(bounds):
        lo, hi = max(0, a - embargo_bars), min(n, b + embargo_bars)
        pieces = [(x, y) for (x, y) in ((0, lo), (hi, n)) if y - x >= min_seg]
        # split every training piece in two -> up to 4 segments for the robust score
        segs = []
        for (x, y) in pieces:
            mid = (x + y) // 2
            segs += [(x, mid), (mid, y)] if (y - x) >= 2 * min_seg else [(x, y)]
        best_i, best_sc = 0, -np.inf
        for gi in range(len(grid)):
            shs, mdds, trades = [], [], 0
            for (x, y) in segs:
                m = simulate(df.iloc[x:y], expos[gi].iloc[x:y], cfg).metrics
                trades += m["trades"]
                if m["trades"] >= max(3, min_trades // 3):
                    shs.append(m["sharpe"]); mdds.append(m["max_drawdown"])
            sc = robust_score(shs, mdds, trades / (sum(y - x for x, y in segs) / cfg.bars_per_year)) if len(shs) >= 2 else -np.inf
            if sc > best_sc:
                best_i, best_sc = gi, sc
        res = simulate(df.iloc[a:b], expos[best_i].iloc[a:b], cfg)
        folds.append({"fold": fi, "test_start": str(df.index[a]), "test_end": str(df.index[b - 1]),
                      "params": grid[best_i], "train_sharpe": None if best_sc == -np.inf else round(float(best_sc), 3),
                      "test_sharpe": res.metrics["sharpe"], "test_return": res.metrics["total_return"],
                      "test_maxdd": res.metrics["max_drawdown"], "test_trades": res.metrics["trades"],
                      "test_cost": res.metrics["total_cost"]})
    sh = [f["test_sharpe"] for f in folds]
    summary = {"folds": k, "embargo_bars": embargo_bars, "mean_oos_sharpe": round(float(np.mean(sh)), 3),
               "median_oos_sharpe": round(float(np.median(sh)), 3), "positive_folds": int(sum(x > 0 for x in sh)),
               "worst_fold_sharpe": round(float(min(sh)), 3), "total_oos_trades": int(sum(f["test_trades"] for f in folds))}
    return PurgedCVResult(folds, summary)


def _cap_weights(w: dict, cap: float) -> dict:
    """Clip any weight above `cap`, redistribute the excess to the others (a few passes)."""
    w = dict(w)
    for _ in range(5):
        over = {k: v - cap for k, v in w.items() if v > cap}
        if not over:
            break
        excess = sum(over.values())
        for k in over:
            w[k] = cap
        free = [k for k in w if k not in over and w[k] > 0]
        if not free:
            break
        tot = sum(w[k] for k in free)
        for k in free:
            w[k] += excess * (w[k] / tot if tot > 0 else 1.0 / len(free))
    return w


def dynamic_portfolio(sleeves: dict[str, "WalkForwardResult"], cfg: Config, start_equity: float,
                      shrink: float = 0.5, floor: float = 0.0, cap: float = 0.6) -> tuple[pd.Series, dict, list[dict]]:
    """Allocate across sleeves (e.g. XAU-long, XAU-short, BTC-long, BTC-short) from trailing
    out-of-sample quality only: at every window start, weight_i ∝ max(0, shrink × mean OOS Sharpe of the
    sleeve's completed windows). No look-ahead — a sleeve earns weight only after it has proven itself
    OOS; sleeves with nothing proven share equally until then."""
    rets = pd.concat({k: v.equity.pct_change() for k, v in sleeves.items()}, axis=1).fillna(0.0).sort_index()
    starts = sorted({pd.Timestamp(w["test_start"]) for v in sleeves.values() for w in v.windows})
    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    history = []
    for i, t in enumerate(starts):
        t_end = starts[i + 1] if i + 1 < len(starts) else rets.index[-1] + pd.Timedelta(seconds=1)
        q = {}
        for k, v in sleeves.items():
            done = [w["test_sharpe"] for w in v.windows if pd.Timestamp(w["test_end"]) < t]
            # shrink the trailing mean towards zero by 1/sqrt(n): one good window proves little
            q[k] = max(0.0, float(np.mean(done)) * (1 - shrink / np.sqrt(len(done)))) if done else None
        if all(x is None for x in q.values()):
            w = {k: 1.0 / len(q) for k in q}          # nothing evaluated yet: start equal-weight
        else:
            raw = {k: (x if x is not None else 0.0) + floor for k, x in q.items()}
            tot = sum(raw.values())
            if tot <= 0:
                w = {k: 0.0 for k in raw}              # nothing proven: stay in cash, do not spread bets on hope
            else:
                w = _cap_weights({k: x / tot for k, x in raw.items()}, cap)
        mask = (rets.index >= t) & (rets.index < t_end)
        for k in w:
            weights.loc[mask, k] = w[k]
        history.append({"from": str(t), **{k: round(x, 3) for k, x in w.items()}})
    port_ret = (weights * rets).sum(axis=1)           # cash periods contribute 0
    port_ret = port_ret[rets.index >= starts[0]] if starts else port_ret
    equity = start_equity * (1.0 + port_ret).cumprod()
    equity.name = "portfolio"
    return equity, {k: round(float(weights[k].iloc[-1]), 3) for k in weights.columns}, history


GATES = {"median_cv_sharpe": 0.5, "positive_folds": 4, "portfolio_sharpe": 1.0, "portfolio_psr": 0.95,
         "portfolio_maxdd": -0.15, "prob_loss_1y": 0.20}


def evaluate_gates(report: dict) -> dict:
    """Quality targets for the next version (not guarantees): each entry is (value, threshold, pass)."""
    out = {}
    for sym, e in report.get("symbols", {}).items():
        cv = (e.get("purged_cv") or {}).get("summary")
        if cv:
            out[f"{sym.split('/')[0]} median purged-CV Sharpe > {GATES['median_cv_sharpe']}"] = (
                cv["median_oos_sharpe"], bool(cv["median_oos_sharpe"] > GATES["median_cv_sharpe"]))
            out[f"{sym.split('/')[0]} positive folds >= {GATES['positive_folds']}/{cv['folds']}"] = (
                cv["positive_folds"], bool(cv["positive_folds"] >= GATES["positive_folds"]))
    pm = (report.get("portfolio") or {}).get("metrics") or {}
    if pm:
        out[f"portfolio Sharpe > {GATES['portfolio_sharpe']}"] = (pm.get("sharpe"), bool((pm.get("sharpe") or 0) > GATES["portfolio_sharpe"]))
        out[f"portfolio PSR > {GATES['portfolio_psr']}"] = (pm.get("psr_gt_0"), bool((pm.get("psr_gt_0") or 0) > GATES["portfolio_psr"]))
        out[f"portfolio MaxDD > {GATES['portfolio_maxdd']:.0%}"] = (pm.get("max_drawdown"), bool((pm.get("max_drawdown") or -1) > GATES["portfolio_maxdd"]))
    bs = report.get("bootstrap_1y") or {}
    if bs:
        out[f"bootstrap P(loss 1y) < {GATES['prob_loss_1y']:.0%}"] = (bs.get("prob_loss"), bool((bs.get("prob_loss") or 1) < GATES["prob_loss_1y"]))
    return out
