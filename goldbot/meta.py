"""Meta-labeling (López de Prado): the primary model decides the side, a secondary model decides
whether to take the trade at all.

Events   = bars where the primary exposure opens or flips a position.
Labels   = triple barrier from the entry: +/- k * sigma_h (vol-scaled), vertical barrier h bars;
           label 1 if the upper barrier (in trade direction) is hit first, else sign of the return
           at the vertical barrier.
Features = 10 backward-looking numbers available at the event bar.
Model    = calibrated HistGradientBoosting (shallow, regularised) averaged with a logistic
           regression — small data, small model.
Filter   = expected value: take the trade only if (2p-1) * barrier > round-trip cost + premium.
Evaluation = purged K-fold with embargo, model fitted on the other folds only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import simulate
from .config import Config
from .strategy import compute_exposure


# ---------------------------------------------------------------- events, labels, features
def primary_events(exposure: pd.Series) -> np.ndarray:
    """Indices where a position is opened or flipped (sign changes from 0 or to the opposite sign)."""
    sgn = np.sign(exposure.to_numpy(dtype=float))
    prev = np.r_[0.0, sgn[:-1]]
    return np.where((sgn != 0) & (sgn != prev))[0]


def triple_barrier(close: np.ndarray, vol_bar: np.ndarray, events: np.ndarray, sides: np.ndarray,
                   k: float = 1.0, h: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels in {0,1}, barrier widths as return fractions) for each event."""
    n = len(close)
    labels = np.zeros(len(events), dtype=int)
    widths = np.zeros(len(events))
    for j, (t, side) in enumerate(zip(events, sides)):
        sigma = vol_bar[t] * np.sqrt(h) if np.isfinite(vol_bar[t]) else np.nan
        if not np.isfinite(sigma) or sigma <= 0:
            widths[j] = np.nan
            continue
        width = k * sigma
        widths[j] = width
        end = min(n - 1, t + h)
        path = (close[t + 1:end + 1] / close[t] - 1.0) * side
        hit_up = np.where(path >= width)[0]
        hit_dn = np.where(path <= -width)[0]
        first_up = hit_up[0] if len(hit_up) else 10**9
        first_dn = hit_dn[0] if len(hit_dn) else 10**9
        if first_up < first_dn:
            labels[j] = 1
        elif first_dn < first_up:
            labels[j] = 0
        else:
            labels[j] = int(path[-1] > 0) if len(path) else 0
    return labels, widths


def build_features(df: pd.DataFrame, feats: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """10 features, all computed from information up to and including the bar."""
    close = df["close"]
    s = cfg.strategy
    ret = close.pct_change()
    vol_bar = ret.ewm(span=s.vol_span, min_periods=s.vol_span).std()
    out = pd.DataFrame(index=df.index)
    for i, L in enumerate(s.lookbacks[:3]):
        out[f"z{i}"] = ((close / close.shift(L) - 1.0) / (vol_bar * np.sqrt(L))).clip(-5, 5)
    out["agree"] = feats["signal"]
    out["vol_pct"] = feats["vol_ann"].rolling(500, min_periods=100).rank(pct=True)
    er_w = max(s.regime.er_window, 20)
    out["er"] = ((close - close.shift(er_w)).abs() / close.diff().abs().rolling(er_w).sum().replace(0, np.nan)).fillna(0)
    out["ret5"] = (close / close.shift(5) - 1.0) / (vol_bar * np.sqrt(5))
    rng_hi, rng_lo = close.rolling(50).max(), close.rolling(50).min()
    out["range_pos"] = ((close - rng_lo) / (rng_hi - rng_lo).replace(0, np.nan)).fillna(0.5)
    out["kelly"] = feats["kelly_mult"]
    out["vol_chg"] = (vol_bar / vol_bar.shift(20) - 1.0).clip(-2, 2)
    out["hurst"] = rolling_hurst(np.log(close.to_numpy(dtype=float)), window=100)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_hurst(logp: np.ndarray, window: int = 100) -> np.ndarray:
    """Rolling variance-ratio Hurst estimate: H ~ 0.5 random walk, > 0.5 trending, < 0.5 mean-reverting.
    Uses lags 1, 2, 4, 8, 16 of the log-price increments."""
    n = len(logp)
    out = np.full(n, np.nan)
    lags = np.array([1, 2, 4, 8, 16])
    for t in range(window, n):
        seg = logp[t - window:t + 1]
        taus = []
        for lag in lags:
            d = seg[lag:] - seg[:-lag]
            taus.append(np.std(d) if len(d) > 2 else np.nan)
        taus = np.array(taus)
        if np.all(np.isfinite(taus)) and np.all(taus > 0):
            slope = np.polyfit(np.log(lags), np.log(taus), 1)[0]
            out[t] = min(max(slope, 0.0), 1.0)
    return out


# ---------------------------------------------------------------- model
def fit_model(X: np.ndarray, y: np.ndarray):
    """Calibrated shallow GBM + logistic regression, averaged. Returns a predict_proba callable."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(y) < 40 or len(set(y)) < 2:
        return None
    gbm = make_pipeline(SimpleImputer(), HistGradientBoostingClassifier(
        max_depth=3, max_iter=60, learning_rate=0.05, min_samples_leaf=max(10, len(y) // 20),
        l2_regularization=1.0, random_state=0))
    lr = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
    cv = min(3, max(2, len(y) // 40))
    gbm_c = CalibratedClassifierCV(gbm, method="sigmoid", cv=cv).fit(X, y)
    lr.fit(X, y)

    def proba(Xn: np.ndarray) -> np.ndarray:
        return 0.5 * gbm_c.predict_proba(Xn)[:, 1] + 0.5 * lr.predict_proba(Xn)[:, 1]
    return proba


def ev_threshold(width: float, cost_rt: float, premium: float) -> float:
    """p* such that (2p-1) * width > cost_rt + premium  ->  p* = 0.5 + (cost_rt + premium) / (2 width)."""
    if not np.isfinite(width) or width <= 0:
        return 1.01
    return 0.5 + (cost_rt + premium) / (2.0 * width)


def apply_meta(exposure: pd.Series, events: np.ndarray, accept: np.ndarray) -> pd.Series:
    """Zero the primary exposure for rejected events until the primary itself goes flat / flips."""
    e = exposure.to_numpy(dtype=float).copy()
    sgn = np.sign(e)
    acc = dict(zip(events.tolist(), accept.tolist()))
    on = False
    for t in range(len(e)):
        if t in acc:
            on = bool(acc[t])
        elif sgn[t] == 0:
            on = False
        if not on:
            e[t] = 0.0
    return pd.Series(e, index=exposure.index)


# ---------------------------------------------------------------- evaluation
@dataclass
class MetaCVResult:
    folds: list[dict]
    summary: dict


def meta_cv(df: pd.DataFrame, cfg: Config, k: int = 6, embargo_bars: int | None = None,
            barrier_k: float = 1.0, horizon: int | None = None, premium: float = 0.0005) -> MetaCVResult:
    """Purged K-fold: primary vs meta-filtered performance per fold, secondary model fitted on the
    other folds only (embargoed). The comparison is like-for-like on the same bars."""
    s = cfg.strategy
    feats = compute_exposure(df, cfg)
    expo = feats["exposure"]
    close = df["close"].to_numpy(dtype=float)
    vol_bar = df["close"].pct_change().ewm(span=s.vol_span, min_periods=s.vol_span).std().to_numpy(dtype=float)
    h = horizon or max(s.min_hold_bars * 5, 30)
    events = primary_events(expo)
    sides = np.sign(expo.to_numpy(dtype=float)[events])
    labels, widths = triple_barrier(close, vol_bar, events, sides, k=barrier_k, h=h)
    X_all = build_features(df, feats, cfg).to_numpy(dtype=float)
    Xe = X_all[events]
    price = close[events]
    c = cfg.costs
    cost_rt = 2 * ((c.spread / 2.0 + (price * c.slippage_pct if c.slippage_pct > 0 else c.slippage)) / price + c.fee_pct)

    n = len(df)
    embargo = embargo_bars or (max(s.lookbacks) + s.kelly.window + 3 * s.vol_span + h)
    bounds = [(i * n // k, (i + 1) * n // k) for i in range(k)]
    folds = []
    for fi, (a, b) in enumerate(bounds):
        lo, hi = max(0, a - embargo), min(n, b + embargo)
        tr_mask = ((events < lo) | (events >= hi)) & np.isfinite(widths) & (events + h < n)
        te_mask = (events >= a) & (events < b)
        model = fit_model(Xe[tr_mask], labels[tr_mask]) if tr_mask.sum() >= 40 else None
        accept = np.ones(len(events), dtype=bool)
        p_te = None
        if model is not None and te_mask.any():
            p_te = model(Xe[te_mask])
            thr = np.array([ev_threshold(w, cr, premium) for w, cr in zip(widths[te_mask], cost_rt[te_mask])])
            accept[te_mask] = p_te >= thr
        prim = simulate(df.iloc[a:b], expo.iloc[a:b], cfg).metrics
        meta_expo = apply_meta(expo, events, accept)
        meta = simulate(df.iloc[a:b], meta_expo.iloc[a:b], cfg).metrics
        folds.append({"fold": fi, "test_start": str(df.index[a]), "test_end": str(df.index[b - 1]),
                      "events": int(te_mask.sum()), "accepted": int(accept[te_mask].sum()) if te_mask.any() else 0,
                      "train_events": int(tr_mask.sum()), "base_rate": round(float(labels[tr_mask].mean()), 3) if tr_mask.sum() else None,
                      "mean_p": round(float(np.mean(p_te)), 3) if p_te is not None and len(p_te) else None,
                      "primary_sharpe": prim["sharpe"], "meta_sharpe": meta["sharpe"],
                      "primary_return": prim["total_return"], "meta_return": meta["total_return"],
                      "primary_trades": prim["trades"], "meta_trades": meta["trades"],
                      "primary_maxdd": prim["max_drawdown"], "meta_maxdd": meta["max_drawdown"]})
    ps = [f["primary_sharpe"] for f in folds]
    ms = [f["meta_sharpe"] for f in folds]
    summary = {"folds": k, "embargo_bars": embargo, "horizon": h, "barrier_k": barrier_k, "events": int(len(events)),
               "primary_median_sharpe": round(float(np.median(ps)), 3), "meta_median_sharpe": round(float(np.median(ms)), 3),
               "primary_positive_folds": int(sum(x > 0 for x in ps)), "meta_positive_folds": int(sum(x > 0 for x in ms)),
               "meta_better_folds": int(sum(m > p for m, p in zip(ms, ps))),
               "accepted_share": round(float(sum(f["accepted"] for f in folds) / max(1, sum(f["events"] for f in folds))), 3)}
    return MetaCVResult(folds, summary)


# ---------------------------------------------------------------- pooled across assets
def _asset_events(df: pd.DataFrame, cfg: Config, barrier_k: float, h: int):
    s = cfg.strategy
    feats = compute_exposure(df, cfg)
    expo = feats["exposure"]
    close = df["close"].to_numpy(dtype=float)
    vol_bar = df["close"].pct_change().ewm(span=s.vol_span, min_periods=s.vol_span).std().to_numpy(dtype=float)
    events = primary_events(expo)
    sides = np.sign(expo.to_numpy(dtype=float)[events])
    labels, widths = triple_barrier(close, vol_bar, events, sides, k=barrier_k, h=h)
    X = build_features(df, feats, cfg).to_numpy(dtype=float)[events]
    price = close[events]
    c = cfg.costs
    cost_rt = 2 * ((c.spread / 2.0 + (price * c.slippage_pct if c.slippage_pct > 0 else c.slippage)) / price + c.fee_pct)
    return expo, events, labels, widths, X, cost_rt, df.index[events]


def meta_cv_pooled(dfs: dict[str, pd.DataFrame], cfgs: dict[str, Config], k: int = 6,
                   barrier_k: float = 1.0, horizon: int | None = None, premium: float = 0.0005) -> dict:
    """One secondary model for all assets (asset identity as one-hot feature). Folds are calendar
    spans over the union of dates; the embargo is applied in time. Train on the other folds of ALL
    assets, evaluate each asset on its own fold slice — primary vs meta, like for like."""
    syms = list(dfs)
    any_cfg = cfgs[syms[0]]
    s = any_cfg.strategy
    h = horizon or max(s.min_hold_bars * 5, 30 if any_cfg.bars_per_day > 1 else 10)
    per = {sym: _asset_events(dfs[sym], cfgs[sym], barrier_k, h) for sym in syms}
    # pooled design matrix
    rows_X, rows_y, rows_w, rows_c, rows_t, rows_s = [], [], [], [], [], []
    for i, sym in enumerate(syms):
        expo, ev, y, w, X, cr, ts = per[sym]
        onehot = np.zeros((len(ev), len(syms))); onehot[:, i] = 1.0
        rows_X.append(np.hstack([X, onehot])); rows_y.append(y); rows_w.append(w); rows_c.append(cr)
        rows_t.append(ts.to_numpy()); rows_s.append(np.full(len(ev), i))
    X_all = np.vstack(rows_X); y_all = np.concatenate(rows_y); w_all = np.concatenate(rows_w)
    c_all = np.concatenate(rows_c); t_all = np.concatenate(rows_t); s_all = np.concatenate(rows_s)
    t0 = min(df.index[0] for df in dfs.values()); t1 = max(df.index[-1] for df in dfs.values())
    edges = [t0 + (t1 - t0) * i / k for i in range(k + 1)]
    tf_secs = 86400 // max(1, any_cfg.bars_per_day)
    embargo_bars = max(s.lookbacks) + s.kelly.window + 3 * s.vol_span + h
    emb = pd.Timedelta(seconds=tf_secs * embargo_bars)
    results = {sym: [] for sym in syms}
    for fi in range(k):
        a, b = edges[fi], edges[fi + 1]
        tr = ((t_all < a - emb) | (t_all >= b + emb)) & np.isfinite(w_all)
        model = fit_model(X_all[tr], y_all[tr]) if tr.sum() >= 40 else None
        for i, sym in enumerate(syms):
            expo, ev, y, w, X, cr, ts = per[sym]
            df = dfs[sym]
            te = (s_all == i) & (t_all >= a) & (t_all < b)
            accept = np.ones(len(ev), dtype=bool)
            p_te = None
            if model is not None and te.any():
                p_te = model(X_all[te])
                thr = np.array([ev_threshold(ww, cc, premium) for ww, cc in zip(w_all[te], c_all[te])])
                loc = (ts >= a) & (ts < b)
                accept[loc] = p_te >= thr
            sl = df.index[(df.index >= a) & (df.index < b)]
            if len(sl) < 30:
                continue
            ia, ib = df.index.get_loc(sl[0]), df.index.get_loc(sl[-1]) + 1
            prim = simulate(df.iloc[ia:ib], expo.iloc[ia:ib], cfgs[sym]).metrics
            meta = simulate(df.iloc[ia:ib], apply_meta(expo, ev, accept).iloc[ia:ib], cfgs[sym]).metrics
            results[sym].append({"fold": fi, "test_start": str(a)[:10], "test_end": str(b)[:10],
                                 "events": int(te.sum()), "accepted": int(accept[(ts >= a) & (ts < b)].sum()),
                                 "train_events": int(tr.sum()), "mean_p": round(float(np.mean(p_te)), 3) if p_te is not None and len(p_te) else None,
                                 "primary_sharpe": prim["sharpe"], "meta_sharpe": meta["sharpe"],
                                 "primary_return": prim["total_return"], "meta_return": meta["total_return"],
                                 "primary_trades": prim["trades"], "meta_trades": meta["trades"]})
    out = {"folds": k, "horizon": h, "embargo_bars": embargo_bars, "pooled_events": int(len(y_all)), "per_symbol": {}}
    all_p, all_m = [], []
    for sym in syms:
        fs = results[sym]
        ps = [f["primary_sharpe"] for f in fs]; ms = [f["meta_sharpe"] for f in fs]
        all_p += ps; all_m += ms
        out["per_symbol"][sym] = {"folds": fs, "primary_median_sharpe": round(float(np.median(ps)), 3) if ps else None,
                                  "meta_median_sharpe": round(float(np.median(ms)), 3) if ms else None,
                                  "primary_positive_folds": int(sum(x > 0 for x in ps)), "meta_positive_folds": int(sum(x > 0 for x in ms)),
                                  "meta_better_folds": int(sum(m > p for m, p in zip(ms, ps))), "n_folds": len(fs),
                                  "accepted_share": round(float(sum(f["accepted"] for f in fs) / max(1, sum(f["events"] for f in fs))), 3)}
    out["pooled_primary_median_sharpe"] = round(float(np.median(all_p)), 3) if all_p else None
    out["pooled_meta_median_sharpe"] = round(float(np.median(all_m)), 3) if all_m else None
    out["pooled_meta_better"] = f"{int(sum(m > p for m, p in zip(all_m, all_p)))}/{len(all_p)}"
    return out
