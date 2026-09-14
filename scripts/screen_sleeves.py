"""Screen candidate sleeves: is it tradeable on the venue, does it have decades of daily history,
does it pass the CV gates, and is it uncorrelated with what we already hold?

Ranking rule: a sleeve only earns a place if it passes the gates on its OWN merit AND its OOS
returns are weakly correlated with the existing book. Correlation is the only factor that reliably
raised portfolio Sharpe in this project, so it is scored explicitly rather than hoped for.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot.backtest import daily_grid, purged_cv, walk_forward  # noqa: E402
from goldbot.config import Config  # noqa: E402
from goldbot.data import _finalize  # noqa: E402

# candidate -> (stooq symbol, yahoo symbol, venue base to look for)
CANDIDATES = {
    "XAG": ("xagusd", "SI=F", "XAG"),        # silver
    "WTI": ("cl.f", "CL=F", "OIL"),          # crude oil
    "COPPER": ("hg.f", "HG=F", "COPPER"),
    "NATGAS": ("ng.f", "NG=F", "GAS"),
    "SPX": ("^spx", "^GSPC", "SPX"),         # S&P 500
    "NDX": ("^ndx", "^NDX", "NDX"),
    "DAX": ("^dax", "^GDAXI", "DAX"),
    "EURUSD": ("eurusd", "EURUSD=X", "EUR"),
    "USDJPY": ("usdjpy", "JPY=X", "JPY"),
    "TLT": ("tlt.us", "TLT", "TLT"),         # long bonds
    "SOL": ("solusd", "SOL-USD", "SOL"),
    "XRP": ("xrpusd", "XRP-USD", "XRP"),
    "DOGE": ("dogeusd", "DOGE-USD", "DOGE"),
    "LTC": ("ltcusd", "LTC-USD", "LTC"),
    "LINK": ("linkusd", "LINK-USD", "LINK"),
    "ADA": ("adausd", "ADA-USD", "ADA"),
    "AVAX": ("avaxusd", "AVAX-USD", "AVAX"),
    "DOT": ("dotusd", "DOT-USD", "DOT"),
    "ATOM": ("atomusd", "ATOM-USD", "ATOM"),
    "BNB": ("bnbusd", "BNB-USD", "BNB"),
    "TRX": ("trxusd", "TRX-USD", "TRX"),
    "BCH": ("bchusd", "BCH-USD", "BCH"),
    "ETC": ("etcusd", "ETC-USD", "ETC"),
    "FIL": ("filusd", "FIL-USD", "FIL"),
    "NEAR": ("nearusd", "NEAR-USD", "NEAR"),
}


def stooq(sym: str) -> pd.DataFrame | None:
    try:
        r = requests.get(f"https://stooq.com/q/d/l/?s={sym}&i=d", timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or "Date" not in r.text[:200]:
            return None
        df = pd.read_csv(io.StringIO(r.text)).rename(
            columns={"Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close"})
        return _finalize(df[["time", "open", "high", "low", "close"]])
    except Exception:  # noqa: BLE001
        return None


def yahoo(sym: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        h = yf.Ticker(sym).history(period="max", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            return None
        h = h.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        h.index = pd.to_datetime(h.index, utc=True).normalize()
        h = h[["open", "high", "low", "close"]].dropna()
        h.index.name = "time"
        return _finalize(h.reset_index())
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--since", default="2005-01-01")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--spread-bps", type=float, default=2.0, help="assumed half-spread in bps of price (price-relative, not absolute)")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    grid = daily_grid()

    # --- what can we actually trade on the venue?
    ex = getattr(ccxt, cfg.exchange.id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    markets = ex.load_markets()
    live_bases = {m.split("/")[0].upper() for m, v in markets.items() if v.get("swap") and v.get("active")}
    print(f"venue {cfg.exchange.id}: {len(live_bases)} active perp bases | cost basis: {a.spread_bps} bps half-spread + 1 bp slippage + {cfg.costs.fee_pct:.2%} fee, price-relative\n")

    held = {s.split("/")[0].upper() for s in cfg.portfolio()}
    curves: dict[str, pd.Series] = {}
    rows = []

    # existing sleeves first, so candidates can be correlated against them
    for base in sorted(held):
        p = Path(f"data/{base.lower()}_D1_long.csv")
        if p.exists():
            df = pd.read_csv(p, parse_dates=["time"]).set_index("time")
            df.index = pd.to_datetime(df.index, utc=True)
            ch = cfg.with_strategy(direction="long")
            pxh = float(df["close"].median())
            ch.costs.spread = pxh * a.spread_bps / 1e4
            ch.costs.slippage = pxh * 1e-4
            ch.costs.slippage_pct = 0.0
            ch.costs.swap_long_annual = ch.costs.swap_short_annual = 0.0
            ch.contract.size = 1.0; ch.contract.min_lot = 1e-6; ch.contract.lot_step = 1e-6
            wf = walk_forward(df, ch, grid=grid, train_bars=3 * 365, test_bars=365)
            curves[base] = wf.equity.pct_change()
            print(f"[held] {base}: OOS Sharpe {wf.metrics['sharpe']}")

    for name, (st, yh, venue_base) in CANDIDATES.items():
        df = stooq(st)
        src = "stooq"
        if df is None or len(df) < 1500:
            df, src = yahoo(yh), "yahoo"
        if df is None or len(df) < 1500:
            print(f"{name:<7} no usable history"); continue
        df = df[df.index >= pd.Timestamp(a.since, tz="UTC")]
        df = df[df["close"] > 0]
        if len(df) < 1500:
            print(f"{name:<7} too short after {a.since}"); continue
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        c = cfg.with_strategy(direction="long")
        c.trading_days_per_year = int(round(len(df) / yrs)); c.bars_per_day = 1
        # costs must be PRICE-RELATIVE: the config's absolute spread (USDT) is calibrated for gold/BTC
        # and would be 25% of the price on copper, 90% on EURUSD. Scale to the instrument.
        px = float(df["close"].median())
        c.costs.spread = px * a.spread_bps / 1e4
        c.costs.slippage = px * 1e-4
        c.costs.slippage_pct = 0.0
        c.costs.swap_long_annual = 0.0
        c.costs.swap_short_annual = 0.0
        c.contract.size = 1.0
        c.contract.min_lot = 1e-6
        c.contract.lot_step = 1e-6
        try:
            wf = walk_forward(df, c, grid=grid, train_bars=3 * c.trading_days_per_year, test_bars=c.trading_days_per_year)
            cv = purged_cv(df, c, grid=grid, k=a.folds)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<7} failed: {type(e).__name__}: {str(e)[:70]}"); continue
        r = wf.equity.pct_change()
        cors = {k: float(pd.concat([r, v], axis=1).dropna().corr().iloc[0, 1]) for k, v in curves.items()}
        max_cor = max((abs(v) for v in cors.values()), default=0.0)
        gate = cv.summary["median_oos_sharpe"] > 0.5 and cv.summary["positive_folds"] >= 4
        rows.append({"name": name, "source": src, "years": round(yrs, 1), "bars": len(df),
                     "tradeable": venue_base in live_bases, "wf_sharpe": wf.metrics["sharpe"],
                     "wf_cagr": wf.metrics["cagr"], "wf_maxdd": wf.metrics["max_drawdown"],
                     "cv_median": cv.summary["median_oos_sharpe"], "cv_pos": cv.summary["positive_folds"],
                     "max_corr": round(max_cor, 2), "corr": {k: round(v, 2) for k, v in cors.items()},
                     "gate_pass": bool(gate)})
        curves[name] = r
        print(f"{name:<7} {yrs:>4.1f}y {src:<6} venue={'yes' if venue_base in live_bases else 'NO ':<3} "
              f"WF SR {wf.metrics['sharpe']:>6} CAGR {wf.metrics['cagr']:>7.1%} | CV median {cv.summary['median_oos_sharpe']:>6} "
              f"pos {cv.summary['positive_folds']}/{a.folds} | max|corr| {max_cor:.2f} {'<- GATE PASS' if gate else ''}")

    rows.sort(key=lambda x: (x["gate_pass"], -x["max_corr"], x["cv_median"]), reverse=True)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/sleeve_screen.json").write_text(json.dumps(rows, indent=1))
    print("\n== RANKING (gates first, then low correlation, then CV median) ==")
    for r in rows[:12]:
        print(f"  {'PASS' if r['gate_pass'] else 'fail'} {r['name']:<7} tradeable={r['tradeable']} "
              f"CV {r['cv_median']:>6} ({r['cv_pos']}/{a.folds})  WF SR {r['wf_sharpe']:>6}  max|corr| {r['max_corr']:.2f}  {r['corr']}")
    keep = [r for r in rows if r["gate_pass"] and r["tradeable"] and r["max_corr"] < 0.35]
    print(f"\nqualified new sleeves (gates + tradeable + |corr| < 0.35): {[r['name'] for r in keep] or 'none'}")
