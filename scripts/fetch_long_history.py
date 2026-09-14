"""Decades of DAILY history for research: spot gold since 2000+, BTC since 2014, ETH since 2016.
Sources: stooq.com CSV (no key), fallback Yahoo via yfinance. Output data/<slug>_D1_long.csv.
Venue costs still come from data/<slug>_D1_meta.json (fetched by fetch_history_ccxt.py)."""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot.config import Config  # noqa: E402
from goldbot.data import save_csv  # noqa: E402

SOURCES = {  # slug -> (stooq symbol, yahoo symbol)
    "xau": ("xauusd", "GC=F"),
    "btc": ("btcusd", "BTC-USD"),
    "eth": ("ethusd", "ETH-USD"),
    "sol": ("solusd", "SOL-USD"),
    "xrp": ("xrpusd", "XRP-USD"),
}


def from_stooq(symbol: str) -> pd.DataFrame | None:
    try:
        r = requests.get(f"https://stooq.com/q/d/l/?s={symbol}&i=d", timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or "Date" not in r.text[:200]:
            print(f"  stooq {symbol}: http {r.status_code} / unexpected body: {r.text[:80]!r}"); return None
        df = pd.read_csv(io.StringIO(r.text))
        df = df.rename(columns={"Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "tick_volume"})
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time")[["open", "high", "low", "close"]].dropna()
    except Exception as e:  # noqa: BLE001
        print(f"  stooq {symbol}: {type(e).__name__}: {str(e)[:80]}"); return None


def from_yahoo(symbol: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            print(f"  yahoo {symbol}: empty"); return None
        h = h.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        h.index = pd.to_datetime(h.index, utc=True).normalize()
        return h[["open", "high", "low", "close"]].dropna()
    except Exception as e:  # noqa: BLE001
        print(f"  yahoo {symbol}: {type(e).__name__}: {str(e)[:80]}"); return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--since", default="2000-01-01")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    Path("data").mkdir(exist_ok=True)
    for sym in cfg.portfolio():
        slug = sym.split("/")[0].lower()
        if slug not in SOURCES:
            print(f"{sym}: no long-history source configured"); continue
        st, yh = SOURCES[slug]
        best = None
        for name, fn, arg in (("stooq", from_stooq, st), ("yahoo", from_yahoo, yh)):
            df = fn(arg)
            if df is not None and len(df) > 500:
                print(f"  {name} {arg}: {len(df)} bars {df.index[0].date()} .. {df.index[-1].date()}")
                if best is None or len(df) > len(best) * 1.1:
                    best = df
        if best is None:
            print(f"{sym}: no long history available"); continue
        best = best[best.index >= pd.Timestamp(a.since, tz="UTC")]
        best = best[(best["close"] > 0) & (best["high"] >= best["low"])]
        save_csv(best, f"data/{slug}_D1_long.csv")
        yrs = (best.index[-1] - best.index[0]).days / 365.25
        print(f"{sym}: wrote {len(best)} daily bars, {yrs:.1f} years ({len(best) / yrs:.0f} bars/year) -> data/{slug}_D1_long.csv")
