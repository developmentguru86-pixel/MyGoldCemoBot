"""Fetch 5m history for the entry engine (venue first, public proxy as fallback)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ccxt  # noqa: E402

from goldbot.broker.ccxt_broker import fetch_ohlcv_range, rows_to_df  # noqa: E402
from goldbot.config import Config  # noqa: E402
from goldbot.data import save_csv  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--symbols", default="BTC/USDT:USDT,ETH/USDT:USDT")
    ap.add_argument("--days", type=float, default=540)
    ap.add_argument("--tf", default="5m")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    now = int(time.time() * 1000)
    since = now - int(a.days * 86400 * 1000)
    Path("data").mkdir(exist_ok=True)
    for sym in a.symbols.split(","):
        sym = sym.strip()
        base = sym.split("/")[0].lower()
        got = None
        for ex_id, s2 in ((cfg.exchange.id, sym), ("bitfinex", f"{sym.split('/')[0]}/USDT"), ("kraken", f"{sym.split('/')[0]}/USD")):
            try:
                ex = getattr(ccxt, ex_id)({"enableRateLimit": True, "timeout": 30000, "options": {"defaultType": "swap" if ":" in s2 else "spot"}})
                ex.load_markets()
                if s2 not in ex.markets:
                    print(f"  {ex_id}: {s2} not listed"); continue
                df = rows_to_df(fetch_ohlcv_range(ex, s2, a.tf, since, now))
                print(f"  {ex_id} {s2}: {len(df)} bars")
                if got is None or len(df) > len(got):
                    got = df
                if len(df) > 100000:
                    break
            except Exception as e:  # noqa: BLE001
                print(f"  {ex_id} {s2}: {type(e).__name__} {str(e)[:70]}")
        if got is None or got.empty:
            print(f"{sym}: no {a.tf} data"); continue
        save_csv(got, f"data/{base}_{a.tf.upper()}.csv")
        days = (got.index[-1] - got.index[0]).days
        print(f"{sym}: {len(got)} {a.tf} bars, {days} Tage -> data/{base}_{a.tf.upper()}.csv")
