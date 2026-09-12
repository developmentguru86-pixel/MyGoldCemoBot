"""Which exchanges are reachable from this machine (geo-blocking!) and offer gold-linked markets?"""
from __future__ import annotations

import sys
import ccxt

CANDIDATES = ["bybit", "okx", "bitget", "gateio", "kucoinfutures", "krakenfutures", "bingx", "mexc",
              "phemex", "hyperliquid", "coinbase", "kraken", "binanceusdm", "alpaca"]
GOLD = ("PAXG", "XAUT", "XAU", "GOLD")

for ex_id in CANDIDATES:
    try:
        ex = getattr(ccxt, ex_id)({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "swap"}})
        mk = ex.load_markets()
        gold = [s for s in mk if any(g in s.upper() for g in GOLD)]
        swaps = [s for s in gold if mk[s].get("swap")]
        spot = [s for s in gold if mk[s].get("spot")]
        demo = "demo" if hasattr(ex, "enable_demo_trading") else ("sandbox" if ex.urls.get("test") else "-")
        print(f"OK      {ex_id:14s} markets={len(mk):5d} {demo:8s} gold_swap={swaps[:6]} gold_spot={spot[:4]}", flush=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")[:140]
        print(f"BLOCKED {ex_id:14s} {type(e).__name__}: {msg}", flush=True)
