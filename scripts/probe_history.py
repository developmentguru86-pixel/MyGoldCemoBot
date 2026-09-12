"""How much OHLCV history can we actually pull from the exchange, and with which pagination?"""
from __future__ import annotations

import sys
import time
import ccxt
import pandas as pd

ex_id = sys.argv[1] if len(sys.argv) > 1 else "okx"
symbols = sys.argv[2:] or ["XAU/USDT:USDT", "PAXG/USDT", "PAXG/USDT:USDT"]
ex = getattr(ccxt, ex_id)({"enableRateLimit": True, "timeout": 30000, "options": {"defaultType": "swap"}})
mk = ex.load_markets()
now = ex.milliseconds()
since6y = now - int(6 * 365.25 * 86400 * 1000)
tf = "4h"


def report(tag, rows):
    if not rows:
        print(f"  {tag:32s} 0 bars"); return
    ts = sorted({r[0] for r in rows})
    print(f"  {tag:32s} {len(ts):6d} bars  {pd.Timestamp(ts[0], unit='ms', tz='UTC').date()} .. {pd.Timestamp(ts[-1], unit='ms', tz='UTC').date()}")


for sym in symbols:
    if sym not in mk:
        print(f"{sym}: NOT LISTED"); continue
    m = mk[sym]
    print(f"{sym}: type={m.get('type')} contractSize={m.get('contractSize')} created={m.get('created')}")
    # A) forward pagination from 6y ago
    try:
        out, cursor, n = [], since6y, 0
        while cursor < now and n < 300:
            rows = ex.fetch_ohlcv(sym, tf, since=cursor, limit=300); n += 1
            if not rows: break
            out += rows
            nxt = rows[-1][0] + 1
            if nxt <= cursor: break
            cursor = nxt
        report("A forward since=6y", out)
    except Exception as e:
        print("  A forward failed:", str(e)[:120])
    # B) backward with until
    try:
        out, end, n = [], now, 0
        while n < 300:
            rows = ex.fetch_ohlcv(sym, tf, limit=100, params={"until": end}); n += 1
            if not rows: break
            out = rows + out
            first = rows[0][0]
            if first <= since6y or len(rows) < 2: break
            end = first - 1
        report("B backward until", out)
    except Exception as e:
        print("  B backward failed:", str(e)[:120])
    # C) ccxt built-in pagination
    try:
        rows = ex.fetch_ohlcv(sym, tf, since=since6y, params={"paginate": True, "paginationCalls": 300})
        report("C ccxt paginate=True", rows)
    except Exception as e:
        print("  C paginate failed:", str(e)[:120])
