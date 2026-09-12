"""Pull real XAU/USD history through the MT5 bridge into data/XAUUSD_H4.csv (chunked, resumable)."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import re

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot.broker.mt5_http import MT5HttpBroker  # noqa: E402
from goldbot.config import Config  # noqa: E402
from goldbot.data import save_csv  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--years", type=float, default=6.0)
    ap.add_argument("--chunk-days", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--update-config", action="store_true", help="write broker contract/spread/swap into config.yaml")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    br = MT5HttpBroker(cfg.bridge)
    info = br.get_symbol_info(cfg.symbol)
    print("symbol_info:", info)
    print(">>> put contract_size/min_lot/lot_step into config.yaml and check swap_long/swap_short "
          "(raw broker units, convert to annualised % of notional for costs.swap_*_annual)")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(a.years * 365.25))
    parts = []
    t0 = start
    while t0 < end:
        t1 = min(t0 + timedelta(days=a.chunk_days), end)
        df = br.get_bars_range(cfg.symbol, cfg.timeframe, t0.isoformat(), t1.isoformat())
        print(f"{t0.date()} .. {t1.date()}: {len(df)} bars")
        if len(df):
            parts.append(df)
        t0 = t1
    if not parts:
        raise SystemExit("no data returned — check MT5 'Max bars in chart' and that the symbol is in Market Watch")
    full = pd.concat(parts)
    full = full[~full.index.duplicated(keep="last")].sort_index()
    out = a.out or cfg.paths.get("data", "data/XAUUSD_H4.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_csv(full, out)
    med_spread = float(full["spread"].median())
    print(f"wrote {len(full)} bars to {out} ({full.index[0]} .. {full.index[-1]}); median spread {med_spread:.2f} USD/oz")

    if a.update_config:
        px = float(full["close"].iloc[-1])
        raw = br._get("/symbol_info", symbol=cfg.symbol)
        mode, sl, ss = raw.get("swap_mode"), raw.get("swap_long"), raw.get("swap_short")
        swaps = None
        if sl is not None and ss is not None:
            if mode == 1:      # points per lot per day
                swaps = (sl * info.point * 365 / px, ss * info.point * 365 / px)
            elif mode == 4:    # deposit currency per lot per day
                swaps = (sl * 365 / (px * info.contract_size), ss * 365 / (px * info.contract_size))
            elif mode in (5, 6):  # annual percent
                swaps = (sl / 100.0, ss / 100.0)
        txt = open(a.config, encoding="utf-8").read()
        def setv(text, key, val):
            return re.sub(rf"^(\s*{key}:\s*)[-\d.]+", lambda m: f"{m.group(1)}{val}", text, count=1, flags=re.M)
        txt = setv(txt, "size", info.contract_size)
        txt = setv(txt, "min_lot", info.min_lot)
        txt = setv(txt, "lot_step", info.lot_step)
        txt = setv(txt, "max_lot", info.max_lot)
        txt = setv(txt, "spread", round(med_spread, 3))
        if swaps:
            txt = setv(txt, "swap_long_annual", round(swaps[0], 5))
            txt = setv(txt, "swap_short_annual", round(swaps[1], 5))
            print(f"swaps (annualised, approx.): long {swaps[0]:+.4f}  short {swaps[1]:+.4f}  (mode {mode})")
        else:
            print(f"swap_mode {mode} not auto-converted — set costs.swap_*_annual manually")
        open(a.config, "w", encoding="utf-8").write(txt)
        print(f"config.yaml updated: contract {info.contract_size}/{info.min_lot}/{info.lot_step}/{info.max_lot}, spread {med_spread:.3f}")
