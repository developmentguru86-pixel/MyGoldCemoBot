"""Pull OHLCV history from the exchange (public endpoints) and optionally sync config.yaml
with the market's contract size / lot step / taker fee / spread / funding."""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot.broker.ccxt_broker import CcxtBroker  # noqa: E402
from goldbot.config import Config  # noqa: E402
from goldbot.data import save_csv  # noqa: E402


def setv(text: str, key: str, val) -> str:
    return re.sub(rf"^(\s*{key}:\s*)[-\d.eE+]+", lambda m: f"{m.group(1)}{val}", text, count=1, flags=re.M)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--years", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--update-config", action="store_true")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    br = CcxtBroker(cfg)
    info = br.get_symbol_info(cfg.symbol)
    print("market:", info)
    now = int(time.time() * 1000)
    since = now - int(a.years * 365.25 * 86400 * 1000)
    df = br.get_bars_range(cfg.symbol, cfg.timeframe, since, now)
    if df.empty:
        raise SystemExit("no OHLCV returned — check symbol (ccxt unified, e.g. XAU/USDT:USDT)")
    hs = cfg.exchange.history_symbol
    if hs:
        alt = br.get_bars_range(hs, cfg.timeframe, since, now)
        if len(alt) > len(df) * 1.2:
            print(f"using {hs} as history proxy: {len(alt)} bars vs {len(df)} for {cfg.symbol} "
                  f"(same underlying, costs still modelled from {cfg.symbol})")
            df = alt
    df = df.iloc[:-1]  # drop forming bar
    out = a.out or cfg.paths.get("data", "data/history.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, out)
    yrs = len(df) / cfg.bars_per_year
    print(f"wrote {len(df)} bars ({yrs:.2f} years) to {out}: {df.index[0]} .. {df.index[-1]}")
    if yrs < 2.5:
        print(f"!!! only {yrs:.1f} years of history for {cfg.symbol} — walk-forward will use shorter windows, low confidence")

    if a.update_config:
        q = br.get_quote(cfg.symbol)
        fee = br.taker_fee(cfg.symbol)
        fund = br.funding_annual(cfg.symbol)
        txt = open(a.config, encoding="utf-8").read()
        txt = setv(txt, "size", info.contract_size)
        txt = setv(txt, "min_lot", info.min_lot)
        txt = setv(txt, "lot_step", info.lot_step)
        txt = setv(txt, "max_lot", info.max_lot)
        txt = setv(txt, "spread", round(q.spread, 4))
        if fee is not None:
            txt = setv(txt, "fee_pct", fee)
        if fund is not None:
            txt = setv(txt, "swap_long_annual", round(-fund, 5))
            txt = setv(txt, "swap_short_annual", round(fund, 5))
        open(a.config, "w", encoding="utf-8").write(txt)
        print(f"config.yaml updated: contract {info.contract_size}/{info.min_lot}/{info.lot_step}, "
              f"spread {q.spread:.4f}, taker fee {fee}, funding annualised {fund}")
