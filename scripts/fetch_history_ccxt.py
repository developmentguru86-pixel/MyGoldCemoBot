"""Pull OHLCV history for every portfolio symbol (venue first, then longer public proxies of the same
underlying) into data/<slug>_H4.csv, plus data/<slug>_meta.json with the venue's contract, spread,
fee and funding — the backtest reads those per symbol."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ccxt  # noqa: E402

from goldbot.broker.ccxt_broker import TF, CcxtBroker, fetch_ohlcv_range, rows_to_df  # noqa: E402
from goldbot.config import Config  # noqa: E402
from goldbot.data import save_csv  # noqa: E402


def slug(symbol: str) -> str:
    return symbol.split("/")[0].lower()


_EX: dict = {}


def public_exchange(ex_id: str):
    """One instance per venue so ccxt's rate limiter actually throttles across symbols."""
    if ex_id not in _EX:
        _EX[ex_id] = getattr(ccxt, ex_id)({"enableRateLimit": True, "timeout": 30000})
        if ex_id == "bitfinex":
            _EX[ex_id].rateLimit = max(_EX[ex_id].rateLimit, 2200)   # candles endpoint: ~30 req/min sustained
        _EX[ex_id].load_markets()
    return _EX[ex_id]


def fetch_symbol(br: CcxtBroker, cfg: Config, sym: str, sources: list[str], years: float) -> tuple:
    now = int(time.time() * 1000)
    since = now - int(years * 365.25 * 86400 * 1000)
    df = br.get_bars_range(sym, cfg.timeframe, since, now)
    if df.empty:
        raise SystemExit(f"no OHLCV returned for {sym}")
    print(f"{sym}: {len(df)} bars ({len(df) / cfg.bars_per_year:.2f} y) from {cfg.exchange.id}")
    best, best_name = df, sym
    tf = TF.get(cfg.timeframe.upper(), cfg.timeframe)
    enough = 0.9 * years * cfg.bars_per_year
    for cand in sources:
        if len(best) >= enough:
            break  # already have (nearly) the requested depth
        ex_id, alt_sym = cand.split(":", 1)
        for attempt in (1, 2):
            try:
                ex = public_exchange(ex_id)
                if alt_sym not in ex.markets:
                    print(f"  {cand}: not listed"); break
                alt = rows_to_df(fetch_ohlcv_range(ex, alt_sym, tf, since, now))
                print(f"  {cand}: {len(alt)} bars ({len(alt) / cfg.bars_per_year:.2f} y)")
                if len(alt) > len(best) * 1.2:
                    best, best_name = alt, cand
                break
            except ccxt.RateLimitExceeded:
                if attempt == 1:
                    print(f"  {cand}: rate limited, waiting 70s"); time.sleep(70)
                else:
                    print(f"  {cand}: rate limited twice, skipped")
            except Exception as e:  # noqa: BLE001
                print(f"  {cand}: failed {type(e).__name__}: {str(e)[:90]}"); break
    if best_name != sym:
        print(f"  using {best_name} as price history ({len(best)} bars); costs modelled from {sym}")
        df = best
    return df.iloc[:-1], best_name  # drop forming bar


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--years", type=float, default=6.0)
    ap.add_argument("--timeframe", default=None, help="override config timeframe, e.g. M15 (files get a suffix)")
    ap.add_argument("--update-config", action="store_true", help="kept for compatibility; meta json is authoritative")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    BARS_PER_DAY = {"M1": 1440, "M5": 288, "M15": 96, "M30": 48, "H1": 24, "H4": 6, "D1": 1}
    if a.timeframe:
        cfg.timeframe = a.timeframe.upper()
        cfg.bars_per_day = BARS_PER_DAY[cfg.timeframe]
    tfx = cfg.timeframe.upper()
    br = CcxtBroker(cfg)
    Path("data").mkdir(exist_ok=True)
    for sym, sc in cfg.portfolio().items():
        info = br.get_symbol_info(sym)
        df, src = fetch_symbol(br, cfg, sym, sc.history_sources, a.years)
        q = br.get_quote(sym)
        meta = {
            "symbol": sym, "history_source": src, "bars": len(df), "years": round(len(df) / cfg.bars_per_year, 2),
            "first": str(df.index[0]), "last": str(df.index[-1]),
            "contract_size": info.contract_size, "min_lot": info.min_lot, "lot_step": info.lot_step, "max_lot": info.max_lot,
            "spread": round(q.spread, 6), "price": round(q.mid, 4),
            "fee_pct": br.taker_fee(sym), "funding_annual": br.funding_annual(sym),
        }
        save_csv(df, f"data/{slug(sym)}_{tfx}.csv")
        Path(f"data/{slug(sym)}_{tfx}_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"  -> data/{slug(sym)}_{tfx}.csv ({meta['years']} y), spread {meta['spread']} fee {meta['fee_pct']} "
              f"funding {meta['funding_annual']} contract {info.contract_size}/{info.min_lot}/{info.lot_step}")
        if meta["years"] < 2.5:
            print(f"  !!! only {meta['years']} years for {sym} — low statistical confidence")
