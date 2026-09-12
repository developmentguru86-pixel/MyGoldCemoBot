"""EUR conversion for reporting. USDT is treated as USD. Best effort, never raises."""
from __future__ import annotations

import logging

log = logging.getLogger("fx")
_cache: dict[str, float] = {}


def usd_per_eur() -> float | None:
    if "EURUSD" in _cache:
        return _cache["EURUSD"]
    try:
        import ccxt
        t = ccxt.kraken({"enableRateLimit": True, "timeout": 15000}).fetch_ticker("EUR/USD")
        px = float(t.get("last") or t.get("close") or 0)
        if px > 0:
            _cache["EURUSD"] = px
            return px
    except Exception as e:  # noqa: BLE001
        log.info("EUR/USD unavailable: %s", str(e)[:80])
    return None


def fmt_money(usd: float, sign: bool = True) -> str:
    r = usd_per_eur()
    s = f"{usd:+,.2f} USDT" if sign else f"{usd:,.2f} USDT"
    if r:
        eur = usd / r
        s += f" (≈ {eur:+,.2f} €)" if sign else f" (≈ {eur:,.2f} €)"
    return s
