"""Exchange broker via ccxt (Bybit USDT-perpetuals, demo or real). One-way position mode."""
from __future__ import annotations

import os

import logging

import ccxt
import pandas as pd

log = logging.getLogger("ccxt_broker")

from ..config import Config
from ..data import from_records
from .base import Account, Broker, Quote, SymbolInfo

TF = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d"}


def fetch_ohlcv_range(ex, symbol: str, tf: str, since_ms: int, until_ms: int, limit: int = 1000) -> list:
    """Forward pagination from since_ms; if the venue returns nothing (listing after since_ms),
    page backwards from until_ms. Returns deduplicated, sorted raw OHLCV rows."""
    out, cursor = [], since_ms
    for _ in range(400):
        if cursor >= until_ms:
            break
        rows = ex.fetch_ohlcv(symbol, tf, since=cursor, limit=limit)
        if not rows:
            break
        out += rows
        nxt = rows[-1][0] + 1
        if nxt <= cursor:
            break
        cursor = nxt
    if not out:
        end = until_ms
        for _ in range(200):
            rows = ex.fetch_ohlcv(symbol, tf, limit=limit, params={"until": end})
            if not rows:
                break
            out = rows + out
            first = rows[0][0]
            if first <= since_ms or len(rows) < 2:
                break
            end = first - 1
    seen, dedup = set(), []
    for r in out:
        if r[0] not in seen and r[0] <= until_ms:
            seen.add(r[0]); dedup.append(r)
    return sorted(dedup)


def rows_to_df(rows: list) -> pd.DataFrame:
    recs = [{"time": pd.Timestamp(r[0], unit="ms", tz="UTC").isoformat(), "open": r[1], "high": r[2],
             "low": r[3], "close": r[4], "tick_volume": r[5]} for r in rows]
    return from_records(recs) if recs else pd.DataFrame()


class CcxtBroker(Broker):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        ex_cfg = cfg.exchange
        key = ex_cfg.api_key or os.environ.get("EXCHANGE_API_KEY") or os.environ.get("BYBIT_API_KEY", "")
        secret = ex_cfg.api_secret or os.environ.get("EXCHANGE_API_SECRET") or os.environ.get("BYBIT_API_SECRET", "")
        pw = ex_cfg.api_passphrase or os.environ.get("EXCHANGE_API_PASSPHRASE", "")
        self.ex = getattr(ccxt, ex_cfg.id)({
            "apiKey": key, "secret": secret, "password": pw, "enableRateLimit": True,
            "options": {"defaultType": "swap", "adjustForTimeDifference": True},
        })
        if ex_cfg.demo:
            if ex_cfg.id == "bybit":
                self.ex.enable_demo_trading(True)       # api-demo.bybit.com
            else:
                self.ex.set_sandbox_mode(True)          # okx: x-simulated-trading header; others: testnet urls
        self.is_okx = ex_cfg.id == "okx"
        # market data always from the public production endpoints (same prices, no auth needed)
        self.pub = getattr(ccxt, ex_cfg.id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self._markets = None
        self._lev_set = False
        self.account_debug = ""

    # ---- helpers
    def _m(self, symbol: str) -> dict:
        if self._markets is None:
            self._markets = self.pub.load_markets()
            self.ex.load_markets()
        if symbol not in self._markets:
            raise ValueError(f"{symbol} not on {self.ex.id}; try one of: "
                             + ", ".join(s for s in self._markets if s.endswith(":USDT"))[:300])
        return self._markets[symbol]

    def _ensure_leverage(self, symbol: str) -> None:
        if self._lev_set or not self.cfg.exchange.leverage:
            return
        if self.is_okx:
            try:
                self.ex.set_position_mode(False, symbol)  # net (one-way) mode
            except Exception as e:  # noqa: BLE001 — fails harmlessly if already net mode or positions open
                if "59000" not in str(e) and "already" not in str(e).lower():
                    log.warning("set_position_mode: %s", str(e)[:120])
        try:
            params = {"marginMode": "cross"} if self.is_okx else {}
            self.ex.set_leverage(self.cfg.exchange.leverage, symbol, params=params)
        except Exception as e:  # noqa: BLE001 — not fatal: venue default leverage applies, max_leverage caps sizing anyway
            if "not modified" not in str(e).lower() and "110043" not in str(e):
                log.warning("set_leverage failed (continuing): %s", str(e)[:120])
        self._lev_set = True

    # ---- Broker interface
    def health(self) -> dict:
        t = self.ex.fetch_time()
        return {"ok": True, "exchange": self.ex.id, "demo": self.cfg.exchange.demo,
                "server_time": pd.Timestamp(t, unit="ms", tz="UTC").isoformat()}

    def _kraken_accounts_equity(self) -> tuple[float | None, str]:
        """Kraken Futures: fetch_balance(type=cash) returns the full /accounts payload in `info`;
        scan every account for an equity-like field (multi-collateral 'flex' may be absent on demo)."""
        bal = None
        for t in ("cash", "flex"):
            try:
                bal = self.ex.fetch_balance({"type": t})
                break
            except Exception as e:  # noqa: BLE001
                last = e
        if bal is None:
            raw = str(getattr(self.ex, "last_http_response", ""))[:400]
            raise RuntimeError(f"kraken fetch_balance failed: {str(last)[:100]} | raw: {raw}")
        accts = (bal.get("info") or {}).get("accounts") or {}
        summary = {k: (sorted(v.keys())[:10] if isinstance(v, dict) else type(v).__name__) for k, v in accts.items()}
        self.account_debug = f"kraken accounts: {summary}"[:300]
        best = None
        for name, a in accts.items():
            if not isinstance(a, dict):
                continue
            for key in ("portfolioValue", "balanceValue", "marginEquity"):
                v = a.get(key)
                if v is not None:
                    return float(v), f"{name}.{key}"
            aux = a.get("auxiliary") or {}
            for key in ("pv", "af", "usd"):
                if isinstance(aux, dict) and aux.get(key) is not None:
                    return float(aux[key]), f"{name}.auxiliary.{key}"
            bals = a.get("balances") or {}
            if isinstance(bals, dict) and bals.get("usd") is not None:
                best = (float(bals["usd"]), f"{name}.balances.usd")
        if best:
            return best
        raise RuntimeError(f"no equity field in kraken accounts: {summary}")

    def get_account(self) -> Account:
        if self.ex.id == "krakenfutures":
            eq, src = self._kraken_accounts_equity()
            log.info("kraken equity from %s", src)
            return Account(eq, eq, "USD", eq, 0 if self.cfg.exchange.demo else 2)
        bal = self.ex.fetch_balance()
        equity = None
        for getter in (lambda b: b["info"]["data"][0]["totalEq"],                 # okx unified
                       lambda b: b["info"]["result"]["list"][0]["totalEquity"],   # bybit unified
                       lambda b: b["info"]["accounts"]["flex"]["portfolioValue"],  # kraken futures multi-collateral
                       lambda b: b["info"]["accounts"]["flex"]["balanceValue"]):
            try:
                equity = float(getter(bal))
                break
            except Exception:  # noqa: BLE001
                continue
        ccy = "USD" if self.cfg.symbol.endswith(":USD") else "USDT"
        cur = bal.get(ccy, {}) or {}
        wallet = float(cur.get("total") or 0.0)
        if equity is None:
            unreal = sum(float(p.get("unrealizedPnl") or 0.0) for p in self.ex.fetch_positions())
            equity = wallet + unreal
        free = float(cur.get("free") or 0.0)
        return Account(equity, wallet or equity, ccy, free, 0 if self.cfg.exchange.demo else 2)

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        m = self._m(symbol)
        lim = (m.get("limits") or {}).get("amount") or {}
        step = (m.get("precision") or {}).get("amount")
        min_lot = lim.get("min") or step or 0.001
        step = step or min_lot
        return SymbolInfo(float(m.get("contractSize") or 1.0), float(min_lot), float(step),
                          float(lim.get("max") or 1e9), float((m.get("precision") or {}).get("price") or 0.01),
                          None, None)

    def get_quote(self, symbol: str) -> Quote:
        t = self.pub.fetch_ticker(symbol)
        bid, ask = t.get("bid"), t.get("ask")
        if not bid or not ask:  # some venues omit bid/ask in ticker; use order book top
            ob = self.pub.fetch_order_book(symbol, 5)
            bid, ask = ob["bids"][0][0], ob["asks"][0][0]
        ts = pd.Timestamp(t.get("timestamp") or self.pub.milliseconds(), unit="ms", tz="UTC")
        return Quote(float(bid), float(ask), ts.isoformat())

    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        tf = TF.get(timeframe.upper(), timeframe)
        rows = self.pub.fetch_ohlcv(symbol, tf, limit=min(count, 1000))
        if len(rows) < count:  # venue caps the per-call limit -> paginate over a time range
            ms = self.pub.parse_timeframe(tf) * 1000
            now = self.pub.milliseconds()
            df = self.get_bars_range(symbol, timeframe, now - int(count * ms * 1.3), now)
            return df.iloc[-count:]
        recs = [{"time": pd.Timestamp(r[0], unit="ms", tz="UTC").isoformat(), "open": r[1], "high": r[2],
                 "low": r[3], "close": r[4], "tick_volume": r[5]} for r in rows]
        return from_records(recs)

    def get_bars_range(self, symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.DataFrame:
        tf = TF.get(timeframe.upper(), timeframe)
        return rows_to_df(fetch_ohlcv_range(self.pub, symbol, tf, since_ms, until_ms))

    def get_position(self, symbol: str) -> float:
        net = 0.0
        for p in self.ex.fetch_positions([symbol]):
            if p.get("symbol") != symbol:
                continue
            c = float(p.get("contracts") or 0.0)
            if c:
                net += c if p.get("side") == "long" else -c
        return round(net, 10)

    def _order(self, symbol: str, side: str, amount: float, reduce_only: bool) -> dict:
        amt = float(self.ex.amount_to_precision(symbol, amount))
        if amt <= 0:
            return {"skipped": "amount rounds to 0"}
        params = {"reduceOnly": True} if reduce_only else {}
        if self.is_okx:
            params["tdMode"] = "cross"
        o = self.ex.create_order(symbol, "market", side, amt, params=params)
        return {"id": o.get("id"), "side": side, "amount": amt, "reduce_only": reduce_only,
                "avg": o.get("average"), "status": o.get("status")}

    def set_target_position(self, symbol: str, lots: float, comment: str = "") -> dict:
        self._ensure_leverage(symbol)
        cur = self.get_position(symbol)
        step = self.get_symbol_info(symbol).lot_step
        executed = []
        if abs(lots - cur) < step / 2:
            return {"ok": True, "noop": True, "net_lots": cur}
        # flip or flatten: reduce-only close first
        if cur != 0.0 and (lots == 0.0 or (lots > 0) != (cur > 0)):
            executed.append(self._order(symbol, "sell" if cur > 0 else "buy", abs(cur), True))
            cur = 0.0
        delta = lots - cur
        if abs(delta) >= step / 2:
            reduce = cur != 0.0 and abs(lots) < abs(cur)
            executed.append(self._order(symbol, "buy" if delta > 0 else "sell", abs(delta), reduce))
        net = self.get_position(symbol)
        return {"ok": abs(net - lots) < step, "executed": executed, "net_lots": net}

    def close_all(self, symbol: str) -> dict:
        return self.set_target_position(symbol, 0.0, "close_all")

    def funding_annual(self, symbol: str) -> float | None:
        """Current funding rate annualised (positive = longs pay). None if unavailable."""
        try:
            fr = self.pub.fetch_funding_rate(symbol)
            rate = fr.get("fundingRate")
            if rate is None:
                return None
            interval_h = 8.0
            iv = fr.get("interval")
            if isinstance(iv, str) and iv.endswith("h"):
                interval_h = float(iv[:-1])
            return float(rate) * (24.0 / interval_h) * 365.0
        except Exception:  # noqa: BLE001
            return None

    def taker_fee(self, symbol: str) -> float | None:
        m = self._m(symbol)
        return float(m["taker"]) if m.get("taker") is not None else None
