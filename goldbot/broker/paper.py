"""Paper broker: real market data from a data broker (the MT5 bridge), simulated fills.
Fills at ask + slippage (buy) / bid - slippage (sell); financing charged once per day."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import Config
from ..costs import swap_cost
from .base import Account, Broker, Quote, SymbolInfo


class PaperBroker(Broker):
    def __init__(self, cfg: Config, data: Broker, state_path: str):
        self.cfg, self.data, self.path = cfg, data, Path(state_path)
        self.st = {"cash": cfg.starting_equity, "lots": 0.0, "avg_price": 0.0, "day": None,
                   "realized": 0.0, "fills": []}
        if self.path.exists():
            self.st.update(json.loads(self.path.read_text()))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.st, indent=1, default=str))

    # ---- market data passes through to the real bridge
    def health(self) -> dict:
        return self.data.health()

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return self.data.get_symbol_info(symbol)

    def get_quote(self, symbol: str) -> Quote:
        return self.data.get_quote(symbol)

    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        return self.data.get_bars(symbol, timeframe, count)

    # ---- simulated account
    def _apply_daily_swap(self, q: Quote) -> None:
        day = q.time[:10]
        if self.st["day"] is not None and day != self.st["day"] and self.st["lots"] != 0.0:
            c, ct = self.cfg.costs, self.cfg.contract
            self.st["cash"] -= swap_cost(self.st["lots"], q.mid, ct.size, c.swap_long_annual,
                                         c.swap_short_annual, self.cfg.trading_days_per_year)
        self.st["day"] = day

    def get_account(self) -> Account:
        q = self.get_quote(self.cfg.symbol)
        self._apply_daily_swap(q)
        lots = self.st["lots"]
        mark = q.bid if lots > 0 else q.ask
        unreal = lots * self.cfg.contract.size * (mark - self.st["avg_price"]) if lots else 0.0
        eq = self.st["cash"] + unreal
        self._save()
        return Account(eq, self.st["cash"], "PAPER", eq, 0)

    def get_position(self, symbol: str) -> float:
        return float(self.st["lots"])

    def set_target_position(self, symbol: str, lots: float, comment: str = "") -> dict:
        q = self.get_quote(symbol)
        cur = self.st["lots"]
        delta = lots - cur
        if abs(delta) < 1e-9:
            return {"ok": True, "noop": True, "net_lots": cur}
        ct, c = self.cfg.contract, self.cfg.costs
        px = (q.ask + c.slippage) if delta > 0 else (q.bid - c.slippage)
        # 1) close the part that is reduced / flipped, realising P&L
        if cur != 0.0 and (lots == 0.0 or (lots > 0) != (cur > 0) or abs(lots) < abs(cur)):
            closed = cur if (lots == 0.0 or (lots > 0) != (cur > 0)) else cur - lots
            pnl = closed * ct.size * (px - self.st["avg_price"])
            self.st["cash"] += pnl - abs(closed) * c.commission_per_lot - abs(closed) * ct.size * px * c.fee_pct
            self.st["realized"] += pnl
            cur -= closed
        # 2) open the remainder at the fill price
        opened = lots - cur
        if abs(opened) > 1e-9:
            tot = abs(cur) + abs(opened)
            self.st["avg_price"] = (abs(cur) * self.st["avg_price"] + abs(opened) * px) / tot
            self.st["cash"] -= abs(opened) * c.commission_per_lot + abs(opened) * ct.size * px * c.fee_pct
        self.st["lots"] = lots
        if lots == 0.0:
            self.st["avg_price"] = 0.0
        self.st["fills"].append({"time": q.time, "delta": delta, "price": px, "lots_after": lots, "comment": comment})
        self.st["fills"] = self.st["fills"][-500:]
        self._save()
        return {"ok": True, "fill_price": px, "fill_avg": px, "net_lots": lots}

    def close_all(self, symbol: str) -> dict:
        return self.set_target_position(symbol, 0.0, "close_all")

    def get_entry_price(self, symbol: str) -> float | None:
        return self.st["avg_price"] or None
