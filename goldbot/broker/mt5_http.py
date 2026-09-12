"""HTTP client for the Flask MT5 bridge (see bridge/mt5_bridge.py for the endpoint contract)."""
from __future__ import annotations

import pandas as pd
import requests

from ..config import BridgeCfg
from ..data import from_records
from .base import Account, Broker, Quote, SymbolInfo


class MT5HttpBroker(Broker):
    def __init__(self, cfg: BridgeCfg):
        self.cfg = cfg
        self.s = requests.Session()

    def _get(self, path: str, **params):
        r = self.s.get(f"{self.cfg.url}{path}", params=params, timeout=self.cfg.timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"bridge {path}: {data['error']}")
        return data

    def _post(self, path: str, payload: dict) -> dict:
        r = self.s.post(f"{self.cfg.url}{path}", json=payload, timeout=self.cfg.timeout * 3)
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            raise RuntimeError(f"bridge {path}: {data['error']}")
        return data

    def health(self) -> dict:
        return self._get("/health")

    def get_account(self) -> Account:
        d = self._get("/account")
        return Account(float(d["equity"]), float(d["balance"]), str(d["currency"]), float(d["margin_free"]),
                       d.get("trade_mode"))

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        d = self._get("/symbol_info", symbol=symbol)
        return SymbolInfo(float(d["contract_size"]), float(d["min_lot"]), float(d["lot_step"]),
                          float(d["max_lot"]), float(d["point"]), d.get("swap_long"), d.get("swap_short"))

    def get_quote(self, symbol: str) -> Quote:
        d = self._get("/tick", symbol=symbol)
        return Quote(float(d["bid"]), float(d["ask"]), str(d["time"]))

    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        return from_records(self._get("/rates", symbol=symbol, timeframe=timeframe, count=count))

    def get_bars_range(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        recs = self._get("/rates_range", symbol=symbol, timeframe=timeframe, start=start, end=end)
        return from_records(recs) if recs else pd.DataFrame()

    def get_position(self, symbol: str) -> float:
        return float(self._get("/position", symbol=symbol, magic=self.cfg.magic)["net_lots"])

    def set_target_position(self, symbol: str, lots: float, comment: str = "") -> dict:
        return self._post("/target_position", {"symbol": symbol, "lots": lots, "comment": comment,
                                               "deviation": self.cfg.deviation_points, "magic": self.cfg.magic})

    def close_all(self, symbol: str) -> dict:
        return self._post("/close_all", {"symbol": symbol, "magic": self.cfg.magic})
