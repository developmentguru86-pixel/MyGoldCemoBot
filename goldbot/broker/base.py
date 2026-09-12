"""Broker interface. Live and paper brokers implement exactly this; the live loop knows nothing else."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class MarketClosed(Exception):
    """Venue refuses orders because the underlying market is closed (e.g. gold perps on weekends)."""


@dataclass
class Quote:
    bid: float
    ask: float
    time: str

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class Account:
    equity: float
    balance: float
    currency: str
    margin_free: float
    trade_mode: int | None = None   # MT5: 0 demo, 1 contest, 2 real


@dataclass
class SymbolInfo:
    contract_size: float
    min_lot: float
    lot_step: float
    max_lot: float
    point: float
    swap_long: float | None = None   # raw broker swap values, informational
    swap_short: float | None = None


class Broker(ABC):
    @abstractmethod
    def health(self) -> dict: ...

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> SymbolInfo: ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Most recent `count` bars, oldest first. The LAST row is the still-forming bar."""

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """Net lots (positive long, negative short) held by this bot."""

    @abstractmethod
    def set_target_position(self, symbol: str, lots: float, comment: str = "") -> dict:
        """Move the net position to `lots` (idempotent)."""

    @abstractmethod
    def close_all(self, symbol: str) -> dict: ...
