"""Kill switches. The same RiskManager runs inside the backtest and the live loop,
so what you see in the backtest is what the live bot will do."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .config import RiskCfg


@dataclass
class RiskState:
    hwm: float
    day: str
    day_start_equity: float
    halted: bool = False
    halted_reason: str = ""
    day_halted: bool = False
    bridge_failures: int = 0


class RiskManager:
    def __init__(self, cfg: RiskCfg, equity: float, ts: pd.Timestamp, state: RiskState | None = None):
        self.cfg = cfg
        self.state = state or RiskState(hwm=equity, day=ts.date().isoformat(), day_start_equity=equity)

    def step(self, equity: float, ts: pd.Timestamp) -> tuple[bool, str]:
        """Mark-to-market update. Returns (may_hold_position, reason). False => target must be flat."""
        st, cfg = self.state, self.cfg
        day = ts.date().isoformat()
        if day != st.day:
            st.day, st.day_start_equity, st.day_halted = day, equity, False
        st.hwm = max(st.hwm, equity)
        if st.halted:
            return False, st.halted_reason
        if cfg.max_drawdown_halt > 0 and equity <= st.hwm * (1.0 - cfg.max_drawdown_halt):
            st.halted, st.halted_reason = True, "max_drawdown"
            return False, st.halted_reason
        if cfg.daily_loss_limit > 0 and equity <= st.day_start_equity * (1.0 - cfg.daily_loss_limit):
            st.day_halted = True
        if st.day_halted:
            return False, "daily_loss"
        return True, ""

    def spread_ok(self, spread: float, price: float | None = None) -> bool:
        if self.cfg.max_spread_pct > 0 and price:
            return spread <= self.cfg.max_spread_pct * price
        return spread <= self.cfg.max_spread

    def record_bridge_failure(self) -> bool:
        st = self.state
        st.bridge_failures += 1
        if st.bridge_failures >= self.cfg.max_bridge_failures and not st.halted:
            st.halted, st.halted_reason = True, "bridge_failures"
        return st.halted

    def record_bridge_ok(self) -> None:
        self.state.bridge_failures = 0

    def reset_halt(self) -> None:
        self.state.halted, self.state.halted_reason, self.state.bridge_failures = False, "", 0

    def to_dict(self) -> dict:
        return asdict(self.state)

    @classmethod
    def from_dict(cls, cfg: RiskCfg, d: dict) -> "RiskManager":
        st = RiskState(**d)
        return cls(cfg, st.hwm, pd.Timestamp(st.day, tz="UTC"), state=st)
