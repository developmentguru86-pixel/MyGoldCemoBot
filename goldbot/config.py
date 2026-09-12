"""Configuration: YAML -> typed dataclasses. One Config object drives backtest, paper and live."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ContractCfg:
    size: float = 100.0        # oz per lot (XAUUSD standard)
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 5.0


@dataclass
class KellyCfg:
    enabled: bool = True
    fraction: float = 0.5      # fractional Kelly (0.5 = half Kelly)
    window: int = 500          # bars used to estimate mu/sigma^2 of the unthrottled strategy
    min_mult: float = 0.25     # never throttle below this multiplier


@dataclass
class StrategyCfg:
    lookbacks: list[int] = field(default_factory=lambda: [30, 90, 180])
    vol_span: int = 60
    target_vol: float = 0.12   # annualised
    max_leverage: float = 3.0  # notional / equity
    rebalance_threshold: float = 0.15  # only trade if |target_exp - current_exp| >= this
    vol_floor: float = 0.05
    kelly: KellyCfg = field(default_factory=KellyCfg)


@dataclass
class CostCfg:
    spread: float = 0.30            # USD/oz, used when data has no 'spread' column
    slippage: float = 0.05          # USD/oz per side, on top of half-spread
    swap_long_annual: float = -0.04 # annualised financing on notional (negative = you pay)
    swap_short_annual: float = 0.01
    commission_per_lot: float = 0.0 # USD per lot per side
    fee_pct: float = 0.0            # exchange taker fee per side, fraction of notional (0.00055 = 0.055%)


@dataclass
class RiskCfg:
    daily_loss_limit: float = 0.03   # flatten + no new trades until next day
    max_drawdown_halt: float = 0.20  # flatten + permanent halt (manual reset)
    max_spread: float = 0.80         # USD/oz; skip new entries above this
    max_bridge_failures: int = 3     # consecutive failures -> halt
    poll_seconds: int = 30


@dataclass
class BridgeCfg:
    url: str = "http://127.0.0.1:5000"
    timeout: float = 10.0
    deviation_points: int = 30
    magic: int = 20260912


@dataclass
class ExchangeCfg:
    id: str = "bybit"          # ccxt exchange id
    demo: bool = True          # Bybit demo trading (api-demo.bybit.com); False = real money
    leverage: int = 3
    api_key: str = ""          # or env BYBIT_API_KEY / EXCHANGE_API_KEY
    api_secret: str = ""       # or env BYBIT_API_SECRET / EXCHANGE_API_SECRET


@dataclass
class TelegramCfg:
    token: str = ""            # from @BotFather
    chat_id: int = 0           # your own chat id; 0 = the supervisor tells you yours on first message
    mode: str = "live"         # bot mode the supervisor starts (demo account => harmless)
    autostart: bool = True
    daily_summary_hour: int = 21


@dataclass
class Config:
    venue: str = "mt5"           # "mt5" (Flask bridge) or "exchange" (ccxt)
    symbol: str = "XAUUSD"
    timeframe: str = "H4"
    bars_per_day: int = 6
    trading_days_per_year: int = 260
    starting_equity: float = 5000.0
    contract: ContractCfg = field(default_factory=ContractCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    costs: CostCfg = field(default_factory=CostCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    bridge: BridgeCfg = field(default_factory=BridgeCfg)
    telegram: TelegramCfg = field(default_factory=TelegramCfg)
    exchange: ExchangeCfg = field(default_factory=ExchangeCfg)
    paths: dict[str, str] = field(default_factory=dict)

    @property
    def bars_per_year(self) -> int:
        return self.bars_per_day * self.trading_days_per_year

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = dict(d)
        strat = dict(d.pop("strategy", {}) or {})
        kelly = KellyCfg(**(strat.pop("kelly", {}) or {}))
        return cls(
            contract=ContractCfg(**(d.pop("contract", {}) or {})),
            strategy=StrategyCfg(kelly=kelly, **strat),
            costs=CostCfg(**(d.pop("costs", {}) or {})),
            risk=RiskCfg(**(d.pop("risk", {}) or {})),
            bridge=BridgeCfg(**(d.pop("bridge", {}) or {})),
            telegram=TelegramCfg(**(d.pop("telegram", {}) or {})),
            exchange=ExchangeCfg(**(d.pop("exchange", {}) or {})),
            **d,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def with_strategy(self, **overrides: Any) -> "Config":
        """Copy with strategy parameters replaced (used by the walk-forward grid)."""
        d = asdict(self)
        d["strategy"].update(overrides)
        return Config.from_dict(d)
