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
    cost_aware: bool = True    # subtract the strategy's own cost drag (turnover x cost) from mu before f* = mu/var
    cost_margin: float = 1.5   # safety multiplier on the estimated cost drag


@dataclass
class RegimeCfg:
    er_window: int = 0         # Kaufman efficiency ratio window (bars); 0 = off
    er_min: float = 0.0        # trade only when ER >= er_min (trending); below = range -> flat
    vol_pct_window: int = 0    # rolling window for the realised-vol percentile; 0 = off
    vol_pct_max: float = 1.0   # above this percentile (e.g. 0.9) exposure is scaled by high_vol_scale
    high_vol_scale: float = 0.5


@dataclass
class StrategyCfg:
    lookbacks: list[int] = field(default_factory=lambda: [30, 90, 180])
    vol_span: int = 60
    target_vol: float = 0.12   # annualised
    max_leverage: float = 3.0  # notional / equity
    rebalance_threshold: float = 0.15  # only trade if |target_exp - current_exp| >= this
    vol_floor: float = 0.05
    entry_min_signal: float = 0.0      # |signal| needed to OPEN (1.0 = all lookbacks agree); 0 = off
    exit_min_signal: float = 0.0       # position is kept while |signal| >= this (hysteresis); 0 = until sign flips
    z_min: float = 0.0                 # mean |z| (vol-normalised momentum) needed to open; 0 = off
    min_hold_bars: int = 0             # no sign flip / re-entry within this many bars of the last entry
    direction: str = "both"            # both | long | short (diagnostics: where does the edge live?)
    confidence: bool = False           # scale exposure by signal strength tiers instead of flat 1x
    confidence_tiers: list[list[float]] = field(default_factory=lambda: [[0.5, 0.25], [1.0, 0.5], [1.5, 1.0], [2.5, 1.5]])
    # [[z_threshold, multiplier], ...] ascending: mean |z| below the first threshold -> 0x; above the last -> last multiplier
    regime: RegimeCfg = field(default_factory=RegimeCfg)
    kelly: KellyCfg = field(default_factory=KellyCfg)


@dataclass
class CostCfg:
    spread: float = 0.30            # USD/oz, used when data has no 'spread' column
    slippage: float = 0.05          # USD/oz per side, on top of half-spread
    swap_long_annual: float = -0.04 # annualised financing on notional (negative = you pay)
    swap_short_annual: float = 0.01
    commission_per_lot: float = 0.0 # USD per lot per side
    fee_pct: float = 0.0            # exchange taker fee per side, fraction of notional (0.00055 = 0.055%)
    slippage_pct: float = 0.0       # if > 0, per-side slippage as fraction of price (overrides absolute `slippage` per symbol)


@dataclass
class RiskCfg:
    daily_loss_limit: float = 0.03   # flatten + no new trades until next day
    max_drawdown_halt: float = 0.20  # flatten + permanent halt (manual reset)
    max_spread: float = 0.80         # absolute (price units); skip new entries above this
    max_spread_pct: float = 0.0      # if > 0, relative spread guard (fraction of price) — use for multi-symbol
    max_bridge_failures: int = 3     # consecutive failures -> halt
    poll_seconds: int = 30


@dataclass
class BridgeCfg:
    url: str = "http://127.0.0.1:5000"
    timeout: float = 10.0
    deviation_points: int = 30
    magic: int = 20260912


@dataclass
class SymbolCfg:
    weight: float = 1.0                       # share of the (virtual) account allocated to this symbol
    history_sources: list[str] = field(default_factory=list)  # "exchange:SYMBOL" public proxies for backtest history


@dataclass
class ExchangeCfg:
    id: str = "bybit"          # ccxt exchange id
    demo: bool = True          # Bybit demo trading (api-demo.bybit.com); False = real money
    leverage: int = 3
    api_key: str = ""          # or env EXCHANGE_API_KEY
    api_secret: str = ""       # or env EXCHANGE_API_SECRET
    api_passphrase: str = ""   # OKX/KuCoin/Bitget need one; env EXCHANGE_API_PASSPHRASE
    history_symbol: str = ""   # optional longer-lived proxy on the same exchange (e.g. PAXG/USDT spot)
    history_sources: list[str] = field(default_factory=list)  # "exchange:SYMBOL" public proxies; longest history wins


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
    equity_cap: float = 0.0        # >0: size/risk as if the account were this big (demo accounts hold 100k)
    contract: ContractCfg = field(default_factory=ContractCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    costs: CostCfg = field(default_factory=CostCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    bridge: BridgeCfg = field(default_factory=BridgeCfg)
    telegram: TelegramCfg = field(default_factory=TelegramCfg)
    exchange: ExchangeCfg = field(default_factory=ExchangeCfg)
    symbols: dict[str, SymbolCfg] = field(default_factory=dict)   # portfolio; empty => just `symbol`
    paths: dict[str, str] = field(default_factory=dict)

    @property
    def bars_per_year(self) -> int:
        return self.bars_per_day * self.trading_days_per_year

    def portfolio(self) -> dict[str, "SymbolCfg"]:
        """Ordered symbol -> SymbolCfg. Falls back to the single `symbol` with weight 1."""
        if self.symbols:
            return dict(self.symbols)
        return {self.symbol: SymbolCfg(1.0, list(self.exchange.history_sources))}

    def with_costs(self, **overrides: Any) -> "Config":
        d = asdict(self)
        d["costs"].update(overrides)
        return Config.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = dict(d)
        strat = dict(d.pop("strategy", {}) or {})
        kelly = KellyCfg(**(strat.pop("kelly", {}) or {}))
        regime = RegimeCfg(**(strat.pop("regime", {}) or {}))
        return cls(
            contract=ContractCfg(**(d.pop("contract", {}) or {})),
            strategy=StrategyCfg(kelly=kelly, regime=regime, **strat),
            costs=CostCfg(**(d.pop("costs", {}) or {})),
            risk=RiskCfg(**(d.pop("risk", {}) or {})),
            bridge=BridgeCfg(**(d.pop("bridge", {}) or {})),
            telegram=TelegramCfg(**(d.pop("telegram", {}) or {})),
            exchange=ExchangeCfg(**(d.pop("exchange", {}) or {})),
            symbols={k: SymbolCfg(**(v or {})) for k, v in (d.pop("symbols", {}) or {}).items()},
            **d,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def with_strategy(self, **overrides: Any) -> "Config":
        """Copy with strategy parameters replaced (used by the walk-forward / CV grids).
        Nested dicts (`regime`, `kelly`) are merged, not replaced."""
        d = asdict(self)
        for key, val in overrides.items():
            if isinstance(val, dict) and isinstance(d["strategy"].get(key), dict):
                d["strategy"][key].update(val)
            else:
                d["strategy"][key] = val
        return Config.from_dict(d)
