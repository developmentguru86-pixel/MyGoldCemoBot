"""Pick the broker from config.venue; paper mode wraps it."""
from __future__ import annotations

from .broker import MT5HttpBroker, PaperBroker
from .broker.base import Broker
from .config import Config


def data_broker(cfg: Config) -> Broker:
    if cfg.venue == "exchange":
        from .broker.ccxt_broker import CcxtBroker
        return CcxtBroker(cfg)
    return MT5HttpBroker(cfg.bridge)


def make_broker(cfg: Config, mode: str) -> Broker:
    b = data_broker(cfg)
    if mode == "paper":
        return PaperBroker(cfg, b, cfg.paths.get("paper_account", "state/paper_account.json"))
    return b


def paths_for(cfg: Config, mode: str) -> tuple[str, str]:
    if mode == "paper":
        return (cfg.paths.get("state_paper", "state/paper_state.json"), cfg.paths.get("journal_paper", "logs/journal_paper.csv"))
    return (cfg.paths.get("state_live", "state/live_state.json"), cfg.paths.get("journal_live", "logs/journal_live.csv"))
