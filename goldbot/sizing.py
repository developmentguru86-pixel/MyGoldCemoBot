"""Exposure (notional/equity) <-> lots. Lot granularity is enforced here, nowhere else."""
from __future__ import annotations

import math

from .config import ContractCfg


def round_lots(lots: float, step: float) -> float:
    return round(math.copysign(round(abs(lots) / step) * step, lots), 8)


def exposure_to_lots(exposure: float, equity: float, price: float,
                     contract: ContractCfg, max_leverage: float) -> float:
    if equity <= 0 or price <= 0 or not math.isfinite(exposure):
        return 0.0
    exposure = max(-max_leverage, min(max_leverage, exposure))
    lots = round_lots(exposure * equity / (contract.size * price), contract.lot_step)
    if abs(lots) < contract.min_lot - 1e-12:
        return 0.0
    return math.copysign(min(abs(lots), contract.max_lot), lots)


def lots_to_exposure(lots: float, equity: float, price: float, contract: ContractCfg) -> float:
    return lots * contract.size * price / equity if equity > 0 else 0.0
