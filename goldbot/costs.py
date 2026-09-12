"""Transaction and financing costs. All results in account currency (positive = cost)."""
from __future__ import annotations


def trade_cost(delta_lots: float, spread: float, slippage: float, contract_size: float,
               commission_per_lot: float = 0.0, price: float = 0.0, fee_pct: float = 0.0) -> float:
    """Cost of changing position by delta_lots: half-spread + slippage per unit, fixed commission per lot,
    and a percentage fee on notional (exchange taker fee)."""
    units = abs(delta_lots) * contract_size
    return units * (spread / 2.0 + slippage) + abs(delta_lots) * commission_per_lot + units * price * fee_pct


def swap_cost(lots: float, price: float, contract_size: float,
              swap_long_annual: float, swap_short_annual: float, bars_per_year: int) -> float:
    """Per-bar financing on notional. Rates are annualised; negative rate = you pay."""
    if lots == 0.0:
        return 0.0
    rate = swap_long_annual if lots > 0 else swap_short_annual
    notional = abs(lots) * contract_size * price
    return -notional * rate / bars_per_year
