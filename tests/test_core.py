import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot.backtest import should_trade, simulate  # noqa: E402
from goldbot.config import Config, ContractCfg  # noqa: E402
from goldbot.costs import swap_cost, trade_cost  # noqa: E402
from goldbot.risk import RiskManager  # noqa: E402
from goldbot.sizing import exposure_to_lots  # noqa: E402
from goldbot.strategy import compute_exposure  # noqa: E402
from scripts.make_synthetic import make  # noqa: E402


def _df(years=2.0, seed=1):
    return make(years, seed)


def test_no_lookahead():
    cfg = Config()
    df = _df()
    a = compute_exposure(df, cfg)["exposure"]
    df2 = df.copy()
    df2.iloc[-100:, df2.columns.get_loc("close")] *= 1.3   # rewrite the future
    b = compute_exposure(df2, cfg)["exposure"]
    pd.testing.assert_series_equal(a.iloc[:-100], b.iloc[:-100])


def test_lot_rounding_and_min_lot():
    ct = ContractCfg(size=100, min_lot=0.01, lot_step=0.01, max_lot=5)
    assert exposure_to_lots(0.8, 5800, 3800, ct, 3.0) == 0.01     # 4640/380000 = 0.0122 -> 0.01
    assert exposure_to_lots(0.3, 5800, 3800, ct, 3.0) == 0.0      # 0.0046 -> below min lot
    assert exposure_to_lots(-2.0, 5800, 3800, ct, 3.0) == -0.03
    assert exposure_to_lots(10.0, 5800, 3800, ct, 3.0) == 0.05    # capped at 3x leverage
    assert exposure_to_lots(1.0, 0.0, 3800, ct, 3.0) == 0.0


def test_costs_signs():
    assert trade_cost(0.01, 0.30, 0.05, 100) == 1 * (0.15 + 0.05)
    assert swap_cost(0.01, 3800, 100, -0.04, 0.01, 1560) > 0      # long pays
    assert swap_cost(-0.01, 3800, 100, -0.04, 0.01, 1560) < 0     # short earns
    assert swap_cost(0.0, 3800, 100, -0.04, 0.01, 1560) == 0.0


def test_flat_strategy_keeps_equity_constant():
    cfg = Config()
    df = _df(1.0)
    res = simulate(df, pd.Series(0.0, index=df.index), cfg)
    assert res.equity.nunique() == 1 and res.metrics["trades"] == 0


def test_costs_reduce_equity_vs_frictionless():
    cfg = Config()
    df = _df(2.0)
    expo = compute_exposure(df, cfg)["exposure"]
    with_costs = simulate(df, expo, cfg)
    free = Config.from_dict({"costs": {"spread": 0, "slippage": 0, "swap_long_annual": 0, "swap_short_annual": 0}})
    free_res = simulate(df.drop(columns=["spread"]), expo, free)
    assert with_costs.total_cost > 0
    assert with_costs.equity.iloc[-1] < free_res.equity.iloc[-1]


def test_daily_loss_limit_flattens():
    cfg = Config.from_dict({"risk": {"daily_loss_limit": 0.02, "max_drawdown_halt": 0.0}})
    idx = pd.date_range("2024-01-01", periods=12, freq="4h", tz="UTC")
    close = np.array([3500] * 3 + [3300] * 9, dtype=float)       # -5.7% inside one day
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)
    res = simulate(df, pd.Series(3.0, index=idx), cfg)
    assert (res.trades["reason"] == "daily_loss").any()
    assert res.lots.iloc[3:6].eq(0).all()


def test_max_dd_halt_is_permanent():
    rm = RiskManager(Config().risk, 1000.0, pd.Timestamp("2024-01-01", tz="UTC"))
    assert rm.step(1000.0, pd.Timestamp("2024-01-01", tz="UTC")) == (True, "")
    assert rm.step(790.0, pd.Timestamp("2024-01-02", tz="UTC")) == (False, "max_drawdown")
    assert rm.step(1200.0, pd.Timestamp("2024-03-01", tz="UTC")) == (False, "max_drawdown")


def test_rebalance_rule():
    assert not should_trade(0.01, 0.01, 0.9, 0.7, 0.15)
    assert should_trade(0.02, 0.01, 0.9, 0.7, 0.15)
    assert not should_trade(0.02, 0.01, 0.75, 0.7, 0.15)
    assert should_trade(-0.01, 0.01, -0.5, 0.5, 0.15)
    assert should_trade(0.0, 0.01, 0.0, 0.7, 0.15)
