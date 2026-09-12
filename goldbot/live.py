"""Live / paper trading loop.

Bar-close detection is broker-time agnostic: the bridge returns the forming bar as the
last row; when the timestamp of the last *closed* bar changes, the bot acts once.
Sizing, rebalance rule and kill switches are the same objects the backtest uses.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import pandas as pd

from .backtest import should_trade
from .broker.base import Broker
from .config import Config
from .risk import RiskManager
from .sizing import exposure_to_lots, lots_to_exposure
from .strategy import bars_needed, compute_exposure

log = logging.getLogger("goldbot")

JOURNAL_COLS = ["time", "bar_time", "price", "spread", "equity", "signal", "vol_ann", "kelly_mult",
                "target_exposure", "current_lots", "target_lots", "action", "reason", "result"]


class LiveTrader:
    def __init__(self, cfg: Config, broker: Broker, state_path: str, journal_path: str,
                 dry_run: bool = False, mode: str = "live", allow_real: bool = False):
        self.cfg, self.broker, self.dry_run, self.mode, self.allow_real = cfg, broker, dry_run, mode, allow_real
        self.start_equity: float | None = None
        self.state_path, self.journal_path = Path(state_path), Path(journal_path)
        self.last_bar: str | None = None
        self.rm: RiskManager | None = None
        self._load_state()

    # ---- persistence
    def _load_state(self) -> None:
        if self.state_path.exists():
            d = json.loads(self.state_path.read_text())
            self.last_bar = d.get("last_bar")
            self.start_equity = d.get("start_equity")
            if d.get("risk"):
                self.rm = RiskManager.from_dict(self.cfg.risk, d["risk"])

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "last_bar": self.last_bar,
            "start_equity": self.start_equity,
            "risk": self.rm.to_dict() if self.rm else None,
            "mode": self.mode,
        }, indent=1))

    def _journal(self, row: dict) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.journal_path.exists()
        with open(self.journal_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=JOURNAL_COLS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k) for k in JOURNAL_COLS})

    def reset_halt(self) -> None:
        if self.rm:
            self.rm.reset_halt()
            self._save_state()
            log.warning("halt reset by operator")

    # ---- startup checks
    def preflight(self) -> None:
        h = self.broker.health()
        log.info("bridge health: %s", h)
        info = self.broker.get_symbol_info(self.cfg.symbol)
        ct = self.cfg.contract
        if any(abs(a - b) > 1e-12 for a, b in ((info.contract_size, ct.size), (info.min_lot, ct.min_lot), (info.lot_step, ct.lot_step))):
            raise RuntimeError(
                f"config contract ({ct.size},{ct.min_lot},{ct.lot_step}) != broker "
                f"({info.contract_size},{info.min_lot},{info.lot_step}) — fix config.yaml")
        acct = self.broker.get_account()
        log.info("account: equity=%.2f %s, trade_mode=%s, position=%.2f lots", acct.equity, acct.currency,
                 {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(acct.trade_mode, acct.trade_mode),
                 self.broker.get_position(self.cfg.symbol))
        if self.mode == "live" and acct.trade_mode == 2 and not self.allow_real:
            raise RuntimeError("REAL-MONEY account detected. Refusing to trade. "
                               "Pass --allow-real-account only after the walk-forward gates in README are met.")
        if self.start_equity is None:
            self.start_equity = acct.equity
        if self.rm is None or self.rm.state.hwm > 10 * max(self.cfg.equity_cap, 1):
            eq0 = self.cfg.equity_cap if self.cfg.equity_cap > 0 else acct.equity
            self.rm = RiskManager(self.cfg.risk, eq0, pd.Timestamp.now(tz="UTC"))
            self._save_state()
        if self.rm is None:
            self.rm = RiskManager(self.cfg.risk, acct.equity, pd.Timestamp.now(tz="UTC"))
            self._save_state()
        if self.rm.state.halted:
            log.critical("bot is HALTED (%s). Use --reset-halt after reviewing.", self.rm.state.halted_reason)

    # ---- one poll
    def poll(self) -> bool:
        """Returns True if a bar was processed."""
        n = bars_needed(self.cfg)
        bars = self.broker.get_bars(self.cfg.symbol, self.cfg.timeframe, n + 1)
        if len(bars) < n:
            raise RuntimeError(f"bridge returned {len(bars)} bars, need {n}")
        closed = bars.iloc[:-1]                      # drop the forming bar
        bar_time = str(closed.index[-1])
        if bar_time == self.last_bar:
            return False
        self.on_bar_close(closed, bar_time)
        self.last_bar = bar_time
        self._save_state()
        return True

    def on_bar_close(self, bars: pd.DataFrame, bar_time: str) -> None:
        cfg, s, ct = self.cfg, self.cfg.strategy, self.cfg.contract
        feats = compute_exposure(bars, cfg).iloc[-1]
        acct = self.broker.get_account()
        q = self.broker.get_quote(cfg.symbol)
        cur_lots = self.broker.get_position(cfg.symbol)

        equity = acct.equity
        if cfg.equity_cap > 0 and self.start_equity is not None:
            equity = cfg.equity_cap + (acct.equity - self.start_equity)   # virtual account: cap + realised P&L
        may_hold, reason = self.rm.step(equity, bars.index[-1])
        tgt_exp = float(feats["exposure"]) if may_hold else 0.0
        tgt_lots = exposure_to_lots(tgt_exp, equity, q.mid, ct, s.max_leverage)
        cur_exp = lots_to_exposure(cur_lots, equity, q.mid, ct)

        row = {"time": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "bar_time": bar_time,
               "price": round(q.mid, 2), "spread": round(q.spread, 2), "equity": round(equity, 2),
               "signal": round(float(feats["signal"]), 3), "vol_ann": round(float(feats["vol_ann"]), 4),
               "kelly_mult": round(float(feats["kelly_mult"]), 3), "target_exposure": round(tgt_exp, 3),
               "current_lots": cur_lots, "target_lots": tgt_lots, "action": "hold", "reason": reason, "result": ""}

        if should_trade(tgt_lots, cur_lots, tgt_exp, cur_exp, s.rebalance_threshold):
            flatten = tgt_lots == 0.0 or not may_hold
            if not flatten and not self.rm.spread_ok(q.spread):
                row.update(action="skip", reason=f"spread {q.spread:.2f} > max {cfg.risk.max_spread}")
            elif self.dry_run:
                row.update(action="dry_run", result=f"would set {cur_lots} -> {tgt_lots}")
            else:
                res = self.broker.set_target_position(cfg.symbol, tgt_lots, comment=f"goldbot {bar_time[:16]}")
                row.update(action="trade", result=json.dumps(res)[:200])
        log.info("%s", {k: row[k] for k in ("bar_time", "price", "equity", "signal", "target_exposure",
                                             "current_lots", "target_lots", "action", "reason")})
        self._journal(row)
        self._save_state()

    # ---- main loop
    def run_forever(self, once: bool = False) -> None:
        self.preflight()
        while True:
            try:
                processed = self.poll()
                self.rm.record_bridge_ok()
                if once and processed:
                    return
            except KeyboardInterrupt:
                log.info("stopped by operator")
                return
            except Exception as e:  # noqa: BLE001 — anything from the bridge must not kill the loop
                halted = self.rm.record_bridge_failure() if self.rm else False
                log.error("poll failed (%d/%d): %s", self.rm.state.bridge_failures if self.rm else 0,
                          self.cfg.risk.max_bridge_failures, e)
                if halted:
                    log.critical("HALTED after repeated bridge failures — no new orders until --reset-halt")
                self._save_state()
            if once:
                return
            time.sleep(self.cfg.risk.poll_seconds)
