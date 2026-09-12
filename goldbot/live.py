"""Live / paper trading loop — portfolio of symbols on one account.

Bar-close detection per symbol (last row from the broker is the forming bar; when the
timestamp of the last *closed* bar changes, the symbol is evaluated once). Capital is
split by config weights; the RiskManager runs on the whole (virtual) account.
Sizing uses the venue's own contract data (fetched at preflight), not config guesses.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import pandas as pd

from .backtest import should_trade
from .broker.base import Broker, MarketClosed, SymbolInfo
from .config import Config, ContractCfg
from .risk import RiskManager
from .sizing import exposure_to_lots, lots_to_exposure
from .strategy import bars_needed, compute_exposure

log = logging.getLogger("goldbot")

JOURNAL_COLS = ["time", "symbol", "bar_time", "price", "spread", "equity", "signal", "vol_ann", "kelly_mult",
                "target_exposure", "current_lots", "target_lots", "action", "reason", "result"]


def _slug(symbol: str) -> str:
    return symbol.split("/")[0].lower()


class LiveTrader:
    def __init__(self, cfg: Config, broker: Broker, state_path: str, journal_path: str,
                 dry_run: bool = False, mode: str = "live", allow_real: bool = False):
        self.cfg, self.broker, self.dry_run, self.mode, self.allow_real = cfg, broker, dry_run, mode, allow_real
        self.state_path, self.journal_path = Path(state_path), Path(journal_path)
        self.symbols = cfg.portfolio()
        self.last_bar: dict[str, str] = {}
        self.rm: RiskManager | None = None
        self.start_equity: float | None = None
        self.last_actual_equity: float | None = None
        self.info: dict[str, SymbolInfo] = {}
        self.last_rows: list[dict] = []
        self._load_state()

    # ---- persistence
    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        d = json.loads(self.state_path.read_text())
        lb = d.get("last_bar")
        self.last_bar = dict(lb) if isinstance(lb, dict) else ({self.cfg.symbol: lb} if lb else {})
        self.start_equity = d.get("start_equity")
        self.last_actual_equity = d.get("last_actual_equity")
        if d.get("risk"):
            self.rm = RiskManager.from_dict(self.cfg.risk, d["risk"])

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "last_bar": self.last_bar,
            "start_equity": self.start_equity,
            "last_actual_equity": self.last_actual_equity,
            "risk": self.rm.to_dict() if self.rm else None,
            "mode": self.mode,
            "symbols": list(self.symbols),
        }, indent=1))

    def _journal(self, row: dict) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        if self.journal_path.exists():
            with open(self.journal_path) as f:
                header = f.readline().strip().split(",")
            if header != JOURNAL_COLS:  # schema changed: archive the old journal
                self.journal_path.rename(self.journal_path.with_suffix(".v1.csv"))
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

    # ---- account view
    def _virtual_equity(self, actual: float) -> float:
        cap = self.cfg.equity_cap
        if cap > 0 and self.start_equity is not None:
            return cap + (actual - self.start_equity)
        return actual

    # ---- startup checks
    def preflight(self) -> None:
        log.info("broker health: %s", self.broker.health())
        for sym in self.symbols:
            self.info[sym] = self.broker.get_symbol_info(sym)
            log.info("%s: contract %s", sym, self.info[sym])
        acct = self.broker.get_account()
        log.info("account: equity=%.2f %s, trade_mode=%s", acct.equity, acct.currency,
                 {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(acct.trade_mode, acct.trade_mode))
        if self.mode == "live" and acct.trade_mode == 2 and not self.allow_real:
            raise RuntimeError("REAL-MONEY account detected. Refusing to trade. "
                               "Pass --allow-real-account only after the walk-forward gates in README are met.")
        if acct.equity < 1.0:
            raise RuntimeError(f"account equity is {acct.equity:.2f} {acct.currency} — unfunded. "
                               "Testnet: claim test USDT in the Assets page, then rerun.")
        if self.start_equity is None or self.start_equity <= 0:
            self.start_equity = acct.equity
        if self.rm is None or self.rm.state.hwm > 10 * max(self.cfg.equity_cap, 1):
            eq0 = self.cfg.equity_cap if self.cfg.equity_cap > 0 else acct.equity
            self.rm = RiskManager(self.cfg.risk, eq0, pd.Timestamp.now(tz="UTC"))
        self._save_state()
        if self.rm.state.halted:
            log.critical("bot is HALTED (%s). Use --reset-halt after reviewing.", self.rm.state.halted_reason)

    # ---- one poll over all symbols
    def poll(self) -> list[str]:
        """Evaluate every symbol whose last closed bar is new. Returns the symbols processed."""
        self.last_rows = []
        processed = []
        n = bars_needed(self.cfg)
        for sym in self.symbols:
            bars = self.broker.get_bars(sym, self.cfg.timeframe, n + 1)
            if len(bars) < n:
                raise RuntimeError(f"{sym}: broker returned {len(bars)} bars, need {n}")
            closed = bars.iloc[:-1]
            age_h = (pd.Timestamp.now(tz="UTC") - closed.index[-1]).total_seconds() / 3600
            if age_h > 60:  # market-closed weekends are ~52h; anything older means the feed is stale
                raise RuntimeError(f"{sym}: last closed bar {closed.index[-1]} is {age_h:.0f}h old — stale data, not trading")
            bar_time = str(closed.index[-1])
            if bar_time == self.last_bar.get(sym):
                continue
            self.on_bar_close(sym, closed, bar_time)
            self.last_bar[sym] = bar_time
            self._save_state()
            processed.append(sym)
        return processed

    def on_bar_close(self, sym: str, bars: pd.DataFrame, bar_time: str) -> None:
        cfg, s = self.cfg, self.cfg.strategy
        info = self.info.get(sym) or self.broker.get_symbol_info(sym)
        ct = ContractCfg(info.contract_size, info.min_lot, info.lot_step, info.max_lot)
        weight = self.symbols[sym].weight

        feats = compute_exposure(bars, cfg).iloc[-1]
        acct = self.broker.get_account()
        q = self.broker.get_quote(sym)
        cur_lots = self.broker.get_position(sym)

        # demo resets: actual equity jumps while flat -> re-baseline the virtual account
        if self.start_equity is not None and self.start_equity <= 0 and acct.equity > 0:
            self.start_equity = acct.equity
        if (self.last_actual_equity is not None and cur_lots == 0.0
                and abs(acct.equity - self.last_actual_equity) > 0.5 * max(self.last_actual_equity, 1.0)):
            log.warning("equity jumped %.0f -> %.0f while flat: demo reset? re-baselining", self.last_actual_equity, acct.equity)
            self.start_equity = acct.equity
            self.rm = RiskManager(cfg.risk, cfg.equity_cap or acct.equity, bars.index[-1])
        self.last_actual_equity = acct.equity

        equity = self._virtual_equity(acct.equity)
        may_hold, reason = self.rm.step(equity, bars.index[-1])
        sleeve = equity * weight
        tgt_exp = float(feats["exposure"]) if may_hold else 0.0
        tgt_lots = exposure_to_lots(tgt_exp, sleeve, q.mid, ct, s.max_leverage)
        cur_exp = lots_to_exposure(cur_lots, sleeve, q.mid, ct)

        row = {"time": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "symbol": sym, "bar_time": bar_time,
               "price": round(q.mid, 2), "spread": round(q.spread, 4), "equity": round(equity, 2),
               "signal": round(float(feats["signal"]), 3), "vol_ann": round(float(feats["vol_ann"]), 4),
               "kelly_mult": round(float(feats["kelly_mult"]), 3), "target_exposure": round(tgt_exp, 3),
               "current_lots": cur_lots, "target_lots": tgt_lots, "action": "hold", "reason": reason, "result": ""}

        if should_trade(tgt_lots, cur_lots, tgt_exp, cur_exp, s.rebalance_threshold):
            flatten = tgt_lots == 0.0 or not may_hold
            if not flatten and not self.rm.spread_ok(q.spread, q.mid):
                row.update(action="skip", reason=f"spread {q.spread:.4f} too wide")
            elif self.dry_run:
                row.update(action="dry_run", result=f"would set {cur_lots} -> {tgt_lots}")
            else:
                try:
                    res = self.broker.set_target_position(sym, tgt_lots, comment=f"goldbot {bar_time[:16]}")
                    row.update(action="trade", result=json.dumps(res)[:200])
                except MarketClosed as e:
                    row.update(action="skip", reason="market_closed", result=str(e)[:120])
                except Exception as e:  # noqa: BLE001
                    row.update(action="error", result=f"{type(e).__name__}: {str(e)[:150]}")
                    self._journal(row)
                    raise
        log.info("%s", {k: row[k] for k in ("symbol", "bar_time", "price", "equity", "signal", "target_exposure",
                                             "current_lots", "target_lots", "action", "reason")})
        self._journal(row)
        self.last_rows.append(row)
        self._save_state()

    # ---- main loop (Weg A: long-running process)
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
            except Exception as e:  # noqa: BLE001 — anything from the broker must not kill the loop
                halted = self.rm.record_bridge_failure() if self.rm else False
                log.error("poll failed (%d/%d): %s", self.rm.state.bridge_failures if self.rm else 0,
                          self.cfg.risk.max_bridge_failures, e)
                if halted:
                    log.critical("HALTED after repeated broker failures — no new orders until --reset-halt")
                self._save_state()
            if once:
                return
            time.sleep(self.cfg.risk.poll_seconds)
