"""Serverless entrypoint (GitHub Actions cron / workflow_dispatch). One action per run, then exit.

  python scripts/run_once.py --action trade        # act on the latest closed bar (default, cron)
  python scripts/run_once.py --action status       # push status to Telegram
  python scripts/run_once.py --action flatten      # close everything + halt until reset
  python scripts/run_once.py --action reset_halt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot import notify  # noqa: E402
from goldbot.config import Config  # noqa: E402
from goldbot.factory import make_broker, paths_for  # noqa: E402
from goldbot.live import LiveTrader  # noqa: E402
from goldbot.status import status_text  # noqa: E402


def last_journal_row(journal: str) -> dict | None:
    p = Path(journal)
    if not p.exists():
        return None
    j = pd.read_csv(p)
    return j.iloc[-1].to_dict() if len(j) else None


def action_trade(cfg: Config, mode: str, allow_real: bool) -> int:
    state, journal = paths_for(cfg, mode)
    trader = LiveTrader(cfg, make_broker(cfg, mode), state, journal, mode=mode, allow_real=allow_real)
    trader.preflight()
    if trader.rm.state.halted:
        notify.send(cfg, f"🛑 Bot ist HALTED ({trader.rm.state.halted_reason}) – keine Aktion. "
                         "Workflow mit action=reset_halt starten, um fortzusetzen.")
        return 0
    processed = trader.poll()
    if not processed:
        logging.info("no new closed bar since %s", trader.last_bar)
        return 0
    r = last_journal_row(journal) or {}
    act = r.get("action")
    if act == "trade":
        notify.send(cfg, f"📈 TRADE {str(r['bar_time'])[:16]}\n{cfg.symbol} {r['price']:.2f}\n"
                         f"{r['current_lots']:+.4f} → {r['target_lots']:+.4f} (Ziel {r['target_exposure']:+.2f}x)\n"
                         f"Equity {r['equity']:,.2f}  Grund: {r['reason'] if isinstance(r.get('reason'), str) else 'signal'}")
    elif act == "skip":
        notify.send(cfg, f"⏸ übersprungen {str(r['bar_time'])[:16]}: {r.get('reason')}")
    if trader.rm.state.halted:
        notify.send(cfg, f"🛑 KILL-SWITCH ausgelöst: {trader.rm.state.halted_reason}. Position glattgestellt, "
                         "Bot pausiert bis action=reset_halt.")
    if datetime.now(timezone.utc).hour < 4:   # the 00:xx UTC run doubles as daily report
        notify.send(cfg, "📊 Tagesbericht\n" + status_text(cfg, mode, n=3))
    return 0


def action_flatten(cfg: Config, mode: str) -> int:
    state, _ = paths_for(cfg, mode)
    br = make_broker(cfg, mode)
    res = br.close_all(cfg.symbol)
    p = Path(state)
    st = json.loads(p.read_text()) if p.exists() else {}
    if st.get("risk"):
        st["risk"].update(halted=True, halted_reason="manual_flatten")
        p.write_text(json.dumps(st, indent=1))
    notify.send(cfg, f"Glattgestellt: net {res.get('net_lots')} ok={res.get('ok')}. Bot pausiert bis reset_halt.")
    return 0


def action_reset(cfg: Config, mode: str) -> int:
    state, _ = paths_for(cfg, mode)
    p = Path(state)
    if p.exists():
        st = json.loads(p.read_text())
        if st.get("risk"):
            st["risk"].update(halted=False, halted_reason="", bridge_failures=0)
            p.write_text(json.dumps(st, indent=1))
    notify.send(cfg, "Halt zurückgesetzt – nächster Cron-Lauf handelt wieder.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--action", choices=["trade", "status", "flatten", "reset_halt"], default="trade")
    ap.add_argument("--mode", choices=["paper", "live"], default=None)
    ap.add_argument("--allow-real-account", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.load(a.config)
    mode = a.mode or cfg.telegram.mode
    Path("state").mkdir(exist_ok=True); Path("logs").mkdir(exist_ok=True)
    try:
        if a.action == "trade":
            rc = action_trade(cfg, mode, a.allow_real_account)
        elif a.action == "status":
            notify.send(cfg, status_text(cfg, mode)); rc = 0
        elif a.action == "flatten":
            rc = action_flatten(cfg, mode)
        else:
            rc = action_reset(cfg, mode)
    except Exception as e:  # noqa: BLE001
        logging.exception("run failed")
        notify.send(cfg, f"⚠️ goldbot Lauf fehlgeschlagen: {type(e).__name__}: {str(e)[:300]}")
        rc = 1
    sys.exit(rc)
