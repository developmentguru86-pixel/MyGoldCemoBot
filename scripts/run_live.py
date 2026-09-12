"""Run the bot against the MT5 bridge.

  python scripts/run_live.py --mode paper            # real data, simulated fills
  python scripts/run_live.py --mode live --dry-run   # real account, logs intended orders only
  python scripts/run_live.py --mode live             # sends orders
  python scripts/run_live.py --mode live --once      # single evaluation, then exit
  python scripts/run_live.py --mode live --reset-halt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot.config import Config  # noqa: E402
from goldbot.factory import make_broker, paths_for  # noqa: E402
from goldbot.live import LiveTrader  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["paper", "live"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--reset-halt", action="store_true")
    ap.add_argument("--allow-real-account", action="store_true", help="required to trade a non-demo account")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    Path(cfg.paths.get("logs", "logs")).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/goldbot_{a.mode}.log")])

    broker = make_broker(cfg, a.mode)
    state, journal = paths_for(cfg, a.mode)

    trader = LiveTrader(cfg, broker, state, journal, dry_run=a.dry_run, mode=a.mode, allow_real=a.allow_real_account)
    if a.reset_halt:
        trader.reset_halt()
    trader.run_forever(once=a.once)
