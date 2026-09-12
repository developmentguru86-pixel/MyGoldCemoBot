"""One-click setup + start (Windows, MT5 terminal open and logged in to the demo account).

  python scripts/bootstrap.py                 # install -> bridge -> history -> backtest -> bot (demo, live orders)
  python scripts/bootstrap.py --mode paper    # same, but simulated fills
  python scripts/bootstrap.py --skip-backtest
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def run(*args: str, check: bool = True) -> int:
    print(f"\n>>> {' '.join(args)}", flush=True)
    return subprocess.run(list(args), cwd=ROOT, check=check).returncode


def spawn(title: str, *args: str) -> subprocess.Popen:
    print(f"\n>>> [{title}] {' '.join(args)}", flush=True)
    if NEW_CONSOLE:
        return subprocess.Popen(list(args), cwd=ROOT, creationflags=NEW_CONSOLE)
    return subprocess.Popen(list(args), cwd=ROOT)


def wait_bridge(url: str, seconds: int = 90) -> None:
    import requests
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            r = requests.get(f"{url}/health", timeout=3).json()
            if r.get("ok"):
                print(f"bridge OK: login {r.get('terminal', {}).get('name')} connected")
                return
            print("bridge up but MT5 not connected — is the terminal open and logged in?")
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("bridge did not come up. Open MetaTrader 5, log in to the Swissquote demo account, retry.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="live")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--years", type=float, default=6.0)
    a = ap.parse_args()
    os.chdir(ROOT)

    if sys.version_info < (3, 10):
        raise SystemExit("Python >= 3.10 required")
    if not a.skip_install:
        run(PY, "-m", "pip", "install", "-q", "-r", "requirements.txt")
        if os.name == "nt":
            run(PY, "-m", "pip", "install", "-q", "-r", "bridge/requirements-bridge.txt")
        else:
            print("not Windows: MetaTrader5 package unavailable, bridge must run on the MT5 machine")

    from goldbot.config import Config  # noqa: E402  (after install)
    cfg = Config.load("config.yaml")

    bridge = spawn("bridge", PY, "bridge/mt5_bridge.py", "--host", "127.0.0.1", "--port", "5000")
    wait_bridge(cfg.bridge.url)

    run(PY, "scripts/fetch_history.py", "--years", str(a.years), "--update-config")
    if not a.skip_backtest:
        run(PY, "scripts/run_backtest.py", "--walk-forward", "--bootstrap", check=False)
        print("\n>>> report: reports/report.json + reports/backtest.png")

    if cfg.telegram.token:
        sup = spawn("telegram", PY, "scripts/run_telegram.py")
        print(f"\nTelegram supervisor started (pid {sup.pid}). Open Telegram on your phone: /status /start /stop /flatten")
    else:
        bot = spawn("goldbot", PY, "scripts/run_live.py", "--mode", a.mode)
        print(f"\nbot started in mode={a.mode} (pid {bot.pid}). Status any time:  python scripts/status.py")
        print("Stop: close the goldbot window (Ctrl+C). Bridge window can stay open.")
