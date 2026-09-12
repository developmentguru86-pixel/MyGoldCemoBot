"""Telegram supervisor — remote control from the phone.

Runs on the MT5 machine next to the bridge. Manages the bot process, forwards
every trade and every kill-switch as a push message, answers commands:

  /status    equity, P&L, position, last decisions
  /start     start the bot (mode from config.telegram.mode)
  /stop      stop the bot process (open position stays)
  /flatten   close all bot positions on the broker, then stop
  /report    last backtest / walk-forward numbers + chart
  /backtest  re-run backtest + walk-forward in the background
  /reset     clear a halt (after you looked at why it halted)
  /help

Only messages from config.telegram.chat_id are accepted. With chat_id = 0 the
supervisor replies to anyone with their chat id so you can paste it into config.yaml.
Plain requests + long polling, no extra library.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from .factory import make_broker
from .config import Config
from .status import status_text

log = logging.getLogger("supervisor")


class Supervisor:
    def __init__(self, cfg: Config, root: Path):
        self.cfg, self.root = cfg, root
        t = cfg.telegram
        if not t.token:
            raise SystemExit("telegram.token missing in config.yaml (create a bot with @BotFather)")
        self.api = f"https://api.telegram.org/bot{t.token}"
        self.chat_id = int(t.chat_id)
        self.mode = t.mode
        self.bot: subprocess.Popen | None = None
        self.bridge: subprocess.Popen | None = None
        self.offset = 0
        self.rows_seen = 0
        self.halt_alerted = False
        self.last_daily: str | None = None
        self.state_p = Path(cfg.paths.get("state_live" if self.mode == "live" else "state_paper",
                                          f"state/{self.mode}_state.json"))
        self.journal_p = Path(cfg.paths.get("journal_live" if self.mode == "live" else "journal_paper",
                                            f"logs/journal_{self.mode}.csv"))
        if self.journal_p.exists():
            self.rows_seen = max(0, sum(1 for _ in open(self.journal_p)) - 1)

    # ---- telegram I/O
    def send(self, text: str, chat_id: int | None = None) -> None:
        try:
            requests.post(f"{self.api}/sendMessage", json={"chat_id": chat_id or self.chat_id, "text": text[:4000]}, timeout=15)
        except Exception as e:  # noqa: BLE001
            log.error("send failed: %s", e)

    def send_photo(self, path: Path, caption: str = "") -> None:
        try:
            with open(path, "rb") as f:
                requests.post(f"{self.api}/sendPhoto", data={"chat_id": self.chat_id, "caption": caption[:1000]},
                              files={"photo": f}, timeout=60)
        except Exception as e:  # noqa: BLE001
            log.error("send_photo failed: %s", e)

    def updates(self) -> list[dict]:
        try:
            r = requests.get(f"{self.api}/getUpdates", params={"offset": self.offset, "timeout": 25}, timeout=35).json()
            out = r.get("result", [])
            if out:
                self.offset = out[-1]["update_id"] + 1
            return out
        except Exception as e:  # noqa: BLE001
            log.error("getUpdates failed: %s", e)
            time.sleep(5)
            return []

    # ---- processes
    def _spawn(self, *args: str, logname: str) -> subprocess.Popen:
        out = open(self.root / "logs" / logname, "a", buffering=1)
        return subprocess.Popen([sys.executable, *args], cwd=self.root, stdout=out, stderr=subprocess.STDOUT)

    def bridge_ok(self) -> bool:
        try:
            return bool(requests.get(f"{self.cfg.bridge.url}/health", timeout=3).json().get("ok"))
        except Exception:  # noqa: BLE001
            return False

    def ensure_bridge(self) -> bool:
        if self.bridge_ok():
            return True
        if self.bridge is None or self.bridge.poll() is not None:
            log.info("starting bridge")
            self.bridge = self._spawn("bridge/mt5_bridge.py", "--host", "127.0.0.1", "--port", "5000", logname="bridge.log")
        for _ in range(30):
            time.sleep(3)
            if self.bridge_ok():
                return True
        return False

    def bot_running(self) -> bool:
        return self.bot is not None and self.bot.poll() is None

    def start_bot(self) -> str:
        if self.bot_running():
            return "Bot läuft bereits."
        if not self.ensure_bridge():
            return "Bridge nicht erreichbar – ist MetaTrader 5 offen und eingeloggt?"
        self.bot = self._spawn("scripts/run_live.py", "--mode", self.mode, logname=f"bot_{self.mode}.log")
        self.halt_alerted = False
        time.sleep(4)
        if not self.bot_running():
            return "Bot sofort beendet – siehe logs/bot_%s.log (Echtgeld-Konto? Kontraktdaten?)" % self.mode
        return f"Bot gestartet ({self.mode}, pid {self.bot.pid}). Handelt am nächsten H4-Bar-Schluss."

    def stop_bot(self) -> str:
        if not self.bot_running():
            return "Bot läuft nicht."
        self.bot.terminate()
        try:
            self.bot.wait(15)
        except subprocess.TimeoutExpired:
            self.bot.kill()
        return "Bot gestoppt. Offene Position bleibt – /flatten schließt sie."

    def flatten(self) -> str:
        msg = self.stop_bot()
        try:
            r = make_broker(self.cfg, self.mode).close_all(self.cfg.symbol)
            return f"{msg}\nGlattgestellt: net {r.get('net_lots')} Lot, ok={r.get('ok')}"
        except Exception as e:  # noqa: BLE001
            return f"{msg}\nFlatten fehlgeschlagen: {str(e)[:120]}"

    def reset_halt(self) -> str:
        if not self.state_p.exists():
            return "Kein State."
        st = json.loads(self.state_p.read_text())
        if st.get("risk"):
            st["risk"].update(halted=False, halted_reason="", bridge_failures=0)
            self.state_p.write_text(json.dumps(st, indent=1))
        self.halt_alerted = False
        return "Halt zurückgesetzt. /start zum Weiterhandeln."

    def report(self) -> str:
        p = self.root / self.cfg.paths.get("reports", "reports") / "report.json"
        if not p.exists():
            return "Kein Report – /backtest ausführen."
        r = json.loads(p.read_text())
        m = (r.get("walk_forward") or {}).get("metrics") or r["in_sample"]
        kind = "Walk-Forward OOS" if r.get("walk_forward") else "In-sample"
        txt = (f"{kind} ({r['data']}):\nCAGR {m['cagr']:+.1%}  Sharpe {m['sharpe']}  MaxDD {m['max_drawdown']:.1%}\n"
               f"PSR {m.get('psr_gt_0')}  Trades {m['trades']}  Kosten {m['total_cost']} + Swap {m['total_swap']}\n"
               f"P&L/Handelstag {m.get('pnl_per_trading_day')}")
        bs = r.get("bootstrap_1y")
        if bs:
            txt += f"\nBootstrap 1J: Return p05/p50/p95 {bs['return_p05']:+.1%}/{bs['return_p50']:+.1%}/{bs['return_p95']:+.1%}, P(Verlust) {bs['prob_loss']:.0%}"
        png = p.parent / "backtest.png"
        if png.exists():
            self.send_photo(png, txt)
            return ""
        return txt

    def backtest_async(self) -> str:
        def job():
            rc = subprocess.run([sys.executable, "scripts/run_backtest.py", "--walk-forward", "--bootstrap"],
                                cwd=self.root, stdout=open(self.root / "logs" / "backtest.log", "a"),
                                stderr=subprocess.STDOUT).returncode
            self.send("Backtest fertig." if rc == 0 else "Backtest fehlgeschlagen – logs/backtest.log")
            if rc == 0:
                t = self.report()
                if t:
                    self.send(t)
        threading.Thread(target=job, daemon=True).start()
        return "Backtest läuft im Hintergrund (2–5 Minuten)…"

    # ---- commands
    HELP = ("/status – Konto, Position, letzte Entscheidungen\n/start – Bot starten\n/stop – Bot stoppen\n"
            "/flatten – alles schließen + stoppen\n/report – Backtest-Zahlen + Chart\n/backtest – neu rechnen\n"
            "/reset – Halt aufheben\n/help")

    def handle(self, text: str) -> str:
        cmd = text.strip().split()[0].lower().split("@")[0] if text.strip() else ""
        if cmd == "/status":
            return status_text(self.cfg, self.mode) + ("\nBot: läuft" if self.bot_running() else "\nBot: GESTOPPT")
        if cmd == "/start":
            return self.start_bot()
        if cmd == "/stop":
            return self.stop_bot()
        if cmd == "/flatten":
            return self.flatten()
        if cmd == "/report":
            return self.report()
        if cmd == "/backtest":
            return self.backtest_async()
        if cmd == "/reset":
            return self.reset_halt()
        return self.HELP

    # ---- background checks
    def check_journal(self) -> None:
        if not self.journal_p.exists():
            return
        try:
            j = pd.read_csv(self.journal_p)
        except Exception:  # noqa: BLE001
            return
        new = j.iloc[self.rows_seen:]
        self.rows_seen = len(j)
        for _, r in new.iterrows():
            if r["action"] == "trade":
                self.send(f"📈 TRADE {str(r['bar_time'])[:16]}\n{self.cfg.symbol} {r['price']:.2f}\n"
                          f"{r['current_lots']:+.2f} → {r['target_lots']:+.2f} Lot (Ziel {r['target_exposure']:+.2f}x)\n"
                          f"Equity {r['equity']:,.2f}  Grund: {r['reason'] if isinstance(r['reason'], str) else 'signal'}")
            elif r["action"] == "skip":
                self.send(f"⏸ übersprungen {str(r['bar_time'])[:16]}: {r['reason']}")

    def check_halt(self) -> None:
        if not self.state_p.exists():
            return
        try:
            risk = json.loads(self.state_p.read_text()).get("risk") or {}
        except Exception:  # noqa: BLE001
            return
        if risk.get("halted") and not self.halt_alerted:
            self.halt_alerted = True
            self.send(f"🛑 KILL-SWITCH: {risk.get('halted_reason')}\nBot handelt nicht mehr. /status ansehen, dann /reset + /start oder /flatten.")

    def check_bot_alive(self) -> None:
        if self.bot is not None and self.bot.poll() is not None:
            rc = self.bot.returncode
            self.bot = None
            self.send(f"⚠️ Bot-Prozess beendet (exit {rc}). logs/bot_{self.mode}.log prüfen. /start zum Neustart.")

    def daily_summary(self) -> None:
        now = datetime.now()
        today = now.date().isoformat()
        if now.hour >= self.cfg.telegram.daily_summary_hour and self.last_daily != today:
            self.last_daily = today
            self.send("📊 Tagesbericht\n" + status_text(self.cfg, self.mode, n=3))

    # ---- main loop
    def run(self) -> None:
        (self.root / "logs").mkdir(exist_ok=True)
        ok = self.ensure_bridge()
        self.send("goldbot Supervisor online.\n" + ("Bridge OK." if ok else "⚠️ Bridge nicht erreichbar.") + "\n" + self.HELP)
        if self.cfg.telegram.autostart and ok:
            self.send(self.start_bot())
        last_bg = 0.0
        while True:
            for u in self.updates():
                msg = u.get("message") or u.get("edited_message") or {}
                text, cid = msg.get("text", ""), (msg.get("chat") or {}).get("id")
                if cid is None or not text:
                    continue
                if self.chat_id == 0:
                    self.send(f"Deine chat_id ist {cid}. In config.yaml unter telegram.chat_id eintragen und Supervisor neu starten.", cid)
                    continue
                if cid != self.chat_id:
                    log.warning("ignored message from chat %s", cid)
                    continue
                log.info("cmd %s", text)
                try:
                    reply = self.handle(text)
                except Exception as e:  # noqa: BLE001
                    reply = f"Fehler: {str(e)[:200]}"
                if reply:
                    self.send(reply)
            if time.time() - last_bg > 20:
                last_bg = time.time()
                try:
                    self.check_journal()
                    self.check_halt()
                    self.check_bot_alive()
                    self.daily_summary()
                except Exception as e:  # noqa: BLE001
                    log.error("background check failed: %s", e)
