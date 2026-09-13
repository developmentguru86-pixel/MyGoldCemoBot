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
from goldbot.fx import fmt_money  # noqa: E402
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
        logging.info("no new closed bar for any symbol (%s)", trader.last_bar)
        return 0
    for r in trader.last_rows:
        sym = str(r["symbol"]).split("/")[0]
        act = r.get("action")
        if act == "trade":
            l0, l1 = float(r["current_lots"]), float(r["target_lots"])
            kind = "CLOSE" if l1 == 0.0 else ("FLIP" if l0 and (l0 > 0) != (l1 > 0) else ("REDUCE" if l0 and abs(l1) < abs(l0) else ("ADD" if l0 else "OPEN")))
            side = "Long" if l1 > 0 else ("Short" if l1 < 0 else "flat")
            msg = (f"📈 {kind} {sym} → {side} {abs(l1):.4f}\n{str(r['bar_time'])[:16]} · Fill {float(r['fill_price'] or r['price']):,.2f}"
                   f" · Ziel {r['target_exposure']:+.2f}x")
            rp = r.get("realized_pnl")
            if rp not in ("", None) and kind in ("CLOSE", "FLIP", "REDUCE"):
                net = float(rp) - float(r.get("fees") or 0.0)
                msg += f"\n💰 Realisiert: {fmt_money(float(rp))}\n   nach Gebühren ≈ {fmt_money(net)}"
            elif r.get("fees") not in ("", None):
                msg += f"\nGebühren ≈ {float(r['fees']):.2f} USDT"
            b = (trader.book or {}).get(r["symbol"], {})
            if b:
                msg += f"\n{sym} gesamt realisiert: {fmt_money(float(b.get('realized', 0.0)) - float(b.get('fees', 0.0)))}"
            msg += f"\nKonto {r['equity']:,.2f} USDT"
            notify.send(cfg, msg)
        elif act == "skip" and r.get("reason") != "market_closed":
            notify.send(cfg, f"⏸ {sym} übersprungen {str(r['bar_time'])[:16]}: {r.get('reason')}")
        elif act == "skip":
            logging.info("%s market closed at %s — target %s not sent", sym, r.get("bar_time"), r.get("target_lots"))
    if trader.rm.state.halted:
        notify.send(cfg, f"🛑 KILL-SWITCH ausgelöst: {trader.rm.state.halted_reason}. Positionen glattgestellt, "
                         "Bot pausiert bis action=reset_halt.")
    if datetime.now(timezone.utc).hour < 4:   # the 00:xx UTC run doubles as daily report
        notify.send(cfg, "📊 Tagesbericht\n" + status_text(cfg, mode, n=4))
    return 0


def action_diag(cfg: Config, mode: str) -> int:
    """Print venue contract specs and the exact sizing arithmetic per symbol."""
    from goldbot.sizing import exposure_to_lots
    from goldbot.config import ContractCfg
    br = make_broker(cfg, mode)
    acct = br.get_account()
    eq = cfg.equity_cap or acct.equity
    lines = [f"account {acct.equity:,.2f} {acct.currency} | virtual {eq:,.0f} | scale {cfg.strategy.exposure_scale} | max_lev {cfg.strategy.max_leverage}"]
    for sym, sc in cfg.portfolio().items():
        i = br.get_symbol_info(sym)
        q = br.get_quote(sym)
        ct = ContractCfg(i.contract_size, i.min_lot, i.lot_step, i.max_lot)
        sleeve = eq * sc.weight
        for exp in (0.05, 0.15, 0.5, 1.0):
            lots = exposure_to_lots(exp, sleeve, q.mid, ct, cfg.strategy.max_leverage)
            notional = lots * i.contract_size * q.mid
            lines.append(f"{sym.split('/')[0]:<4} w={sc.weight} sleeve={sleeve:,.0f} px={q.mid:,.2f} "
                         f"cs={i.contract_size} min={i.min_lot} step={i.lot_step} | exp {exp:.2f} -> {lots} lots = {notional:,.0f} USDT")
    txt = "\n".join(lines)
    Path("logs/diag.txt").write_text(txt + "\n")
    print(txt)
    notify.send(cfg, "🔧 Diagnose\n" + txt[:3500])
    return 0


def action_flatten(cfg: Config, mode: str) -> int:
    state, _ = paths_for(cfg, mode)
    br = make_broker(cfg, mode)
    res = {sym: br.close_all(sym) for sym in cfg.portfolio()}
    p = Path(state)
    st = json.loads(p.read_text()) if p.exists() else {}
    if st.get("risk"):
        st["risk"].update(halted=True, halted_reason="manual_flatten")
        p.write_text(json.dumps(st, indent=1))
    notify.send(cfg, "Glattgestellt: " + ", ".join(f"{k.split('/')[0]} net {v.get('net_lots')} ok={v.get('ok')}" for k, v in res.items())
                + ". Bot pausiert bis reset_halt.")
    return 0


def action_chat_id(cfg: Config) -> int:
    import os, requests
    token = cfg.telegram.token or os.environ.get("TELEGRAM_TOKEN", "")
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20).json()
    seen = {}
    for u in r.get("result", []):
        m = u.get("message") or u.get("edited_message") or {}
        c = m.get("chat") or {}
        if c.get("id"):
            seen[c["id"]] = f"{c.get('first_name', '')} {c.get('username', '')}".strip()
    out = Path("logs/telegram_chat.txt")
    out.write_text("\n".join(f"{cid} {name}" for cid, name in seen.items()) or "NO_MESSAGES_YET")
    for cid in seen:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": cid, "text": f"goldbot verbunden. Deine Chat-ID: {cid}"}, timeout=15)
    logging.info("chat ids: %s", seen)
    return 0


def action_chat_id(cfg: Config) -> int:
    """Discover the Telegram chat id of whoever messaged the bot; written to logs/telegram_chat.txt."""
    import os, requests
    token = cfg.telegram.token or os.environ.get("TELEGRAM_TOKEN", "")
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20).json()
    seen = {}
    for u in r.get("result", []):
        m = u.get("message") or u.get("edited_message") or {}
        c = m.get("chat") or {}
        if c.get("id"):
            seen[c["id"]] = f"{c.get('first_name', '')} {c.get('last_name', '')} @{c.get('username', '')} '{(m.get('text') or '')[:30]}'"
    out = "\n".join(f"{cid}: {who}" for cid, who in seen.items()) or "no messages yet — write to the bot first"
    Path("logs/telegram_chat.txt").write_text(out + "\n")
    print(out)
    for cid in seen:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": cid, "text": f"goldbot verbunden. Deine Chat-ID: {cid}"}, timeout=15)
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
    ap.add_argument("--action", choices=["trade", "status", "flatten", "reset_halt", "chat_id", "diag"], default="trade")
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
            txt = status_text(cfg, mode)
            Path("logs/last_status.txt").write_text(txt + "\n")
            print(txt)
            notify.send(cfg, txt); rc = 0
        elif a.action == "flatten":
            rc = action_flatten(cfg, mode)
        elif a.action == "chat_id":
            rc = action_chat_id(cfg)
        elif a.action == "diag":
            rc = action_diag(cfg, mode)
        elif a.action == "chat_id":
            rc = action_chat_id(cfg)
        elif a.action == "diag":
            rc = action_diag(cfg, mode)
        else:
            rc = action_reset(cfg, mode)
    except Exception as e:  # noqa: BLE001
        logging.exception("run failed")
        import traceback
        Path("logs/last_error.txt").write_text(f"{pd.Timestamp.now(tz='UTC').isoformat()}\n{traceback.format_exc()[-3000:]}")
        if "unfunded" in str(e):
            notify.send(cfg, "⏳ Konto hat 0 Guthaben – Bot wartet. Auf testnet.phemex.com unter Assets Test-USDT anfordern.")
            rc = 0
        else:
            notify.send(cfg, f"⚠️ goldbot Lauf fehlgeschlagen: {type(e).__name__}: {str(e)[:300]}")
            rc = 1
    sys.exit(rc)
