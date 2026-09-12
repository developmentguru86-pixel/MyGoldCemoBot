"""Human-readable status of the running bot (used by scripts/status.py and the Telegram supervisor)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Config
from .factory import make_broker, paths_for

MODES = {0: "DEMO", 1: "CONTEST", 2: "REAL"}


def status_text(cfg: Config, mode: str = "live", n: int = 5) -> str:
    br = make_broker(cfg, mode)
    state_p, journal_p = paths_for(cfg, mode)

    st = json.loads(Path(state_p).read_text()) if Path(state_p).exists() else {}
    risk = st.get("risk") or {}
    lines = []
    try:
        acct = br.get_account()
        pos = br.get_position(cfg.symbol)
        q = br.get_quote(cfg.symbol)
    except Exception as e:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc().strip().splitlines()
        lines.append(f"⚠️ broker offline: {type(e).__name__}: {str(e)[:120]}")
        lines.append("   " + " | ".join(l.strip() for l in tb[-4:-1])[:400])
        acct = pos = q = None

    if acct:
        start = st.get("start_equity") or acct.balance
        pnl = acct.equity - start
        base = cfg.equity_cap if cfg.equity_cap > 0 else start
        veq = base + pnl
        lines.append(f"[{mode.upper()} / {MODES.get(acct.trade_mode, acct.trade_mode)}] {cfg.symbol} {q.bid:.2f}/{q.ask:.2f} spread {q.spread:.2f}")
        lines.append(f"Konto {veq:,.2f} {acct.currency} (Start {base:,.0f}; Demo-Konto real {acct.equity:,.0f})")
        lines.append(f"P&L seit Start {pnl:+,.2f} ({pnl / base * 100:+.2f}%)")
        lines.append(f"Position {pos:+.4f} = {pos * cfg.contract.size * q.mid / veq:+.2f}x")
        if risk:
            dd = (acct.equity / risk["hwm"] - 1) * 100 if risk.get("hwm") else 0.0
            lines.append(f"DD vom Hoch {dd:+.2f}%  Tagesstart {risk.get('day_start_equity', 0):,.2f}")
    dbg = getattr(br, "account_debug", "") or getattr(getattr(br, "data", None), "account_debug", "")
    if dbg:
        lines.append(dbg)
    if risk.get("halted"):
        lines.append(f"🛑 HALTED: {risk.get('halted_reason')}")
    lines.append(f"Letzte Bar {str(st.get('last_bar') or '-')[:16]}")

    if journal_p and Path(journal_p).exists():
        j = pd.read_csv(journal_p).tail(n)
        lines.append("Letzte Entscheidungen:")
        for _, r in j.iterrows():
            lines.append(f"  {str(r['bar_time'])[5:16]} {r['price']:.0f} sig {r['signal']:+.2f} "
                         f"{r['current_lots']:+.2f}->{r['target_lots']:+.2f} {r['action']} {r['reason'] if isinstance(r['reason'], str) else ''}".rstrip())
    else:
        lines.append("Noch kein Journal – erste Entscheidung am nächsten H4-Bar-Schluss.")
    return "\n".join(lines)
