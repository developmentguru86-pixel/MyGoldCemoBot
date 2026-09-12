"""Human-readable status of the running bot (used by scripts/status.py, run_once and the Telegram supervisor)."""
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
    except Exception as e:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc().strip().splitlines()
        lines.append(f"⚠️ broker offline: {type(e).__name__}: {str(e)[:120]}")
        lines.append("   " + " | ".join(l.strip() for l in tb[-4:-1])[:400])
        acct = None

    if acct:
        start = st.get("start_equity") or acct.balance
        pnl = acct.equity - start
        base = cfg.equity_cap if cfg.equity_cap > 0 else start
        veq = base + pnl
        lines.append(f"[{mode.upper()} / {MODES.get(acct.trade_mode, acct.trade_mode)}] {cfg.exchange.id if cfg.venue == 'exchange' else 'MT5'}")
        lines.append(f"Konto {veq:,.2f} {acct.currency} (Start {base:,.0f}; Demo-Konto real {acct.equity:,.0f})")
        lines.append(f"P&L seit Start {pnl:+,.2f} ({pnl / base * 100:+.2f}%)")
        for sym, sc in cfg.portfolio().items():
            try:
                q = br.get_quote(sym)
                pos = br.get_position(sym)
                info = br.get_symbol_info(sym)
                notional = pos * info.contract_size * q.mid
                lines.append(f"{sym}: {q.mid:,.2f} (spread {q.spread:.2f})  Position {pos:+.4f} = {notional:+,.0f} "
                             f"({notional / max(veq * sc.weight, 1):+.2f}x der {sc.weight:.0%}-Tranche)")
            except Exception as e:  # noqa: BLE001
                lines.append(f"{sym}: ⚠️ {type(e).__name__}: {str(e)[:80]}")
        if risk:
            dd = (veq / risk["hwm"] - 1) * 100 if risk.get("hwm") else 0.0
            lines.append(f"DD vom Hoch {dd:+.2f}%  Tagesstart {risk.get('day_start_equity', 0):,.2f}")
    if risk.get("halted"):
        lines.append(f"🛑 HALTED: {risk.get('halted_reason')}")
    lb = st.get("last_bar")
    if isinstance(lb, dict):
        lines.append("Letzte Bar " + ", ".join(f"{k.split('/')[0]} {str(v)[5:16]}" for k, v in lb.items()))
    else:
        lines.append(f"Letzte Bar {str(lb or '-')[:16]}")

    if journal_p and Path(journal_p).exists():
        j = pd.read_csv(journal_p).tail(n)
        lines.append("Letzte Entscheidungen:")
        for _, r in j.iterrows():
            sym = str(r.get("symbol", cfg.symbol)).split("/")[0]
            lines.append(f"  {sym} {str(r['bar_time'])[5:16]} {r['price']:,.0f} sig {r['signal']:+.2f} "
                         f"{r['current_lots']:+.3f}->{r['target_lots']:+.3f} {r['action']} "
                         f"{r['reason'] if isinstance(r['reason'], str) else ''}".rstrip())
    else:
        lines.append("Noch kein Journal – erste Entscheidung am nächsten H4-Bar-Schluss.")
    return "\n".join(lines)
