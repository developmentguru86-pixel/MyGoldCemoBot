"""Reference MT5 bridge (Windows, MetaTrader 5 terminal running, package `MetaTrader5`).

Endpoint contract used by goldbot.broker.MT5HttpBroker — merge these routes into your
existing mt5_bridge.py or run this file as-is:

  GET  /health                                    -> {ok, terminal, server_time}
  GET  /account                                   -> {equity, balance, currency, margin_free, leverage}
  GET  /symbol_info?symbol=                       -> {contract_size, min_lot, lot_step, max_lot, point, digits,
                                                      spread_points, swap_long, swap_short, swap_mode}
  GET  /tick?symbol=                              -> {bid, ask, time}
  GET  /rates?symbol=&timeframe=&count=           -> [{time, open, high, low, close, tick_volume, spread}]
  GET  /rates_range?symbol=&timeframe=&start=&end=-> same (ISO dates)
  GET  /position?symbol=&magic=                   -> {net_lots, positions: [...]}
  POST /target_position {symbol, lots, comment, deviation, magic} -> {ok, net_lots, executed: [...]}
  POST /close_all       {symbol, magic}           -> {ok, closed: [...]}

Notes
- `spread` in /rates is converted to price units (points * point).
- Only positions with the bot's `magic` are touched.
- Works in hedging and netting accounts: flips close everything first, reductions
  send an opposite deal against the position ticket (partial close).
- Run: `python mt5_bridge.py --host 127.0.0.1 --port 5000` — keep it bound to localhost.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

from flask import Flask, jsonify, request

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

app = Flask(__name__)

TIMEFRAMES = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1", "W1": "TIMEFRAME_W1",
}


def _tf(name: str):
    return getattr(mt5, TIMEFRAMES[name.upper()])


def _err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _rates_to_records(rates, point: float) -> list[dict]:
    return [{
        "time": _iso(float(r["time"])), "open": float(r["open"]), "high": float(r["high"]),
        "low": float(r["low"]), "close": float(r["close"]), "tick_volume": int(r["tick_volume"]),
        "spread": float(r["spread"]) * point,
    } for r in rates]


def _point(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"unknown symbol {symbol}")
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return info.point


@app.get("/health")
def health():
    ti = mt5.terminal_info()
    return jsonify({"ok": ti is not None and ti.connected, "terminal": ti._asdict() if ti else None,
                    "server_time": datetime.now(timezone.utc).isoformat()})


@app.get("/account")
def account():
    a = mt5.account_info()
    if a is None:
        return _err("account_info failed: %s" % (mt5.last_error(),), 500)
    return jsonify({"equity": a.equity, "balance": a.balance, "currency": a.currency,
                    "margin_free": a.margin_free, "leverage": a.leverage, "login": a.login,
                    "trade_mode": a.trade_mode, "server": a.server})


@app.get("/symbol_info")
def symbol_info():
    s = request.args["symbol"]
    _point(s)
    i = mt5.symbol_info(s)
    return jsonify({"contract_size": i.trade_contract_size, "min_lot": i.volume_min, "lot_step": i.volume_step,
                    "max_lot": i.volume_max, "point": i.point, "digits": i.digits, "spread_points": i.spread,
                    "swap_long": i.swap_long, "swap_short": i.swap_short, "swap_mode": i.swap_mode})


@app.get("/tick")
def tick():
    s = request.args["symbol"]
    _point(s)
    t = mt5.symbol_info_tick(s)
    if t is None:
        return _err("no tick", 500)
    return jsonify({"bid": t.bid, "ask": t.ask, "time": _iso(t.time)})


@app.get("/rates")
def rates():
    s, tf, count = request.args["symbol"], request.args.get("timeframe", "H4"), int(request.args.get("count", 500))
    point = _point(s)
    r = mt5.copy_rates_from_pos(s, _tf(tf), 0, count)
    if r is None:
        return _err("copy_rates_from_pos failed: %s" % (mt5.last_error(),), 500)
    return jsonify(_rates_to_records(r, point))


@app.get("/rates_range")
def rates_range():
    s, tf = request.args["symbol"], request.args.get("timeframe", "H4")
    start = datetime.fromisoformat(request.args["start"]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(request.args["end"]).replace(tzinfo=timezone.utc)
    point = _point(s)
    r = mt5.copy_rates_range(s, _tf(tf), start, end)
    if r is None:
        return _err("copy_rates_range failed: %s" % (mt5.last_error(),), 500)
    return jsonify(_rates_to_records(r, point))


def _our_positions(symbol: str, magic: int):
    pos = mt5.positions_get(symbol=symbol) or []
    return [p for p in pos if p.magic == magic]


def _net(positions) -> float:
    return round(sum(p.volume if p.type == mt5.POSITION_TYPE_BUY else -p.volume for p in positions), 8)


@app.get("/position")
def position():
    s, magic = request.args["symbol"], int(request.args.get("magic", 0))
    pos = _our_positions(s, magic)
    return jsonify({"net_lots": _net(pos), "positions": [
        {"ticket": p.ticket, "type": "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell",
         "volume": p.volume, "price_open": p.price_open, "profit": p.profit} for p in pos]})


def _send(req: dict) -> dict:
    """Try the filling modes brokers commonly accept until one is accepted."""
    last = None
    for filling in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
        req["type_filling"] = filling
        res = mt5.order_send(req)
        last = res
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            return {"ok": True, "retcode": res.retcode, "deal": res.deal, "volume": res.volume, "price": res.price}
        if res is not None and res.retcode != mt5.TRADE_RETCODE_INVALID_FILL:
            break
    return {"ok": False, "retcode": getattr(last, "retcode", None), "comment": getattr(last, "comment", str(mt5.last_error()))}


def _market(symbol: str, volume: float, side: str, deviation: int, magic: int, comment: str,
            position_ticket: int | None = None) -> dict:
    t = mt5.symbol_info_tick(symbol)
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(volume),
        "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
        "price": t.ask if side == "buy" else t.bid, "deviation": int(deviation), "magic": int(magic),
        "comment": comment[:31], "type_time": mt5.ORDER_TIME_GTC,
    }
    if position_ticket is not None:
        req["position"] = int(position_ticket)
    return _send(req)


def _close_volume(positions, volume: float, deviation: int, magic: int, comment: str) -> list[dict]:
    """Close `volume` lots across the given same-direction positions (partial closes allowed)."""
    out, remaining = [], volume
    for p in sorted(positions, key=lambda p: p.volume, reverse=True):
        if remaining <= 1e-9:
            break
        v = min(p.volume, remaining)
        side = "sell" if p.type == mt5.POSITION_TYPE_BUY else "buy"
        r = _market(p.symbol, v, side, deviation, magic, comment, position_ticket=p.ticket)
        out.append({"ticket": p.ticket, "closed": v, **r})
        if not r["ok"]:
            break
        remaining -= v
    return out


@app.post("/target_position")
def target_position():
    b = request.get_json(force=True)
    symbol, target = b["symbol"], float(b["lots"])
    magic, deviation, comment = int(b.get("magic", 0)), int(b.get("deviation", 30)), str(b.get("comment", "goldbot"))
    info = mt5.symbol_info(symbol)
    step = info.volume_step
    target = round(round(target / step) * step, 8)
    pos = _our_positions(symbol, magic)
    net = _net(pos)
    executed = []

    if abs(target - net) < step / 2:
        return jsonify({"ok": True, "noop": True, "net_lots": net})

    # flip or flatten: close everything first
    if net != 0.0 and (target == 0.0 or math.copysign(1, target) != math.copysign(1, net)):
        executed += _close_volume(pos, abs(net), deviation, magic, comment)
        if any(not e["ok"] for e in executed):
            return jsonify({"ok": False, "executed": executed, "net_lots": _net(_our_positions(symbol, magic))})
        net = 0.0

    delta = round(target - net, 8)
    if abs(delta) >= step - 1e-9:
        if net == 0.0 or math.copysign(1, delta) == math.copysign(1, net):
            r = _market(symbol, abs(delta), "buy" if delta > 0 else "sell", deviation, magic, comment)
            executed.append({"opened": delta, **r})
        else:
            executed += _close_volume(_our_positions(symbol, magic), abs(delta), deviation, magic, comment)

    ok = all(e.get("ok", False) for e in executed)
    return jsonify({"ok": ok, "executed": executed, "net_lots": _net(_our_positions(symbol, magic))})


@app.post("/close_all")
def close_all():
    b = request.get_json(force=True)
    symbol, magic = b["symbol"], int(b.get("magic", 0))
    pos = _our_positions(symbol, magic)
    closed = _close_volume(pos, abs(_net(pos)), int(b.get("deviation", 30)), magic, "goldbot close_all")
    return jsonify({"ok": all(c["ok"] for c in closed), "closed": closed,
                    "net_lots": _net(_our_positions(symbol, magic))})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--login", type=int, default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--server", default=None)
    a = ap.parse_args()
    if mt5 is None:
        raise SystemExit("pip install MetaTrader5 (Windows only)")
    kw = {k: v for k, v in dict(login=a.login, password=a.password, server=a.server).items() if v}
    if not mt5.initialize(**kw):
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")
    print("MT5 connected:", mt5.account_info().login, mt5.account_info().server)
    app.run(host=a.host, port=a.port, threaded=False)
