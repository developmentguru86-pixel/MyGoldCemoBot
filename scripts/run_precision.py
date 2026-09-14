"""Layer 1 (quant direction) + Layer 2 (precision entry) — out-of-sample evaluation.

  python scripts/run_precision.py --timeframe H4 --folds 6
Layer 1 gates which bars may be entered (long-only exposure > 0). Layer 2 supplies the entry.
Entry parameters are NOT optimised per fold: one fixed rule set, so the result is not a search.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot.config import Config  # noqa: E402
from goldbot.data import load_csv  # noqa: E402
from goldbot.precision import EntryCfg, MTFCfg, equity_curve, find_setups_mtf, metrics, resolve, ruin_probabilities, run_asset  # noqa: E402
from goldbot.strategy import compute_exposure  # noqa: E402

BPD = {"5M": 288, "M15": 96, "H1": 24, "H4": 6, "D1": 1}


def slug(s: str) -> str:
    return s.split("/")[0].lower()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--timeframe", default="H4")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--risk", type=float, default=0.025)
    ap.add_argument("--target-r", type=float, default=2.0)
    ap.add_argument("--no-layer1", action="store_true", help="entry engine alone, without the quant gate")
    ap.add_argument("--mtf", action="store_true", help="5m entries at 1h/4h/PDH-PDL/weekly levels with volume confirmation")
    ap.add_argument("--symbols", default=None, help="comma list, e.g. BTC/USDT:USDT,ETH/USDT:USDT")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    tfx = a.timeframe.upper()
    cfg.timeframe, cfg.bars_per_day = tfx, BPD[tfx]
    ecfg = MTFCfg(risk_per_trade=a.risk, target_r=a.target_r) if a.mtf else EntryCfg(risk_per_trade=a.risk, target_r=a.target_r)
    if a.mtf:
        ecfg.min_stop_frac, ecfg.max_stop_frac = 0.002, 0.02   # the 0.3-0.6% regime
        ecfg.sweep_max_bars, ecfg.bos_max_bars, ecfg.time_stop_bars = 6, 24, 288
    if tfx == "D1":
        ecfg.pivot_left, ecfg.pivot_right, ecfg.level_window = 5, 3, 120
        ecfg.time_stop_bars, ecfg.bos_max_bars = 20, 5

    all_trades: list[dict] = []
    per_symbol = {}
    symbols = [x.strip() for x in a.symbols.split(",")] if a.symbols else list(cfg.portfolio())
    for sym in symbols:
        p = Path(f"data/{slug(sym)}_{tfx}.csv")
        if not p.exists():
            p = Path(f"data/{slug(sym)}_{tfx}_long.csv")
        if not p.exists() and a.mtf:
            p = Path(f"data/{slug(sym)}_5M.csv")
        if not p.exists():
            print(f"{sym}: no {tfx} data"); continue
        df = load_csv(p)
        meta_p = Path(f"data/{slug(sym)}_{tfx}_meta.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        px = float(df["close"].median())
        fee = float(meta.get("fee_pct") or cfg.costs.fee_pct)
        spread = float(meta.get("spread") or cfg.costs.spread)
        cost_frac = 2 * (fee + (spread / 2 + px * 1e-4) / px)      # round trip as a fraction of price

        c = cfg.with_strategy(direction="long",
                              lookbacks=[int(round(x * cfg.bars_per_day)) for x in (20, 60, 120)],
                              vol_span=max(20 * cfg.bars_per_day, 20),
                              min_hold_bars=3 * cfg.bars_per_day,
                              kelly={"window": 120 * cfg.bars_per_day},
                              regime={"er_window": 0, "vol_pct_window": 120 * cfg.bars_per_day})
        allowed = None
        if not a.no_layer1:
            expo = compute_exposure(df, c)["exposure"].to_numpy(dtype=float)
            allowed = expo > 0                                     # layer 1: long or flat
        if a.mtf:
            tr = []
            for st in find_setups_mtf(df, ecfg, allowed):
                x = resolve(df, st, ecfg)
                x["r_net"] = x["r"] - cost_frac / st["risk_frac"]
                tr.append(x)
            if tr:
                import numpy as _np
                print(f"      Median-Stop {_np.median([t['risk_frac'] for t in tr])*100:.3f} % "
                      f"-> Kosten {cost_frac*100:.3f} % = {_np.median([cost_frac/t['risk_frac'] for t in tr]):.2f}R pro Trade")
        else:
            tr = run_asset(df, ecfg, allowed, cost_frac)
        for t in tr:
            t["symbol"] = sym
        all_trades += tr
        m = metrics(tr, ecfg, cfg.starting_equity)
        per_symbol[sym] = m
        print(f"{sym.split('/')[0]:<5} {len(df):>6} bars {str(df.index[0])[:10]}..{str(df.index[-1])[:10]} | "
              f"Trades {m.get('trades', 0):>4} ({m.get('trades_per_year', 0)}/J) Winrate {m.get('winrate')} "
              f"Exp {m.get('expectancy_R')}R PF {m.get('profit_factor')}")

    if not all_trades:
        raise SystemExit("no trades generated")

    # ---- portfolio: chronological, capped concurrency, compounding
    all_trades.sort(key=lambda t: t["time"])
    open_until: list = []
    kept = []
    for t in all_trades:
        open_until = [x for x in open_until if x > t["time"]]
        if len(open_until) >= ecfg.max_concurrent:
            continue
        kept.append(t)
        open_until.append(t["time"])
    print(f"\nSetups gesamt {len(all_trades)}, nach Concurrency-Cap ({ecfg.max_concurrent}) {len(kept)}")

    eq, det = equity_curve(kept, ecfg, cfg.starting_equity)
    span = (kept[-1]["time"] - kept[0]["time"]).days
    m = metrics(kept, ecfg, cfg.starting_equity, trading_days=span)
    ruin = ruin_probabilities(kept, ecfg)

    # ---- out-of-sample split: the entry rules are fixed, so every fold is OOS by construction;
    # report per-fold stability rather than a single aggregate
    edges = [kept[0]["time"] + (kept[-1]["time"] - kept[0]["time"]) * i / a.folds for i in range(a.folds + 1)]
    folds = []
    for i in range(a.folds):
        seg = [t for t in kept if edges[i] <= t["time"] < edges[i + 1]]
        if len(seg) < 5:
            continue
        fm = metrics(seg, ecfg, cfg.starting_equity)
        folds.append({"fold": i, "from": str(edges[i])[:10], "to": str(edges[i + 1])[:10],
                      "trades": fm["trades"], "winrate": fm["winrate"], "expectancy_R": fm["expectancy_R"],
                      "profit_factor": fm["profit_factor"]})

    print("\n================ ENTRY-ENGINE (Layer 1 + Layer 2) ================")
    for k, v in m.items():
        print(f"  {k:<22} {v}")
    print(f"  {'risk_per_trade':<22} {ecfg.risk_per_trade:.1%}")
    print(f"  {'target':<22} {ecfg.target_r}R")
    for k, v in ruin.items():
        print(f"  {k:<22} {v}")
    print("\n  Folds (feste Regeln, daher jedes Fenster OOS):")
    for f in folds:
        print(f"   {f['from']}..{f['to']} Trades {f['trades']:>4} Winrate {f['winrate']:.2f} "
              f"Exp {f['expectancy_R']:+.3f}R PF {f['profit_factor']}")
    pos = sum(1 for f in folds if f["expectancy_R"] > 0)
    print(f"   -> {pos}/{len(folds)} Fenster mit positivem Erwartungswert")

    out = Path(a.out); out.mkdir(exist_ok=True)
    (out / "precision.json").write_text(json.dumps(
        {"timeframe": tfx, "layer1": not a.no_layer1, "risk_per_trade": ecfg.risk_per_trade,
         "target_r": ecfg.target_r, "portfolio": m, "ruin": ruin, "folds": folds,
         "per_symbol": per_symbol}, indent=1, default=str))
    print(f"\nwritten: {out}/precision.json")
