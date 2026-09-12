"""Backtest + walk-forward + bootstrap, per symbol and for the weighted portfolio.

  python scripts/run_backtest.py --walk-forward --bootstrap            # all portfolio symbols from data/<slug>_H4.csv
  python scripts/run_backtest.py --data data/SYNTHETIC_H4.csv         # single file, primary symbol
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot.backtest import DAILY_OVERRIDES, ROBUST_W, daily_grid, default_grid, dynamic_portfolio, evaluate_gates, grid_label, purged_cv, run_backtest, walk_forward  # noqa: E402
from goldbot.meta import meta_cv, meta_cv_pooled  # noqa: E402
from goldbot.config import Config  # noqa: E402
from goldbot.data import load_csv  # noqa: E402
from goldbot.metrics import block_bootstrap, drawdown, summarize  # noqa: E402


def slug(symbol: str) -> str:
    return symbol.split("/")[0].lower()


def print_metrics(title: str, m: dict) -> None:
    print(f"\n== {title} ==")
    for k, v in m.items():
        print(f"  {k:<22} {v}")


def wf_windows(cfg: Config, n: int) -> tuple[int, int] | None:
    bpy = cfg.bars_per_year
    train_bars, test_bars = 2 * bpy, bpy // 2
    if n < train_bars + 2 * test_bars:
        test_bars = max(bpy // 4, 60)
        train_bars = max(bpy // 2, n - 3 * test_bars)
        print(f"!!! short history ({n / bpy:.1f} y): walk-forward train={train_bars} / test={test_bars} bars — low confidence")
    if n < train_bars + test_bars:
        test_bars = max(n // 5, 200)
        train_bars = n - 2 * test_bars
        print(f"!!! very short history: train={train_bars} / test={test_bars} bars (2 windows) — indicative only")
    if train_bars < 2 * test_bars or n < train_bars + test_bars:
        print("!!! not enough history for walk-forward — skipped")
        return None
    return train_bars, test_bars


def plot(curves: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for name, eq in curves.items():
        ax[0].plot(eq.index, eq.values / eq.iloc[0] * 100, label=name)
    ax[0].set_ylabel("equity (start = 100)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    port = curves["portfolio"] if "portfolio" in curves else next(iter(curves.values()))
    ax[1].fill_between(port.index, drawdown(port).values, 0, alpha=0.4)
    ax[1].set_ylabel("drawdown"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def symbol_cfg(cfg: Config, meta: dict | None, df: pd.DataFrame) -> Config:
    """Per-symbol costs from the venue meta; slippage from slippage_pct if set."""
    if not meta:
        return cfg
    over = {"spread": float(meta.get("spread") or cfg.costs.spread)}
    if meta.get("fee_pct") is not None:
        over["fee_pct"] = float(meta["fee_pct"])
    if meta.get("funding_annual") is not None:
        over["swap_long_annual"] = -float(meta["funding_annual"])
        over["swap_short_annual"] = float(meta["funding_annual"])
    if cfg.costs.slippage_pct > 0:
        over["slippage"] = round(float(df["close"].median()) * cfg.costs.slippage_pct, 6)
    c = cfg.with_costs(**over)
    c.contract.size = float(meta.get("contract_size") or 1.0)
    c.contract.min_lot = float(meta.get("min_lot") or c.contract.min_lot)
    c.contract.lot_step = float(meta.get("lot_step") or c.contract.lot_step)
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", default=None, help="single CSV (primary symbol only)")
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--purged-cv", action="store_true", help="purged K-fold CV with embargo (independent OOS segments)")
    ap.add_argument("--directions", action="store_true", help="walk-forward long-only and short-only diagnostics")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--timeframe", default=None, help="override config timeframe (data/<slug>_<TF>.csv)")
    ap.add_argument("--grid", choices=["full", "small", "daily"], default="full")
    ap.add_argument("--meta", action="store_true", help="meta-labeling purged CV on the long-only primary (and both)")
    ap.add_argument("--meta-pooled", action="store_true", help="one meta-model across all portfolio symbols (asset one-hot)")
    ap.add_argument("--selector", choices=["robust", "sharpe"], default="robust",
                    help="parameter selection inside training windows: robust (median segment Sharpe minus penalties) or plain Sharpe")
    ap.add_argument("--out", default=None)
    ap.add_argument("--spread", type=float, default=None, help="override spread for ALL symbols (venue sensitivity)")
    ap.add_argument("--fee", type=float, default=None, help="override taker fee for ALL symbols")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    BARS_PER_DAY = {"M1": 1440, "M5": 288, "M15": 96, "M30": 48, "H1": 24, "H4": 6, "D1": 1}
    if a.timeframe:
        cfg.timeframe = a.timeframe.upper()
        cfg.bars_per_day = BARS_PER_DAY[cfg.timeframe]
    tfx = cfg.timeframe.upper()
    if tfx == "D1":
        cfg = cfg.with_strategy(**DAILY_OVERRIDES)
        if a.grid == "full":
            a.grid = "daily"
    grid = daily_grid() if a.grid == "daily" else default_grid()
    if a.grid == "small":
        grid = [g for g in grid if g["target_vol"] == 0.12 and not g["regime"]["er_window"]]
    print(f"timeframe {tfx}  bars/day {cfg.bars_per_day}  grid {len(grid)} combos  selector {a.selector} {ROBUST_W if a.selector == 'robust' else ''}")
    out = Path(a.out or cfg.paths.get("reports", "reports")); out.mkdir(parents=True, exist_ok=True)
    portfolio = cfg.portfolio()
    if a.data:
        portfolio = {cfg.symbol: next(iter(portfolio.values()))}

    report: dict = {"symbols": {}, "weights": {s: sc.weight for s, sc in portfolio.items()}, "selector": a.selector}
    wf_curves: dict[str, pd.Series] = {}
    is_curves: dict[str, pd.Series] = {}
    sleeves: dict = {}   # "XAU-long", "XAU-short", ... -> WalkForwardResult (independent strategies)
    pooled_dfs: dict = {}; pooled_cfgs: dict = {}
    for sym, sc in portfolio.items():
        path = a.data or f"data/{slug(sym)}_{tfx}.csv"
        meta_p = Path(f"data/{slug(sym)}_{tfx}_meta.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() and not a.data else None
        df = load_csv(path)
        c = symbol_cfg(cfg, meta, df)
        if a.spread is not None:
            c.costs.spread = a.spread
            df = df.drop(columns=[x for x in ("spread",) if x in df.columns])
        if a.fee is not None:
            c.costs.fee_pct = a.fee
        c.starting_equity = cfg.starting_equity * (sc.weight if sc.weight > 0 else 1.0 / max(1, len(portfolio)))  # research sleeve capital; live weight may be 0
        pooled_dfs[sym] = df; pooled_cfgs[sym] = c.with_strategy(direction="long")
        print(f"\n######## {sym}  weight {sc.weight:.0%}  {len(df)} bars  {df.index[0]} .. {df.index[-1]}")
        print(f"costs: spread {c.costs.spread} fee {c.costs.fee_pct} slippage {c.costs.slippage} "
              f"funding L/S {c.costs.swap_long_annual:+.4f}/{c.costs.swap_short_annual:+.4f}  "
              f"contract {c.contract.size}/{c.contract.min_lot}/{c.contract.lot_step}")
        if "SYNTHETIC" in str(path).upper():
            print("!!! SYNTHETIC DATA — pipeline check only, numbers are meaningless !!!")

        res = run_backtest(df, c)
        print_metrics(f"{sym} in-sample, fixed params {res.params}", res.metrics)
        entry = {"data": path, "history_source": (meta or {}).get("history_source"), "costs": c.costs.__dict__,
                 "params": res.params, "in_sample": res.metrics}
        res.equity.to_csv(out / f"equity_insample_{slug(sym)}.csv")
        res.trades.to_csv(out / f"trades_insample_{slug(sym)}.csv", index=False)
        is_curves[sym] = res.equity

        if a.walk_forward and len(df) < c.bars_per_year // 2 + 2 * max(c.bars_per_year // 4, 60):
            print(f"!!! {sym}: only {len(df) / c.bars_per_year:.2f} years — too short for any walk-forward window, skipped")
        elif a.walk_forward:
            tr, te = wf_windows(c, len(df))
            wf = walk_forward(df, c, grid=grid, train_bars=tr, test_bars=te, selector=a.selector)
            print_metrics(f"{sym} walk-forward OUT-OF-SAMPLE (both directions)", wf.metrics)
            print("  windows:")
            for w in wf.windows:
                print(f"   {w['test_start'][:10]}..{w['test_end'][:10]} {grid_label(w['params'])} | train SR {w['train_sharpe']} "
                      f"-> test SR {w['test_sharpe']} ret {w['test_return']:+.3f} mdd {w['test_maxdd']:.3f} trades {w['test_trades']}")
            entry["walk_forward"] = {"metrics": wf.metrics, "windows": wf.windows}
            wf.equity.to_csv(out / f"equity_walkforward_{slug(sym)}.csv")
            wf_curves[sym] = wf.equity
            if a.directions:
                entry["directions"] = {}
                for d in ("long", "short"):
                    wfd = walk_forward(df, c.with_strategy(direction=d), grid=grid, train_bars=tr, test_bars=te, selector=a.selector)
                    md = wfd.metrics
                    entry["directions"][d] = {**md, "windows": wfd.windows}
                    sleeves[f"{slug(sym).upper()}-{d}"] = wfd
                    print(f"  {sym} {d}-only walk-forward: CAGR {md['cagr']:+.1%} Sharpe {md['sharpe']} DSR {md['dsr']} MaxDD {md['max_drawdown']:.1%} "
                          f"trades {md['trades']} cost {md['total_cost']}")
        if a.meta:
            entry["meta"] = {}
            for d in ("long", "both"):
                mc = meta_cv(df, c.with_strategy(direction=d), k=a.folds)
                entry["meta"][d] = {"summary": mc.summary, "folds": mc.folds}
                sm = mc.summary
                print(f"\n== {sym} META-LABELING purged {a.folds}-fold, primary={d} (events {sm['events']}, horizon {sm['horizon']} bars, "
                      f"accepted {sm['accepted_share']:.0%}) ==")
                print(f"  median OOS Sharpe primary {sm['primary_median_sharpe']} -> meta {sm['meta_median_sharpe']} | "
                      f"positive folds {sm['primary_positive_folds']} -> {sm['meta_positive_folds']} | meta better in {sm['meta_better_folds']}/{a.folds}")
                for f in mc.folds:
                    print(f"   fold {f['fold']} {f['test_start'][:10]}..{f['test_end'][:10]} events {f['events']} acc {f['accepted']} "
                          f"p̄ {f['mean_p']} base {f['base_rate']} | SR {f['primary_sharpe']} -> {f['meta_sharpe']} "
                          f"ret {f['primary_return']:+.3f} -> {f['meta_return']:+.3f} trades {f['primary_trades']} -> {f['meta_trades']}")
        if a.purged_cv:
            cv = purged_cv(df, c, grid=grid, k=a.folds)
            print_metrics(f"{sym} purged {a.folds}-fold CV (embargo {cv.summary['embargo_bars']} bars)", cv.summary)
            for f in cv.folds:
                print(f"   fold {f['fold']} {f['test_start'][:10]}..{f['test_end'][:10]} {grid_label(f['params'])} | "
                      f"train SR {f['train_sharpe']} -> test SR {f['test_sharpe']} ret {f['test_return']:+.3f} trades {f['test_trades']}")
            entry["purged_cv"] = {"summary": cv.summary, "folds": cv.folds}
        report["symbols"][sym] = entry

    # ---- weighted portfolio: sum of sleeve equities on the common time axis (OOS if available)
    curves = wf_curves or is_curves
    if len(curves) > 1:
        # union time axis: a sleeve without data yet simply holds its starting cash (idle capital)
        aligned = pd.concat(curves, axis=1).ffill()
        for col in aligned.columns:
            aligned[col] = aligned[col].fillna(curves[col].iloc[0])
        port = aligned.sum(axis=1)
        port.name = "portfolio"
        pm = summarize(port, cfg.bars_per_year, cfg.bars_per_day,
                       trades=sum(((report["symbols"][s].get("walk_forward") or {}).get("metrics") or report["symbols"][s]["in_sample"])["trades"]
                                  for s in curves))
        print_metrics("PORTFOLIO " + ("walk-forward OOS" if wf_curves else "in-sample") + " (weighted sleeves)", pm)
        report["portfolio"] = {"metrics": pm, "kind": "walk_forward" if wf_curves else "in_sample"}
        port.to_csv(out / "equity_portfolio.csv")
        curves = {**{s: aligned[s] for s in aligned.columns}, "portfolio": port}
        boot_src = port
    else:
        boot_src = next(iter(curves.values()))
        only = next(iter(report["symbols"].values()))
        report["portfolio"] = {"metrics": (only.get("walk_forward") or {}).get("metrics") or only["in_sample"],
                               "kind": "walk_forward" if wf_curves else "in_sample"}

    if a.meta_pooled and len(pooled_dfs) >= 2:
        mp = meta_cv_pooled(pooled_dfs, pooled_cfgs, k=a.folds)
        report["meta_pooled"] = mp
        print(f"\n== POOLED META-LABELING (long primary, {len(pooled_dfs)} assets, {mp['pooled_events']} events, horizon {mp['horizon']}) ==")
        print(f"  pooled median Sharpe primary {mp['pooled_primary_median_sharpe']} -> meta {mp['pooled_meta_median_sharpe']} | meta better {mp['pooled_meta_better']}")
        for sym, ps in mp["per_symbol"].items():
            print(f"  {sym.split('/')[0]}: median {ps['primary_median_sharpe']} -> {ps['meta_median_sharpe']} | pos {ps['primary_positive_folds']} -> {ps['meta_positive_folds']} "
                  f"| better {ps['meta_better_folds']}/{ps['n_folds']} | accepted {ps['accepted_share']}")
            for f in ps["folds"]:
                print(f"     fold {f['fold']} {f['test_start']}..{f['test_end']} ev {f['events']} acc {f['accepted']} p̄ {f['mean_p']} | "
                      f"SR {f['primary_sharpe']} -> {f['meta_sharpe']} ret {f['primary_return']:+.3f} -> {f['meta_return']:+.3f} trades {f['primary_trades']} -> {f['meta_trades']}")

    # ---- dynamic allocation across long/short sleeves from trailing OOS quality (no look-ahead)
    if len(sleeves) >= 2:
        deq, wfinal, whist = dynamic_portfolio(sleeves, cfg, cfg.starting_equity)
        dm = summarize(deq, cfg.bars_per_year, cfg.bars_per_day,
                       trades=sum(v.metrics["trades"] for v in sleeves.values()))
        print_metrics("PORTFOLIO dynamic (long/short sleeves, weights from trailing OOS Sharpe)", dm)
        print("  final weights:", wfinal)
        for h in whist[-4:]:
            print("  ", h)
        report["portfolio_dynamic"] = {"metrics": dm, "final_weights": wfinal, "weight_history": whist,
                                       "sleeves": {k: v.metrics for k, v in sleeves.items()}}
        report["portfolio_static"] = report["portfolio"]
        report["portfolio"] = {"metrics": dm, "kind": "walk_forward_dynamic"}
        deq.to_csv(out / "equity_portfolio_dynamic.csv")
        curves["portfolio"] = deq
        boot_src = deq

    if a.bootstrap:
        bs = block_bootstrap(boot_src.pct_change().dropna().to_numpy(), horizon=cfg.bars_per_year)
        print_metrics("Block bootstrap, 1-year horizon (portfolio bar returns)", bs)
        report["bootstrap_1y"] = bs

    gates = evaluate_gates(report)
    if gates:
        print("\n== QUALITY GATES (targets for the next version, not guarantees) ==")
        for k, (v, ok) in gates.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {k:<45} {v}")
        report["gates"] = {k: {"value": v, "pass": ok} for k, (v, ok) in gates.items()}
        print(f"  -> {sum(ok for _, ok in gates.values())}/{len(gates)} passed")

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    plot(curves, out / "backtest.png")
    print(f"\nwritten: {out}/report.json, equity_*.csv, trades_insample_*.csv, backtest.png")
