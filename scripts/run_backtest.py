"""Backtest + walk-forward + bootstrap report.

  python scripts/run_backtest.py --data data/XAUUSD_H4.csv --walk-forward --bootstrap
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot.backtest import run_backtest, walk_forward  # noqa: E402
from goldbot.config import Config  # noqa: E402
from goldbot.data import load_csv  # noqa: E402
from goldbot.metrics import block_bootstrap, drawdown  # noqa: E402


def print_metrics(title: str, m: dict) -> None:
    print(f"\n== {title} ==")
    for k, v in m.items():
        print(f"  {k:<22} {v}")


def plot(res, wf, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(res.equity.index, res.equity.values, label="in-sample (fixed params)")
    if wf is not None:
        ax[0].plot(wf.equity.index, wf.equity.values, label="walk-forward OOS", alpha=0.85)
    ax[0].set_ylabel("equity"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].fill_between(res.equity.index, drawdown(res.equity).values, 0, alpha=0.4)
    ax[1].set_ylabel("drawdown"); ax[1].grid(alpha=0.3)
    ax[2].step(res.lots.index, res.lots.values, where="post")
    ax[2].set_ylabel("lots"); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    data_path = a.data or cfg.paths.get("data")
    out = Path(a.out or cfg.paths.get("reports", "reports")); out.mkdir(parents=True, exist_ok=True)
    df = load_csv(data_path)
    print(f"data: {data_path}  {len(df)} bars  {df.index[0]} .. {df.index[-1]}")
    if "SYNTHETIC" in str(data_path).upper():
        print("!!! SYNTHETIC DATA — pipeline check only, numbers are meaningless !!!")

    res = run_backtest(df, cfg)
    print_metrics(f"In-sample, fixed params {res.params}", res.metrics)
    report = {"data": str(data_path), "params": res.params, "in_sample": res.metrics}
    res.equity.to_csv(out / "equity_insample.csv"); res.trades.to_csv(out / "trades_insample.csv", index=False)

    wf = None
    if a.walk_forward:
        bpy = cfg.bars_per_year
        train_bars, test_bars = 2 * bpy, bpy // 2
        if len(df) < train_bars + 2 * test_bars:
            test_bars = max(bpy // 4, 60)
            train_bars = max(bpy // 2, len(df) - 3 * test_bars)
            print(f"!!! short history ({len(df) / bpy:.1f} y): walk-forward with train={train_bars} / test={test_bars} bars "
                  f"— low statistical confidence")
        if len(df) < train_bars + test_bars:
            raise SystemExit("not enough history for any walk-forward window")
        wf = walk_forward(df, cfg, train_bars=train_bars, test_bars=test_bars)
        print_metrics("Walk-forward OUT-OF-SAMPLE (stitched)", wf.metrics)
        print("\n  windows:")
        for w in wf.windows:
            print(f"   {w['test_start'][:10]}..{w['test_end'][:10]} lb={w['params']['lookbacks']} tv={w['params']['target_vol']}"
                  f" | train SR {w['train_sharpe']} -> test SR {w['test_sharpe']} ret {w['test_return']:+.3f} "
                  f"mdd {w['test_maxdd']:.3f} trades {w['test_trades']}")
        report["walk_forward"] = {"metrics": wf.metrics, "windows": wf.windows}
        wf.equity.to_csv(out / "equity_walkforward.csv")

    if a.bootstrap:
        src = wf.equity if wf is not None else res.equity
        bs = block_bootstrap(src.pct_change().dropna().to_numpy(), horizon=cfg.bars_per_year)
        print_metrics("Block bootstrap, 1-year horizon (from " + ("OOS" if wf else "in-sample") + " bar returns)", bs)
        report["bootstrap_1y"] = bs

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    plot(res, wf, out / "backtest.png")
    print(f"\nwritten: {out}/report.json, equity_*.csv, trades_insample.csv, backtest.png")
