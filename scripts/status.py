"""python scripts/status.py [--mode live|paper]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldbot.config import Config  # noqa: E402
from goldbot.status import status_text  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["paper", "live"], default=None)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    cfg = Config.load(a.config)
    print(status_text(cfg, a.mode or cfg.telegram.mode, a.n))
