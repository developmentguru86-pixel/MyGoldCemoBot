"""Telegram push for the serverless runner (no listener; commands come via GitHub workflow_dispatch)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

from .config import Config

log = logging.getLogger("notify")


def _creds(cfg: Config) -> tuple[str, str]:
    return (cfg.telegram.token or os.environ.get("TELEGRAM_TOKEN", ""),
            str(cfg.telegram.chat_id or os.environ.get("TELEGRAM_CHAT_ID", "") or ""))


def send(cfg: Config, text: str) -> None:
    token, chat = _creds(cfg)
    if not token or chat in ("", "0"):
        log.info("telegram not configured; message: %s", text[:200])
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text[:4000]}, timeout=15)
    except Exception as e:  # noqa: BLE001
        log.error("telegram send failed: %s", e)


def send_photo(cfg: Config, path: Path, caption: str = "") -> None:
    token, chat = _creds(cfg)
    if not token or chat in ("", "0") or not Path(path).exists():
        return
    try:
        with open(path, "rb") as f:
            requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                          data={"chat_id": chat, "caption": caption[:1000]}, files={"photo": f}, timeout=60)
    except Exception as e:  # noqa: BLE001
        log.error("telegram photo failed: %s", e)
