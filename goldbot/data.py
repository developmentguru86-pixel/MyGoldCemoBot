"""OHLC data loading and validation. The index is always a tz-aware UTC DatetimeIndex.

Note on MT5 timestamps: the MetaTrader5 package labels bar times in *server* time
but tags them as UTC. The bot is consistent about this (day boundaries follow
server time), so treat 'UTC' here as 'broker server time'.
"""
from __future__ import annotations

import pandas as pd

REQUIRED = ("open", "high", "low", "close")


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("data needs a 'time' column or a DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in REQUIRED + ("spread", "tick_volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("NaN in OHLC")
    if (df["close"] <= 0).any():
        raise ValueError("non-positive close")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index not sorted")


def load_csv(path: str) -> pd.DataFrame:
    return _finalize(pd.read_csv(path))


def from_records(records: list[dict]) -> pd.DataFrame:
    return _finalize(pd.DataFrame(records))


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index_label="time")
