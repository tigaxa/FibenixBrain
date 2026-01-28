#!/usr/bin/env python3
"""Shared utilities for FibenixBrain."""

from __future__ import annotations

import csv
from typing import Iterable, List, Tuple


def parse_rows(path: str) -> Tuple[List[int], List[float]]:
    times: List[int] = []
    prices: List[float] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        idx_time = 0
        idx_open = 1
        idx_high = 2
        idx_low = 3
        idx_close = 4
        rows: Iterable[List[str]]
        if header and any(h.lower() in {"time", "datetime", "timestamp"} for h in header):
            lower = [h.lower() for h in header]
            idx_time = lower.index("time") if "time" in lower else 0
            idx_open = lower.index("open") if "open" in lower else 1
            idx_high = lower.index("high") if "high" in lower else 2
            idx_low = lower.index("low") if "low" in lower else 3
            idx_close = lower.index("close") if "close" in lower else 4
            rows = reader
        else:
            rows = [header] + list(reader) if header else reader

        for row in rows:
            if not row:
                continue
            try:
                t = int(float(row[idx_time]))
                o = float(row[idx_open])
                h = float(row[idx_high])
                l = float(row[idx_low])
                c = float(row[idx_close])
            except (ValueError, IndexError):
                continue
            times.append(t)
            prices.append((o + h + l + c) / 4.0)
    return times, prices


def linear_regression(xs: Iterable[float], ys: Iterable[float]) -> Tuple[float, float]:
    xs = list(xs)
    ys = list(ys)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 points")
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        raise ValueError("degenerate regression")
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def load_ohlc_rows(path: str) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        idx_time = 0
        idx_open = 1
        idx_high = 2
        idx_low = 3
        idx_close = 4
        if header and any(h.lower() in {"time", "datetime", "timestamp"} for h in header):
            lower = [h.lower() for h in header]
            idx_time = lower.index("time") if "time" in lower else 0
            idx_open = lower.index("open") if "open" in lower else 1
            idx_high = lower.index("high") if "high" in lower else 2
            idx_low = lower.index("low") if "low" in lower else 3
            idx_close = lower.index("close") if "close" in lower else 4
            data_iter: Iterable[List[str]] = reader
        else:
            data_iter = [header] + list(reader) if header else reader
        for row in data_iter:
            if not row:
                continue
            try:
                t = int(float(row[idx_time]))
                o = float(row[idx_open])
                h = float(row[idx_high])
                l = float(row[idx_low])
                c = float(row[idx_close])
            except (ValueError, IndexError):
                continue
            rows.append((t, o, h, l, c))
    return rows


def ohlc4(rows: List[Tuple[int, float, float, float, float]]) -> Tuple[List[int], List[float], List[float]]:
    times: List[int] = []
    prices: List[float] = []
    closes: List[float] = []
    for t, o, h, l, c in rows:
        times.append(t)
        closes.append(c)
        prices.append((o + h + l + c) / 4.0)
    return times, prices, closes


def extract_series(
    rows: List[Tuple[int, float, float, float, float]]
) -> Tuple[List[int], List[float], List[float], List[float]]:
    times: List[int] = []
    highs: List[float] = []
    lows: List[float] = []
    closes: List[float] = []
    for t, _o, h, l, c in rows:
        times.append(t)
        highs.append(h)
        lows.append(l)
        closes.append(c)
    return times, highs, lows, closes


def compute_regression(times: List[int], prices: List[float]) -> Tuple[float, float, int, int]:
    t0 = int(times[0])
    t1 = int(times[-1])
    xs = [t - t0 for t in times]
    slope, intercept = linear_regression(xs, prices)
    return slope, intercept, t0, t1


def median(values: List[int]) -> int:
    if not values:
        return 0
    vs = sorted(values)
    mid = len(vs) // 2
    return vs[mid]


def pip_size_for_symbol(symbol: str) -> float:
    symbol = symbol.upper()
    if symbol.endswith("JPY"):
        return 0.01
    return 0.0001
