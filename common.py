#!/usr/bin/env python3
"""FibenixBrainの共通ユーティリティ.

メモ:
- EnvSnapshot はチャネル上限/下限とブレイクアウト判定ライン(p0/p1)を保持する。
- assess_timeframe は高値/安値の傾きから基準傾きを決め、全バーを走査して
  チャネルを最も広く包む切片を算出する。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class EnvSnapshot:
    """1つの時間足に対するトレンド/チャネルのスナップショット."""
    trend_dir: int
    slope: float
    line_now: float
    last_close: float
    t0: int
    t1: int
    top_p0: float
    top_p1: float
    btm_p0: float
    btm_p1: float
    p0: float
    p1: float


@dataclass
class SwingPoint:
    """固定depthのウィンドウで検出したスイングポイント."""
    index: int
    time: int
    price: float


@dataclass
class BreakoutInfo:
    """トレンドラインに対する直近のブレイク情報."""
    index: int
    time: int
    direction: int


def parse_rows(path: str) -> Tuple[List[int], List[float]]:
    """OHLC CSVを読み込み、(time, ohlc4価格)を返す."""
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
    """y = slope * x + intercept の回帰係数を返す."""
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
    """OHLC CSVを (time, open, high, low, close) のタプルで返す."""
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
    """time, ohlc4価格, close を返す."""
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
    """time, high, low, close の配列を返す."""
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
    """回帰係数を計算し (slope, intercept, t0, t1) を返す."""
    t0 = int(times[0])
    t1 = int(times[-1])
    xs = [t - t0 for t in times]
    slope, intercept = linear_regression(xs, prices)
    return slope, intercept, t0, t1


def median(values: List[int]) -> int:
    """中央値(ソート後の中央要素)を返す."""
    if not values:
        return 0
    vs = sorted(values)
    mid = len(vs) // 2
    return vs[mid]


def pip_size_for_symbol(symbol: str) -> float:
    """FXシンボルのpipサイズを返す (JPYペアは0.01)."""
    symbol = symbol.upper()
    if symbol.endswith("JPY"):
        return 0.01
    return 0.0001


def line_at(snapshot: EnvSnapshot, t: int) -> float:
    """時刻tにおけるブレイクアウト判定ラインの価格."""
    return snapshot.p0 + snapshot.slope * float(t - snapshot.t0)


def find_swings(
    rows: List[Tuple[int, float, float, float, float]], depth: int
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """固定depthウィンドウでスイング高値/安値を検出する."""
    highs: List[SwingPoint] = []
    lows: List[SwingPoint] = []
    if depth < 1 or len(rows) < (2 * depth + 1):
        return highs, lows
    for i in range(depth, len(rows) - depth):
        _t, _o, h, l, _c = rows[i]
        window_highs = [rows[j][2] for j in range(i - depth, i + depth + 1)]
        window_lows = [rows[j][3] for j in range(i - depth, i + depth + 1)]
        if h == max(window_highs):
            highs.append(SwingPoint(index=i, time=rows[i][0], price=h))
        if l == min(window_lows):
            lows.append(SwingPoint(index=i, time=rows[i][0], price=l))
    return highs, lows


def find_last_breakout(
    times: List[int],
    closes: List[float],
    snapshot: EnvSnapshot,
    buffer_price: float,
) -> BreakoutInfo | None:
    """バッファ込みで直近のブレイクアウトを検出する."""
    if snapshot.trend_dir == 0:
        return None
    for i in range(len(times) - 1, 0, -1):
        line_i = line_at(snapshot, times[i])
        line_prev = line_at(snapshot, times[i - 1])
        if snapshot.trend_dir < 0:
            if closes[i] > line_i + buffer_price and closes[i - 1] <= line_prev + buffer_price:
                return BreakoutInfo(index=i, time=times[i], direction=1)
        elif snapshot.trend_dir > 0:
            if closes[i] < line_i - buffer_price and closes[i - 1] >= line_prev - buffer_price:
                return BreakoutInfo(index=i, time=times[i], direction=-1)
    return None


def assess_timeframe(
    rows: List[Tuple[int, float, float, float, float]],
    lookback: int,
    min_slope_pips: float,
    pip_size: float,
) -> EnvSnapshot | None:
    """時間足のトレンド/チャネルを計算してスナップショットを返す.

    手順:
    - 回帰で slope_high / slope_low を算出
    - min_slope_pips を使って trend_dir を判定
    - 基準傾きを選び、切片を走査してチャネル幅を決定
    - trend_dir に応じてブレイク判定ライン(p0/p1)を選択
    """
    if not rows:
        return None
    if lookback > 0 and len(rows) > lookback:
        rows = rows[-lookback:]
    times, highs, lows, closes = extract_series(rows)
    if len(times) < 2:
        return None
    t0 = int(times[0])
    t1 = int(times[-1])
    xs = [t - t0 for t in times]
    slope_high, _ = linear_regression(xs, highs)
    slope_low, _ = linear_regression(xs, lows)

    diffs = [times[i] - times[i - 1] for i in range(1, len(times))]
    bar_sec = median(diffs)
    min_slope = 0.0
    if min_slope_pips > 0 and bar_sec > 0:
        min_slope = (min_slope_pips * pip_size) / float(bar_sec)

    trend_dir = 0
    if slope_high > min_slope and slope_low > min_slope:
        trend_dir = 1
    elif slope_high < -min_slope and slope_low < -min_slope:
        trend_dir = -1

    if trend_dir > 0:
        slope = slope_low
    elif trend_dir < 0:
        slope = slope_high
    else:
        slope = (slope_high + slope_low) / 2.0

    c_max = -1e20
    c_min = 1e20
    for i in range(len(times)):
        x_i = float(times[i] - t0)
        c_h = highs[i] - slope * x_i
        c_l = lows[i] - slope * x_i
        if c_h > c_max:
            c_max = c_h
        if c_l < c_min:
            c_min = c_l

    top_p0 = c_max
    top_p1 = c_max + slope * float(t1 - t0)
    btm_p0 = c_min
    btm_p1 = c_min + slope * float(t1 - t0)

    if trend_dir > 0:
        p0 = btm_p0
        p1 = btm_p1
    elif trend_dir < 0:
        p0 = top_p0
        p1 = top_p1
    else:
        p0 = (top_p0 + btm_p0) / 2.0
        p1 = (top_p1 + btm_p1) / 2.0

    t_last = times[-1]
    line_now = p0 + slope * float(t_last - t0)

    return EnvSnapshot(
        trend_dir=trend_dir,
        slope=slope,
        line_now=line_now,
        last_close=closes[-1],
        t0=t0,
        t1=t1,
        p0=p0,
        p1=p1,
        top_p0=top_p0,
        top_p1=top_p1,
        btm_p0=btm_p0,
        btm_p1=btm_p1,
    )
