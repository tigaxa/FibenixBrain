#!/usr/bin/env python3
"""Compute a linear-regression trendline or environment/trade state from OHLC CSV data."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from common import (
    compute_regression,
    extract_series,
    linear_regression,
    load_ohlc_rows,
    median,
    parse_rows,
    pip_size_for_symbol,
)

@dataclass
class EnvSnapshot:
    trend_dir: int
    slope: float
    line_now: float
    last_close: float
    t0: int
    t1: int
    p0: float
    p1: float
    # 追加: チャネルライン（トレンドラインの対になるライン）の座標
    ch_p0: float
    ch_p1: float


@dataclass
class SwingPoint:
    index: int
    time: int
    price: float


@dataclass
class BreakoutInfo:
    index: int
    time: int
    direction: int


def line_at(snapshot: EnvSnapshot, t: int) -> float:
    return snapshot.p0 + snapshot.slope * float(t - snapshot.t0)


def find_swings(
    rows: List[Tuple[int, float, float, float, float]], depth: int
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
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
    # 現在のトレンドライン（上昇なら下限、下降なら上限）からのブレイクを判定
    if snapshot.trend_dir == 0:
        return None
    for i in range(len(times) - 1, 0, -1):
        line_i = line_at(snapshot, times[i])
        line_prev = line_at(snapshot, times[i - 1])
        if snapshot.trend_dir < 0:
            # 下降トレンド：上限ライン(Resistance)を上に抜けたらブレイク
            if closes[i] > line_i + buffer_price and closes[i - 1] <= line_prev + buffer_price:
                return BreakoutInfo(index=i, time=times[i], direction=1)
        elif snapshot.trend_dir > 0:
            # 上昇トレンド：下限ライン(Support)を下に抜けたらブレイク
            if closes[i] < line_i - buffer_price and closes[i - 1] >= line_prev - buffer_price:
                return BreakoutInfo(index=i, time=times[i], direction=-1)
    return None


def compute_trade_signal(
    rows_h4: List[Tuple[int, float, float, float, float]],
    snap_h4: EnvSnapshot,
    snap_d1: EnvSnapshot | None,
    snap_w1: EnvSnapshot | None,
    snap_mn: EnvSnapshot | None,
    pip_size: float,
    breakout_buffer_pips: float,
    fast_retrace_bars: int,
    max_retrace_bars: int,
    swing_depth: int,
) -> dict | None:
    if not rows_h4 or snap_h4.trend_dir == 0:
        return None
    times, _highs, _lows, closes = extract_series(rows_h4)
    buffer_price = breakout_buffer_pips * pip_size
    breakout = find_last_breakout(times, closes, snap_h4, buffer_price)
    if not breakout:
        return None

    # D1のブレイク確認
    d1_breakout_long = False
    d1_breakout_short = False
    if snap_d1:
        if snap_d1.trend_dir < 0 and snap_d1.last_close > snap_d1.line_now + buffer_price:
            d1_breakout_long = True
        if snap_d1.trend_dir > 0 and snap_d1.last_close < snap_d1.line_now - buffer_price:
            d1_breakout_short = True
    if breakout.direction > 0 and not d1_breakout_long:
        return None
    if breakout.direction < 0 and not d1_breakout_short:
        return None

    # 週足/月足フィルター
    if snap_w1 and snap_w1.trend_dir == -1 and breakout.direction > 0:
        return None
    if snap_w1 and snap_w1.trend_dir == 1 and breakout.direction < 0:
        return None
    if snap_mn and snap_mn.trend_dir == -1 and breakout.direction > 0:
        return None
    if snap_mn and snap_mn.trend_dir == 1 and breakout.direction < 0:
        return None

    highs, lows = find_swings(rows_h4, swing_depth)
    start = None
    if breakout.direction > 0:
        for sp in reversed(lows):
            if sp.index < breakout.index:
                start = sp
                break
    else:
        for sp in reversed(highs):
            if sp.index < breakout.index:
                start = sp
                break
    if not start:
        return None

    wave_end = None
    if breakout.direction > 0:
        for sp in highs:
            if sp.index > breakout.index:
                wave_end = sp
                break
    else:
        for sp in lows:
            if sp.index > breakout.index:
                wave_end = sp
                break
    if not wave_end:
        return None

    wave_size = abs(wave_end.price - start.price)
    if wave_size <= 0:
        return None

    bars_since_breakout = (len(rows_h4) - 1) - breakout.index
    if bars_since_breakout > max_retrace_bars:
        return None
    level = 0.618 if bars_since_breakout <= fast_retrace_bars else 0.382

    current_price = closes[-1]
    stop_edge = 5 * pip_size
    if breakout.direction > 0:
        if current_price <= start.price - stop_edge:
            return None
        entry_price = wave_end.price - wave_size * level
        if current_price > entry_price:
            return None
        sl_price = wave_end.price - wave_size * (0.309 if level > 0.5 else 0.118)
        tp_price = wave_end.price + wave_size * 0.618
    else:
        if current_price >= start.price + stop_edge:
            return None
        entry_price = wave_end.price + wave_size * level
        if current_price < entry_price:
            return None
        sl_price = wave_end.price + wave_size * (0.309 if level > 0.5 else 0.118)
        tp_price = wave_end.price - wave_size * 0.618

    order_id = breakout.time * 10 + (1 if breakout.direction > 0 else 2)
    return {
        "id": order_id,
        "direction": breakout.direction,
        "entry_price": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "breakout_time": breakout.time,
    }


def assess_timeframe(rows: List[Tuple[int, float, float, float, float]], lookback: int,
                     min_slope_pips: float, pip_size: float) -> EnvSnapshot | None:
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
    slope_high, intercept_high = linear_regression(xs, highs)
    slope_low, intercept_low = linear_regression(xs, lows)
    
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
    
    # ----------------------------------------------------
    # トレンドラインとチャネルラインの選定
    # ----------------------------------------------------
    slope = 0.0
    intercept = 0.0
    ch_slope = 0.0
    ch_intercept = 0.0

    if trend_dir > 0:
        # 上昇トレンド
        # トレンドライン（主） = 安値（下側）
        # チャネルライン（対） = 高値（上側）
        slope = slope_low
        intercept = intercept_low
        ch_slope = slope_high
        ch_intercept = intercept_high
    elif trend_dir < 0:
        # 下降トレンド
        # トレンドライン（主） = 高値（上側）
        # チャネルライン（対） = 安値（下側）
        slope = slope_high
        intercept = intercept_high
        ch_slope = slope_low
        ch_intercept = intercept_low
    else:
        # レンジ：主ラインは平均、チャネルは高値（便宜上）
        slope = (slope_high + slope_low) / 2.0
        intercept = (intercept_high + intercept_low) / 2.0
        ch_slope = slope_high
        ch_intercept = intercept_high

    t_last = times[-1]
    line_now = intercept + slope * float(t_last - t0)
    
    # 主トレンドライン座標
    p0 = intercept
    p1 = intercept + slope * float(t1 - t0)
    
    # チャネルライン座標
    ch_p0 = ch_intercept
    ch_p1 = ch_intercept + ch_slope * float(t1 - t0)

    return EnvSnapshot(
        trend_dir=trend_dir,
        slope=slope,
        line_now=line_now,
        last_close=closes[-1],
        t0=t0,
        t1=t1,
        p0=p0,
        p1=p1,
        ch_p0=ch_p0,
        ch_p1=ch_p1,
    )


def run_env_mode(args: argparse.Namespace) -> int:
    if not args.env_dir or not args.symbols:
        print("env mode requires --env-dir and --symbols", file=sys.stderr)
        return 2
    env_dir = Path(args.env_dir)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    if not symbols or not timeframes:
        print("no symbols/timeframes", file=sys.stderr)
        return 2

    out_rows: List[List[str]] = []
    line_rows: List[List[str]] = []
    order_rows: List[List[str]] = []
    for symbol in symbols:
        pip = pip_size_for_symbol(symbol)
        tf_snap: dict[str, EnvSnapshot] = {}
        for tf in timeframes:
            path = env_dir / f"{args.file_prefix}{symbol}_{tf}.csv"
            if not path.exists():
                continue
            rows = load_ohlc_rows(str(path))
            snap = assess_timeframe(rows, args.lookback, args.min_slope_pips, pip)
            if snap:
                tf_snap[tf] = snap
        h4 = tf_snap.get("H4")
        d1 = tf_snap.get("D1")
        w1 = tf_snap.get("W1")
        mn = tf_snap.get("MN")

        d1_breakout_long = False
        d1_breakout_short = False
        if d1:
            buffer_price = args.breakout_buffer_pips * pip
            if d1.trend_dir < 0 and d1.last_close > d1.line_now + buffer_price:
                d1_breakout_long = True
            if d1.trend_dir > 0 and d1.last_close < d1.line_now - buffer_price:
                d1_breakout_short = True

        allow_long = bool(d1_breakout_long)
        allow_short = bool(d1_breakout_short)
        if w1 and w1.trend_dir == -1:
            allow_long = False
        if w1 and w1.trend_dir == 1:
            allow_short = False
        if mn and mn.trend_dir == -1:
            allow_long = False
        if mn and mn.trend_dir == 1:
            allow_short = False

        out_rows.append([
            symbol,
            str(args.now_ts),
            "1" if allow_long else "0",
            "1" if allow_short else "0",
            str(d1.trend_dir if d1 else 0),
            str(w1.trend_dir if w1 else 0),
            str(mn.trend_dir if mn else 0),
            "1" if d1_breakout_long else "0",
            "1" if d1_breakout_short else "0",
            f"{d1.line_now:.6f}" if d1 else "",
            f"{d1.last_close:.6f}" if d1 else "",
        ])

        for tf, snap in tf_snap.items():
            breakout_line = ""
            breakout_long = "0"
            breakout_short = "0"
            if tf == "D1":
                if d1_breakout_long:
                    breakout_long = "1"
                if d1_breakout_short:
                    breakout_short = "1"
                if snap.trend_dir < 0:
                    breakout_line = f"{snap.line_now + (args.breakout_buffer_pips * pip):.6f}"
                elif snap.trend_dir > 0:
                    breakout_line = f"{snap.line_now - (args.breakout_buffer_pips * pip):.6f}"
            line_rows.append([
                symbol,
                tf,
                str(snap.t0),
                f"{snap.p0:.6f}",
                str(snap.t1),
                f"{snap.p1:.6f}",
                str(snap.trend_dir),
                f"{snap.line_now:.6f}",
                f"{snap.last_close:.6f}",
                breakout_long,
                breakout_short,
                breakout_line,
                f"{snap.ch_p0:.6f}", # チャネルライン始点
                f"{snap.ch_p1:.6f}", # チャネルライン終点
            ])

        if args.out_orders and h4:
            path_h4 = env_dir / f"{args.file_prefix}{symbol}_H4.csv"
            rows_h4 = load_ohlc_rows(str(path_h4)) if path_h4.exists() else []
            if args.lookback > 0 and len(rows_h4) > args.lookback:
                rows_h4 = rows_h4[-args.lookback:]
            signal = compute_trade_signal(
                rows_h4=rows_h4,
                snap_h4=h4,
                snap_d1=d1,
                snap_w1=w1,
                snap_mn=mn,
                pip_size=pip,
                breakout_buffer_pips=args.breakout_buffer_pips,
                fast_retrace_bars=args.fast_retrace_bars,
                max_retrace_bars=args.max_retrace_bars,
                swing_depth=args.swing_depth,
            )
            if signal:
                side = "BUY" if signal["direction"] > 0 else "SELL"
                order_rows.append([
                    str(signal["id"]),
                    symbol,
                    side,
                    f"{args.order_lots:.2f}",
                    f"{signal['sl']:.6f}",
                    f"{signal['tp']:.6f}",
                    str(args.order_magic),
                    args.order_comment,
                    str(args.now_ts),
                ])

    if args.out_env:
        out_path = Path(args.out_env)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "symbol", "ts", "allow_long", "allow_short",
                "d1_trend", "w1_trend", "mn_trend",
                "d1_breakout_long", "d1_breakout_short",
                "d1_line", "d1_last",
            ])
            writer.writerows(out_rows)
    else:
        writer = csv.writer(sys.stdout)
        writer.writerow([
            "symbol", "ts", "allow_long", "allow_short",
            "d1_trend", "w1_trend", "mn_trend",
            "d1_breakout_long", "d1_breakout_short",
            "d1_line", "d1_last",
        ])
        writer.writerows(out_rows)

    if args.out_lines:
        out_path = Path(args.out_lines)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "symbol", "tf", "t1", "p1", "t2", "p2",
                "trend_dir", "line_now", "last_close",
                "breakout_long", "breakout_short", "breakout_line",
                "ch_p1", "ch_p2",
            ])
            writer.writerows(line_rows)
    if args.out_orders:
        out_path = Path(args.out_orders)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        order_rows.sort(key=lambda r: int(r[0]))
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "symbol", "side", "lots", "sl", "tp", "magic", "comment", "ts",
            ])
            writer.writerows(order_rows)
    return 0


def run_regression_mode(args: argparse.Namespace) -> int:
    times, prices = parse_rows(args.input)
    if not times:
        print("no data", file=sys.stderr)
        return 1

    if args.start_time is not None:
        filtered = [(t, p) for t, p in zip(times, prices) if t >= args.start_time]
        if not filtered:
            print("no data after start_time", file=sys.stderr)
            return 1
        times, prices = zip(*filtered)
        times = list(times)
        prices = list(prices)

    slope, intercept, t0, t1 = compute_regression(times, prices)
    p0 = intercept
    p1 = intercept + slope * (t1 - t0)

    out = [t0, f"{p0:.8f}", t1, f"{p1:.8f}", f"{slope:.12f}", f"{intercept:.8f}", len(times)]

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t1", "p1", "t2", "p2", "slope", "intercept", "n"])
            writer.writerow(out)
    else:
        print(",".join(str(x) for x in out))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="CSV with columns: time,open,high,low,close")
    parser.add_argument("--output", help="Output CSV path (t1,p1,t2,p2,slope,intercept,n)")
    parser.add_argument("--start-time", type=int, help="Use rows with time >= start_time")
    parser.add_argument("--env-dir", help="Directory containing per-symbol OHLC CSVs")
    parser.add_argument("--symbols", help="Comma-separated symbol list")
    parser.add_argument("--timeframes", default="H4,D1,W1,MN", help="Timeframes to read")
    parser.add_argument("--file-prefix", default="fibenix_prices_", help="Input CSV filename prefix")
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--min-slope-pips", type=float, default=0.0)
    parser.add_argument("--breakout-buffer-pips", type=float, default=2.0)
    parser.add_argument("--out-env", help="Output env CSV path")
    parser.add_argument("--out-lines", help="Output trendline CSV path")
    parser.add_argument("--out-orders", help="Output orders CSV path")
    parser.add_argument("--order-lots", type=float, default=0.1)
    parser.add_argument("--order-magic", type=int, default=20250101)
    parser.add_argument("--order-comment", default="FIBENIX_PY")
    parser.add_argument("--fast-retrace-bars", type=int, default=6)
    parser.add_argument("--max-retrace-bars", type=int, default=40)
    parser.add_argument("--swing-depth", type=int, default=12)
    parser.add_argument("--now-ts", type=int, default=0, help="Unix timestamp to write (0 = auto)")
    args = parser.parse_args()

    if args.now_ts == 0:
        args.now_ts = int(__import__("time").time())

    if args.env_dir:
        return run_env_mode(args)

    if not args.input:
        print("either --input or --env-dir is required", file=sys.stderr)
        return 2

    return run_regression_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
