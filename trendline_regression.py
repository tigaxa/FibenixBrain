#!/usr/bin/env python3
"""Compute a linear-regression trendline from OHLC CSV data."""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Iterable, List, Tuple


def parse_rows(path: str) -> Tuple[List[int], List[float]]:
    times: List[int] = []
    prices: List[float] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # ヘッダなしCSVの既定の列位置。
        idx_time = 0
        idx_open = 1
        idx_high = 2
        idx_low = 3
        idx_close = 4
        rows: Iterable[List[str]]
        if header and any(h.lower() in {"time", "datetime", "timestamp"} for h in header):
            # ヘッダ名がある場合は列インデックスを解決する。
            lower = [h.lower() for h in header]
            idx_time = lower.index("time") if "time" in lower else 0
            idx_open = lower.index("open") if "open" in lower else 1
            idx_high = lower.index("high") if "high" in lower else 2
            idx_low = lower.index("low") if "low" in lower else 3
            idx_close = lower.index("close") if "close" in lower else 4
            rows = reader
        else:
            # ヘッダに見えない場合は1行目もデータとして扱う。
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
            # OHLC平均を代表価格として使う。
            times.append(t)
            prices.append((o + h + l + c) / 4.0)
    return times, prices


def linear_regression(xs: Iterable[float], ys: Iterable[float]) -> Tuple[float, float]:
    xs = list(xs)
    ys = list(ys)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 points")
    # y = slope * x + intercept の最小二乗の閉形式。
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with columns: time,open,high,low,close")
    parser.add_argument("--output", help="Output CSV path (t1,p1,t2,p2,slope,intercept,n)")
    parser.add_argument("--start-time", type=int, help="Use rows with time >= start_time")
    args = parser.parse_args()

    times, prices = parse_rows(args.input)
    if not times:
        print("no data", file=sys.stderr)
        return 1

    if args.start_time is not None:
        # start_time以上のみを対象にする。
        filtered = [(t, p) for t, p in zip(times, prices) if t >= args.start_time]
        if not filtered:
            print("no data after start_time", file=sys.stderr)
            return 1
        times, prices = zip(*filtered)
        times = list(times)
        prices = list(prices)

    t0 = int(times[0])
    t1 = int(times[-1])
    # 数値安定性のため、時刻差分を使って回帰する。
    xs = [t - t0 for t in times]
    slope, intercept = linear_regression(xs, prices)
    p0 = intercept
    p1 = intercept + slope * (t1 - t0)

    # 出力: トレンドライン端点 + 回帰パラメータ + サンプル数。
    out = [t0, f"{p0:.8f}", t1, f"{p1:.8f}", f"{slope:.12f}", f"{intercept:.8f}", len(times)]

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t1", "p1", "t2", "p2", "slope", "intercept", "n"])
            writer.writerow(out)
    else:
        print(",".join(str(x) for x in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
