#!/usr/bin/env python3
"""OHLC CSVから回帰トレンドライン/環境/トレード状態を計算する.

修正: XAUUSD向け, M1/M5/M15/M30構成に対応
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

from god_common import (
    EnvSnapshot,
    assess_timeframe,
    compute_regression,
    extract_series,
    find_last_breakout,
    find_swings,
    line_at,
    load_ohlc_rows,
    parse_rows,
    pip_size_for_symbol,
)


def compute_trade_signal(
    rows_main: List[Tuple[int, float, float, float, float]],
    snap_main: EnvSnapshot,
    snap_f1: EnvSnapshot | None,
    snap_f2: EnvSnapshot | None,
    snap_f3: EnvSnapshot | None,
    pip_size: float,
    breakout_buffer_pips: float,
    fast_retrace_bars: int,
    max_retrace_bars: int,
    swing_depth: int,
) -> dict | None:
    """
    メイン足(M1)のブレイクアウトと上位足(M5, M15, M30)フィルタからシグナルを作る.
    snap_main: Entry timeframe (e.g. M1)
    snap_f1: Primary filter (e.g. M5) - Direction & Breakout check
    snap_f2: Secondary filter (e.g. M15) - Trend check
    snap_f3: Tertiary filter (e.g. M30) - Trend check
    """
    if not rows_main or snap_main.trend_dir == 0:
        return None
    times, _highs, _lows, closes = extract_series(rows_main)
    buffer_price = breakout_buffer_pips * pip_size
    breakout = find_last_breakout(times, closes, snap_main, buffer_price)
    if not breakout:
        return None

    # F1 (M5) フィルター: トレンド方向と、もしF1もブレイクしているならその方向に従う
    f1_breakout_long = False
    f1_breakout_short = False
    if snap_f1:
        if snap_f1.trend_dir < 0 and snap_f1.last_close > snap_f1.line_now + buffer_price:
            f1_breakout_long = True
        if snap_f1.trend_dir > 0 and snap_f1.last_close < snap_f1.line_now - buffer_price:
            f1_breakout_short = True
    
    # フィルタールール:
    # ブレイク方向と逆のF1ブレイクが発生していないか、あるいはトレンド方向と合致するか
    # 簡略化: メイン足のブレイク方向に対し、F1が逆行する強いシグナル(逆ブレイク)が出ていないこと、
    # かつ上位足トレンドが逆向いていないことなどを確認する。
    
    # ここでは元のロジックを踏襲: 
    # 「上位足(F1)で逆方向へのブレイクが起きていないこと」を確認したいが、
    # 元ロジック: if breakout.direction > 0 and not d1_breakout_long: return None
    # つまり「上位足(D1/M5)も同じ方向にブレイク(またはトレンド回帰)している必要がある」という強いフィルタ。
    # XAUUSDの短期足だと厳しすぎる可能性があるため、
    # 「上位足(F1)のトレンドが逆でない、またはブレイクしている」程度に緩和するか、元のままにするか。
    # 依頼は「ロジックそのまま」なので、元の厳しめのフィルタを維持します。
    
    if breakout.direction > 0 and not f1_breakout_long:
        return None
    if breakout.direction < 0 and not f1_breakout_short:
        return None

    # F2 (M15), F3 (M30) フィルター: 単純なトレンド方向一致チェック
    # 逆トレンドならエントリーしない
    if snap_f2 and snap_f2.trend_dir == -1 and breakout.direction > 0:
        return None
    if snap_f2 and snap_f2.trend_dir == 1 and breakout.direction < 0:
        return None
    if snap_f3 and snap_f3.trend_dir == -1 and breakout.direction > 0:
        return None
    if snap_f3 and snap_f3.trend_dir == 1 and breakout.direction < 0:
        return None

    # スイングポイント計算 (リトレース基準)
    highs, lows = find_swings(rows_main, swing_depth)
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

    bars_since_breakout = (len(rows_main) - 1) - breakout.index
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


def run_env_mode(args: argparse.Namespace) -> int:
    """環境/トレンドラインCSVを生成する."""
    if not args.env_dir or not args.symbols:
        print("env mode requires --env-dir and --symbols", file=sys.stderr)
        return 2
    env_dir = Path(args.env_dir)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    if not symbols or not timeframes:
        print("no symbols/timeframes", file=sys.stderr)
        return 2

    # 出力用バッファ
    out_rows: List[List[str]] = []
    line_rows: List[List[str]] = []
    order_rows: List[List[str]] = []

    for symbol in symbols:
        pip = pip_size_for_symbol(symbol)
        tf_snap: dict[str, EnvSnapshot] = {}
        tf_rows: dict[str, List] = {}

        # 全時間足の読み込みと解析
        for tf in timeframes:
            path = env_dir / f"{args.file_prefix}{symbol}_{tf}.csv"
            if not path.exists():
                continue
            rows = load_ohlc_rows(str(path))
            tf_rows[tf] = rows
            snap = assess_timeframe(rows, args.lookback, args.min_slope_pips, pip)
            if snap:
                tf_snap[tf] = snap
        
        # 時間足のマッピング (M1, M5, M15, M30)
        # ユーザー指定順序が M1, M5, M15, M30 であると仮定
        # timeframesリストのインデックスまたは名前で取得
        
        snap_main = tf_snap.get("M1") or tf_snap.get("H4") # H4は後方互換用
        snap_f1 = tf_snap.get("M5") or tf_snap.get("D1")
        snap_f2 = tf_snap.get("M15") or tf_snap.get("W1")
        snap_f3 = tf_snap.get("M30") or tf_snap.get("MN")

        # F1 (M5) の状態判定
        f1_breakout_long = False
        f1_breakout_short = False
        if snap_f1:
            buffer_price = args.breakout_buffer_pips * pip
            if snap_f1.trend_dir < 0 and snap_f1.last_close > snap_f1.line_now + buffer_price:
                f1_breakout_long = True
            if snap_f1.trend_dir > 0 and snap_f1.last_close < snap_f1.line_now - buffer_price:
                f1_breakout_short = True

        allow_long = bool(f1_breakout_long)
        allow_short = bool(f1_breakout_short)
        
        # 上位足トレンドフィルター
        if snap_f2 and snap_f2.trend_dir == -1:
            allow_long = False
        if snap_f2 and snap_f2.trend_dir == 1:
            allow_short = False
        if snap_f3 and snap_f3.trend_dir == -1:
            allow_long = False
        if snap_f3 and snap_f3.trend_dir == 1:
            allow_short = False

        # 環境認識ステータス出力 (CSVヘッダは d1_... だが中身は f1/f2/f3)
        out_rows.append([
            symbol,
            str(args.now_ts),
            "1" if allow_long else "0",
            "1" if allow_short else "0",
            str(snap_f1.trend_dir if snap_f1 else 0),
            str(snap_f2.trend_dir if snap_f2 else 0),
            str(snap_f3.trend_dir if snap_f3 else 0),
            "1" if f1_breakout_long else "0",
            "1" if f1_breakout_short else "0",
            f"{snap_f1.line_now:.6f}" if snap_f1 else "",
            f"{snap_f1.last_close:.6f}" if snap_f1 else "",
        ])

        # ライン描画用データ出力
        for tf, snap in tf_snap.items():
            breakout_line = ""
            breakout_long = "0"
            breakout_short = "0"
            
            # F1 (M5) の場合のみブレイクライン情報を付与
            if tf == "M5" or tf == "D1":
                if f1_breakout_long:
                    breakout_long = "1"
                if f1_breakout_short:
                    breakout_short = "1"
                if snap.trend_dir < 0:
                    breakout_line = f"{snap.line_now + (args.breakout_buffer_pips * pip):.6f}"
                elif snap.trend_dir > 0:
                    breakout_line = f"{snap.line_now - (args.breakout_buffer_pips * pip):.6f}"
            
            bo_time = ""
            bo_price = ""
            bo_dir = "0"
            we_time = ""
            we_price = ""
            
            if tf in tf_rows:
                rows = tf_rows[tf]
                if args.lookback > 0 and len(rows) > args.lookback:
                    rows = rows[-args.lookback:]
                times, _, _, closes = extract_series(rows)
                buffer_p = args.breakout_buffer_pips * pip
                bo = find_last_breakout(times, closes, snap, buffer_p)
                if bo:
                    bo_time = str(bo.time)
                    price_at = line_at(snap, bo.time)
                    bo_price = f"{price_at:.6f}"
                    bo_dir = str(bo.direction)
                    
                    # 波の終点探索
                    search_slice = rows[bo.index:] 
                    if bo.direction > 0:
                        max_h = -1.0
                        max_t = 0
                        for r in search_slice:
                            if r[2] > max_h:
                                max_h = r[2]
                                max_t = r[0]
                        if max_t > 0:
                            we_time = str(max_t)
                            we_price = f"{max_h:.6f}"
                    elif bo.direction < 0:
                        min_l = 1e20
                        min_t = 0
                        for r in search_slice:
                            if r[3] < min_l:
                                min_l = r[3]
                                min_t = r[0]
                        if min_t > 0:
                            we_time = str(min_t)
                            we_price = f"{min_l:.6f}"

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
                f"{snap.top_p0:.6f}",
                f"{snap.top_p1:.6f}",
                f"{snap.btm_p0:.6f}",
                f"{snap.btm_p1:.6f}",
                bo_time,
                bo_price,
                bo_dir,
                we_time,
                we_price
            ])

        # トレードシグナル生成 (Main: M1)
        if args.out_orders and snap_main:
            rows_main = tf_rows.get("M1") or tf_rows.get("H4", [])
            if args.lookback > 0 and len(rows_main) > args.lookback:
                rows_main = rows_main[-args.lookback:]
            
            signal = compute_trade_signal(
                rows_main=rows_main,
                snap_main=snap_main,
                snap_f1=snap_f1,
                snap_f2=snap_f2,
                snap_f3=snap_f3,
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
                "f1_trend", "f2_trend", "f3_trend",
                "f1_breakout_long", "f1_breakout_short",
                "f1_line", "f1_last",
            ])
            writer.writerows(out_rows)
    else:
        # stdout出力用
        writer = csv.writer(sys.stdout)
        writer.writerow([
            "symbol", "ts", "allow_long", "allow_short",
            "f1_trend", "f2_trend", "f3_trend",
            "f1_breakout_long", "f1_breakout_short",
            "f1_line", "f1_last",
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
                "top_p0", "top_p1", "btm_p0", "btm_p1",
                "bo_time", "bo_price", "bo_dir",
                "we_time", "we_price"
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
    """単一CSVの回帰ラインを算出する."""
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
    """CLIエントリポイント."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="CSV with columns: time,open,high,low,close")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--start-time", type=int, help="Use rows with time >= start_time")
    parser.add_argument("--env-dir", help="Directory containing per-symbol OHLC CSVs")
    parser.add_argument("--symbols", default="XAUUSD", help="Comma-separated symbol list")
    # デフォルト時間足: M1, M5, M15, M30
    parser.add_argument("--timeframes", default="M1,M5,M15,M30", help="Timeframes to read")
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
