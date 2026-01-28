#!/bin/bash

# コンテナ内のデータマウントポイント
DATA_DIR="/app/mt5_data"

# --- 設定: 為替用 (既存) ---
LINE_FILE="fibenix_env_lines.csv"
ORDER_FILE="fibenix_orders.csv"
ENV_FILE="fibenix_env_status.csv"

# --- 設定: Gold用 (新規) ---
# ファイル名が被らないように _gold を付与
GOLD_LINE_FILE="fibenix_gold_lines.csv"
GOLD_ORDER_FILE="fibenix_gold_orders.csv"
GOLD_ENV_FILE="fibenix_gold_status.csv"

echo "=== FIBENIX BRAIN (MULTI-ASSET) STARTED ==="
echo "Watching: $DATA_DIR"

while true; do
    # 現在時刻
    TZ=Asia/Tokyo date "+%Y-%m-%d %H:%M:%S"

    # ---------------------------------------------------------
    # 1. 既存: 為替ペア用 (H4/D1/W1/MN)
    # ---------------------------------------------------------
    echo "[Currency] Running logic..."
    python3 trendline_regression.py \
        --env-dir "$DATA_DIR" \
        --file-prefix "fibenix_prices_" \
        --out-env "$DATA_DIR/$ENV_FILE" \
        --out-lines "$DATA_DIR/$LINE_FILE" \
        --out-orders "$DATA_DIR/$ORDER_FILE" \
        --symbols "USDJPY,EURUSD,AUDUSD,GBPUSD,GBPJPY,EURJPY,AUDJPY,GBPAUD,EURAUD" \
        --timeframes "H4,D1,W1,MN" \
        --lookback 600 \
        --min-slope-pips 0.5 \
        --breakout-buffer-pips 2.0

    # ---------------------------------------------------------
    # 2. 新規: Gold用 (M1/M5/M15/M30)
    # ---------------------------------------------------------
    # ※ god_common.py というファイル名なら、import文も合わせる必要があります
    echo "[Gold] Running logic..."
    python3 god_trendline_regression.py \
        --env-dir "$DATA_DIR" \
        --file-prefix "fibenix_prices_" \
        --out-env "$DATA_DIR/$GOLD_ENV_FILE" \
        --out-lines "$DATA_DIR/$GOLD_LINE_FILE" \
        --out-orders "$DATA_DIR/$GOLD_ORDER_FILE" \
        --symbols "XAUUSD" \
        --timeframes "M1,M5,M15,M30" \
        --lookback 600 \
        --min-slope-pips 0.5 \
        --breakout-buffer-pips 2.0 \
        --fast-retrace-bars 6 \
        --max-retrace-bars 40

    # 次の計算まで待機
    echo "Waiting for next cycle..."
    sleep 10
done
