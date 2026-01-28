#!/bin/bash

# コンテナ内のデータマウントポイント
DATA_DIR="/app/mt5_data"
# 出力ファイル名（MQL5側の定数定義と一致させる）
LINE_FILE="fibenix_env_lines.csv"
ORDER_FILE="fibenix_orders.csv"

echo "=== FIBENIX BRAIN STARTED ==="
echo "Watching: $DATA_DIR"

while true; do
    # 現在時刻
    date "+%Y-%m-%d %H:%M:%S"

    # Pythonスクリプト実行
    # MQL5が書き出した fibenix_prices_*.csv を読み込み、ラインとオーダーを書き出す
    python3 trendline_regression.py \
        --env-dir "$DATA_DIR" \
        --file-prefix "fibenix_prices_" \
        --out-env "$DATA_DIR/fibenix_env_status.csv" \
        --out-lines "$DATA_DIR/$LINE_FILE" \
        --out-orders "$DATA_DIR/$ORDER_FILE" \
        --symbols "USDJPY,EURUSD,AUDUSD,GBPUSD,GBPJPY,EURJPY,AUDJPY,GBPAUD,EURAUD" \
        --timeframes "H4,D1,W1,MN" \
        --lookback 600 \
        --min-slope-pips 0.5 \
        --breakout-buffer-pips 2.0

    # 次の計算まで待機（EAのPoll秒数に合わせて調整。ここでは10秒）
    echo "Waiting for next cycle..."
    sleep 10
done
