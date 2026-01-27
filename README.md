# Fibenix

## trendline_regression.py via Docker Compose

This repo includes a small Python script that computes a linear-regression trendline from OHLC CSV data.

### Quick start

1. Put your CSV under `data/` (see `data/sample.csv`).
2. Run:

```bash
docker compose run --rm trendline --input /data/sample.csv
```

### Optional: write output CSV

```bash
docker compose run --rm trendline --input /data/sample.csv --output /data/out.csv
```

### CSV format

Expected columns: `time,open,high,low,close` (header is optional, but supported).
