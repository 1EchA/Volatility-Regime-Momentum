#!/usr/bin/env python3
"""
Restore data directory by re-downloading raw price CSVs and regenerating
pipeline outputs. Use this when data/ has been cleaned for publishing.

Usage examples:
  python scripts/restore_data.py --limit 300
  python scripts/restore_data.py --start 20200101 --end 20241231 --limit 500 \
      --top-n 40 --cost-bps 0.0005 --run-turnover-grid
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collector import DataCollector

ROOT = ROOT
DATA = DATA


def load_universe(limit: int | None = None) -> List[str]:
    fp = ROOT / 'stock_universe.csv'
    if not fp.exists():
        raise FileNotFoundError('stock_universe.csv not found at project root')
    df = pd.read_csv(fp, dtype={'code': str})
    codes = df['code'].astype(str).str.zfill(6).tolist()
    return codes[:limit] if limit else codes


def download_prices(codes: List[str], start: str, end: str):
    dc = DataCollector(data_dir=str(DATA), cache_dir=str(ROOT / 'cache'))
    dc.download_stock_batch(codes, start_date=start, end_date=end, max_workers=5, batch_size=20)


def run_pipeline(recompute_factors: bool, recompute_regime: bool, top_n: int, bottom_n: int,
                 cost_bps: float, start_oos: str, train_window: int, alpha: float, run_turnover_grid: bool):
    cmd = [
        'python3', 'run_full_pipeline.py',
        '--start-oos', start_oos,
        '--train-window', str(train_window),
        '--alpha', str(alpha),
        '--top-n', str(top_n),
        '--bottom-n', str(bottom_n),
        '--cost-bps', str(cost_bps),
    ]
    if recompute_factors:
        cmd.append('--recompute-factors')
    if recompute_regime:
        cmd.append('--recompute-regime')
    if run_turnover_grid:
        cmd.append('--run-turnover-grid')
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description='Restore data directory with fresh artifacts')
    p.add_argument('--start', default='20200101', help='download start date YYYYMMDD')
    p.add_argument('--end', default='20241231', help='download end date YYYYMMDD')
    p.add_argument('--limit', type=int, default=500, help='limit number of stocks to download')
    p.add_argument('--start-oos', default='2022-01-01')
    p.add_argument('--train-window', type=int, default=756)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--top-n', type=int, default=40)
    p.add_argument('--bottom-n', type=int, default=40)
    p.add_argument('--cost-bps', type=float, default=0.0005)
    p.add_argument('--run-turnover-grid', action='store_true')
    args = p.parse_args()

    codes = load_universe(limit=args.limit)
    print(f'Downloading {len(codes)} stocks from {args.start} to {args.end} ...')
    download_prices(codes, start=args.start, end=args.end)
    print('Prices ready. Running pipeline ...')
    run_pipeline(
        recompute_factors=True,
        recompute_regime=True,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        cost_bps=args.cost_bps,
        start_oos=args.start_oos,
        train_window=args.train_window,
        alpha=args.alpha,
        run_turnover_grid=args.run_turnover_grid,
    )
    print('Done. Check data/ for predictions_*, pipeline_execution_* and reports.')


if __name__ == '__main__':
    main()
