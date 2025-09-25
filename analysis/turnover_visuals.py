#!/usr/bin/env python3
"""
Turnover visuals and execution profile utilities.

1) From a turnover grid CSV (turnover_strategy_grid_*.csv), generate:
   - Scatter: LS IR vs Avg Turnover by strategy
   - Heatmaps: For a picked strategy, pivot TopN × Cost over metric (IR/Ann)

2) From an execution timeseries CSV (pipeline_execution_*_timeseries.csv),
   generate an execution profile:
   - Daily turnover and estimated overlap tracks
   - Estimated average trades/day and holding half-life (approximation)

Notes:
 - Estimated trades/day ≈ turnover × N when TopN=BottomN=N
 - Estimated daily overlap ≈ 1 - turnover/2 (assumes long/short symmetric)
 - Holding half-life ~ ln(0.5) / ln(mean_overlap), if 0<overlap<1
"""

from __future__ import annotations

from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def scatter_ir_vs_turnover(grid_path: Path, output: Path):
    df = pd.read_csv(grid_path)
    if 'avg_turnover' not in df.columns:
        raise ValueError('Grid file lacks avg_turnover — use turnover_strategy_grid_*.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in df.groupby('strategy'):
        ax.scatter(g['avg_turnover'], g['ls_ir'], s=30, alpha=0.6, label=name)
    ax.set_xlabel('Average Turnover')
    ax.set_ylabel('LS IR')
    ax.set_title('IR vs Avg Turnover')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def heatmap_by_strategy(grid_path: Path, strategy: str, metric: str, output: Path):
    df = pd.read_csv(grid_path)
    if strategy:
        df = df[df['strategy'].str.contains(strategy, na=False)]
    if df.empty:
        raise ValueError('No rows after strategy filter')
    pivot = df.pivot_table(index='top_n', columns='cost_bps', values=metric, aggfunc='max')
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax)
    ax.set_title(f'{metric} heatmap — {strategy}')
    ax.set_xlabel('Cost (bps, decimal)')
    ax.set_ylabel('Top N')
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def execution_profile(ts_path: Path, top_n: int, bottom_n: int, output_prefix: Path):
    ts = pd.read_csv(ts_path, parse_dates=['date'])
    if ts.empty:
        raise ValueError('Empty timeseries CSV')
    # Estimated overlap and trades/day
    ts = ts.sort_values('date').reset_index(drop=True)
    ts['est_overlap'] = 1.0 - ts['turnover'] / 2.0
    n_mean = (top_n + bottom_n) / 2.0
    ts['est_trades'] = ts['turnover'] * n_mean

    # Stats
    avg_turn = float(ts['turnover'].mean())
    avg_overlap = float(ts['est_overlap'].mean())
    est_half_life = np.nan
    if 0 < avg_overlap < 1:
        est_half_life = float(np.log(0.5) / np.log(avg_overlap))

    # Plot tracks
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(ts['date'], ts['turnover'], color='#1f77b4')
    axes[0].set_title('Daily Turnover')
    axes[1].plot(ts['date'], ts['est_overlap'], color='#2ca02c')
    axes[1].set_title('Estimated Daily Overlap')
    axes[2].plot(ts['date'], ts['est_trades'], color='#ff7f0e')
    axes[2].set_title('Estimated Trades per Day')
    axes[2].set_xlabel('Date')
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + '_profile.png'), dpi=180)
    plt.close(fig)

    # Write summary JSON
    summary = {
        'avg_turnover': avg_turn,
        'avg_overlap': avg_overlap,
        'est_half_life_days': est_half_life,
        'top_n': top_n,
        'bottom_n': bottom_n,
        'source': str(ts_path)
    }
    import json
    with open(output_prefix.with_name(output_prefix.name + '_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description='Turnover visuals and execution profile')
    sub = p.add_subparsers(dest='cmd')

    p1 = sub.add_parser('grid', help='Render visuals from turnover grid CSV')
    p1.add_argument('--grid', required=True, help='turnover_strategy_grid_*.csv')
    p1.add_argument('--strategy', default='B_hysteresis', help='Strategy filter for heatmap')
    p1.add_argument('--metric', default='ls_ir', choices=['ls_ir', 'ls_ann'])

    p2 = sub.add_parser('profile', help='Render execution profile from timeseries CSV')
    p2.add_argument('--timeseries', required=True, help='pipeline_execution_*_timeseries.csv')
    p2.add_argument('--top-n', type=int, default=40)
    p2.add_argument('--bottom-n', type=int, default=40)

    args = p.parse_args()
    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)

    if args.cmd == 'grid':
        grid_path = Path(args.grid)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        scatter_ir_vs_turnover(grid_path, out_dir / f'turnover_scatter_{ts}.png')
        heatmap_by_strategy(grid_path, args.strategy, args.metric, out_dir / f'turnover_heatmap_{args.strategy}_{args.metric}_{ts}.png')
        print('Saved scatter & heatmap to data/.')
    elif args.cmd == 'profile':
        ts_path = Path(args.timeseries)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        execution_profile(ts_path, args.top_n, args.bottom_n, out_dir / f'exec_profile_{ts}')
        print('Saved execution profile to data/.')
    else:
        p.print_help()


if __name__ == '__main__':
    main()

