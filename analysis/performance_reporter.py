#!/usr/bin/env python3
"""
Performance analytics and visualisation helper.

Generates:
1. Portfolio time-series statistics (equity curve, drawdown, turnover) from a
   predictions CSV produced by predictive_model.
2. Regime-level contribution analysis by joining with a regime-labelled data set.
3. Sensitivity heatmaps from the cost grid search output.

Aligns with execution tasks 2.1-2.3 of the project plan.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')


def load_predictions(pred_path: Path) -> pd.DataFrame:
    if not pred_path.exists():
        raise FileNotFoundError(f'Predictions file missing: {pred_path}')
    df = pd.read_csv(pred_path)
    df['date'] = pd.to_datetime(df['date'])
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    return df.sort_values(['date', 'y_pred'], ascending=[True, False])


def compute_portfolio_timeseries(pred_df: pd.DataFrame,
                                top_n: int,
                                bottom_n: int,
                                cost_bps: float) -> pd.DataFrame:
    rows = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    for date, group in pred_df.groupby('date'):
        g = group.sort_values('y_pred', ascending=False)
        longs = g.head(top_n)
        shorts = g.tail(bottom_n)
        if longs.empty or shorts.empty:
            continue
        long_ret = longs['y_true'].mean()
        short_ret = shorts['y_true'].mean()
        curr_long_codes = set(longs['stock_code'])
        curr_short_codes = set(shorts['stock_code'])
        overlap_long = len(prev_long & curr_long_codes) / max(1, len(curr_long_codes))
        overlap_short = len(prev_short & curr_short_codes) / max(1, len(curr_short_codes))
        turnover = (1 - overlap_long) + (1 - overlap_short)
        ls_gross = long_ret - short_ret
        ls_net = ls_gross - cost_bps * turnover
        rows.append({
            'date': date,
            'long': long_ret,
            'short': short_ret,
            'ls_gross': ls_gross,
            'turnover': turnover,
            'cost_bps': cost_bps,
            'ls_net': ls_net,
        })
        prev_long, prev_short = curr_long_codes, curr_short_codes
    ts = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    ts['cum_ls_net'] = ts['ls_net'].cumsum()
    ts['cum_ls_gross'] = ts['ls_gross'].cumsum()
    peak = ts['cum_ls_net'].cummax()
    ts['drawdown'] = ts['cum_ls_net'] - peak
    return ts


def compute_summary_metrics(ts: pd.DataFrame, ic_series: pd.Series) -> dict:
    metrics = {
        'n_days': len(ts),
        'ic_mean': float(ic_series.mean()),
        'ic_ir': float(ic_series.mean() / (ic_series.std(ddof=1) + 1e-12)),
        'ic_win_rate': float((ic_series > 0).mean()),
        'long_mean': float(ts['long'].mean()),
        'short_mean': float(ts['short'].mean()),
        'ls_mean': float(ts['ls_net'].mean()),
        'ls_ann': float(ts['ls_net'].mean() * 252),
        'ls_ir': float((ts['ls_net'].mean() / (ts['ls_net'].std(ddof=1) + 1e-12)) * np.sqrt(252)),
        'ls_win_rate': float((ts['ls_net'] > 0).mean()),
        'avg_turnover': float(ts['turnover'].mean()),
        'max_drawdown': float(ts['drawdown'].min()),
    }
    return metrics


def compute_ic_series(pred_df: pd.DataFrame) -> pd.Series:
    ic = pred_df.groupby('date').apply(lambda g: g['y_pred'].corr(g['y_true'], method='spearman'))
    return ic.dropna()


def plot_equity_curve(ts: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts['date'], ts['cum_ls_net'], label='Cumulative LS (net)', color='#1f77b4')
    ax.plot(ts['date'], ts['cum_ls_gross'], label='Cumulative LS (gross)', color='#ff7f0e', linestyle='--')
    ax.set_title('Strategy Equity Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_drawdown(ts: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(ts['date'], ts['drawdown'], color='#d62728', alpha=0.5)
    ax.set_title('Drawdown (net performance)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def compute_regime_contributions(pred_df: pd.DataFrame, regime_df: pd.DataFrame,
                                 top_n: int, bottom_n: int, cost_bps: float) -> pd.DataFrame:
    merged = pred_df.merge(regime_df[['date', 'regime']], on='date', how='left')
    rows = []
    for regime, df_reg in merged.groupby('regime'):
        if regime in (None, '未分类'):
            continue
        ic = df_reg.groupby('date').apply(lambda g: g['y_pred'].corr(g['y_true'], method='spearman')).dropna()
        ts = compute_portfolio_timeseries(df_reg, top_n, bottom_n, cost_bps)
        if ts.empty:
            continue
        rows.append({
            'regime': regime,
            'ic_mean': ic.mean(),
            'ic_ir': ic.mean() / (ic.std(ddof=1) + 1e-12),
            'ls_mean': ts['ls_net'].mean(),
            'ls_ann': ts['ls_net'].mean() * 252,
            'ls_ir': (ts['ls_net'].mean() / (ts['ls_net'].std(ddof=1) + 1e-12)) * np.sqrt(252),
            'weight_days': len(ts),
        })
    return pd.DataFrame(rows)


def plot_heatmaps(grid_path: Path, metric: str, output_prefix: Path):
    if not grid_path.exists():
        raise FileNotFoundError(f'Grid search csv not found: {grid_path}')
    grid = pd.read_csv(grid_path)
    summary = grid.groupby(['train_window', 'top_n', 'cost_bps']).agg({metric: 'max'}).reset_index()
    train_windows = sorted(summary['train_window'].unique())
    n_rows = len(train_windows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]
    for ax, tw in zip(axes, train_windows):
        subset = summary[summary['train_window'] == tw]
        pivot = subset.pivot(index='top_n', columns='cost_bps', values=metric)
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax)
        ax.set_title(f'{metric} heatmap (train_window={tw})')
        ax.set_xlabel('Cost (bps)')
        ax.set_ylabel('Top N')
    fig.tight_layout()
    output_path = output_prefix.with_name(f'{output_prefix.name}_{metric}.png')
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_json(data: dict, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Generate portfolio performance reports and visuals')
    parser.add_argument('--predictions', type=str, required=True, help='Predictions CSV path')
    parser.add_argument('--regime-data', type=str, required=True, help='Regime-labelled factor dataset for regime attribution')
    parser.add_argument('--grid-results', type=str, required=True, help='Cost sensitivity grid CSV path')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--bottom-n', type=int, default=30)
    parser.add_argument('--cost-bps', type=float, default=0.0005)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    regime_path = Path(args.regime_data)
    grid_path = Path(args.grid_results)

    preds = load_predictions(pred_path)
    regime_df = pd.read_csv(regime_path, usecols=['date', 'regime'])
    regime_df['date'] = pd.to_datetime(regime_df['date'])

    ic_series = compute_ic_series(preds)
    ts = compute_portfolio_timeseries(preds, args.top_n, args.bottom_n, args.cost_bps)
    metrics = compute_summary_metrics(ts, ic_series)
    regime_contrib = compute_regime_contributions(preds, regime_df, args.top_n, args.bottom_n, args.cost_bps)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)
    prefix = out_dir / f'performance_report_{timestamp}'

    # Save tables
    ts.to_csv(prefix.with_name(prefix.name + '_timeseries.csv'), index=False, encoding='utf-8-sig')
    regime_contrib.to_csv(prefix.with_name(prefix.name + '_regime_contrib.csv'), index=False, encoding='utf-8-sig')
    save_json({'metrics': metrics}, prefix.with_name(prefix.name + '_metrics.json'))

    # Plots
    plot_equity_curve(ts, prefix.with_name(prefix.name + '_equity.png'))
    plot_drawdown(ts, prefix.with_name(prefix.name + '_drawdown.png'))
    plot_heatmaps(grid_path, 'ls_ann', prefix.with_name(prefix.name + '_heatmap'))
    plot_heatmaps(grid_path, 'ls_ir', prefix.with_name(prefix.name + '_heatmap'))

    print('\n📈 Performance report generated:')
    print(f"  Metrics JSON: {prefix.with_name(prefix.name + '_metrics.json')}")
    print(f"  Timeseries CSV: {prefix.with_name(prefix.name + '_timeseries.csv')}")
    print(f"  Regime contributions CSV: {prefix.with_name(prefix.name + '_regime_contrib.csv')}")
    print(f"  Equity plot: {prefix.with_name(prefix.name + '_equity.png')}")
    print(f"  Drawdown plot: {prefix.with_name(prefix.name + '_drawdown.png')}")
    print(f"  Heatmaps: {prefix.with_name(prefix.name + '_heatmap_ls_ann.png')}, {prefix.with_name(prefix.name + '_heatmap_ls_ir.png')}")


if __name__ == '__main__':
    main()
