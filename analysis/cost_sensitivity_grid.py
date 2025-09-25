#!/usr/bin/env python3
"""
Cost & portfolio intensity sensitivity grid search.

Iterates across training window, regularisation alpha, portfolio breadth (TopN)
and per-side trading cost (basis points) to record out-of-sample metrics using
the regime-specific predictive pipeline. Designed for the execution plan item
1.3 so the research team can identify the combinations that dominate on IR or
returns.
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
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from predictive_model import (
    detect_factor_columns,
    evaluate_predictions,
    walk_forward_predict,
    walk_forward_predict_regime_specific,
    find_latest_regime_file,
)
from regime_model_grid_search import find_latest_mapping


def _parse_list(text: str, cast=float) -> list:
    items = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(cast(chunk))
    return items


def _load_dataframe(data_file: str | None,
                    industry_neutral: bool,
                    factor_subset: list[str] | None = None,
                    neutral_shrink: float = 1.0,
                    neutral_industries: set[str] | None = None) -> pd.DataFrame:
    if data_file:
        path = Path(data_file)
    else:
        latest = find_latest_regime_file()
        if not latest:
            raise FileNotFoundError('Unable to locate volatility regime dataset. Specify --data-file explicitly.')
        path = Path(latest)
    if not path.exists():
        raise FileNotFoundError(f'Unable to locate volatility regime dataset: {path}')
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df = df.sort_values(['date', 'stock_code']).reset_index(drop=True)
    if industry_neutral:
        if 'industry' not in df.columns:
            raise ValueError('Dataset lacks industry column but --industry-neutral was enabled')
        factor_cols = factor_subset or detect_factor_columns(df)
        shrink = max(0.0, min(1.0, neutral_shrink))
        group_keys = ['date', 'industry']

        def adjust(group: pd.DataFrame) -> pd.DataFrame:
            ind = group.name[1]
            if neutral_industries and ind not in neutral_industries:
                return group
            local = group.copy()
            means = local[factor_cols].mean()
            local[factor_cols] = local[factor_cols].sub(shrink * means)
            if 'forward_return_1d' in local.columns:
                mean_ret = local['forward_return_1d'].mean()
                local['forward_return_1d'] = local['forward_return_1d'] - shrink * mean_ret
            return local

        if shrink > 0.0:
            df = df.groupby(group_keys, group_keys=False).apply(adjust)
    return df


def _load_mapping(mapping_file: str | None) -> dict[str, list[str]] | None:
    if mapping_file:
        path = Path(mapping_file)
        if not path.exists():
            raise FileNotFoundError(f'Mapping JSON not found: {mapping_file}')
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    latest = find_latest_mapping()
    if latest:
        with open(latest, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def run_cost_grid(data_file: str | None,
                  mapping_file: str | None,
                  start_oos: str,
                  train_windows: Iterable[int],
                  alphas: Iterable[float],
                  top_ns: Iterable[int],
                  cost_bps_list: Iterable[float],
                  bottom_n_ratio: float,
                  industry_neutral: bool,
                  fallback_use_regime: bool,
                  neutral_shrink: float = 1.0,
                  neutral_industries: set[str] | None = None) -> tuple[pd.DataFrame, Path]:
    mapping = _load_mapping(mapping_file)
    factor_cols = None
    if mapping:
        used = sorted({f for fs in mapping.values() for f in fs})
        factor_cols = used if used else None

    df = _load_dataframe(data_file, industry_neutral, factor_subset=factor_cols,
                        neutral_shrink=neutral_shrink, neutral_industries=neutral_industries)
    factor_cols = factor_cols or detect_factor_columns(df)

    combinations = []
    for win, alpha, top_n, cost in product(train_windows, alphas, top_ns, cost_bps_list):
        bottom_n = int(round(top_n * bottom_n_ratio))
        bottom_n = max(1, bottom_n)
        combinations.append((win, alpha, top_n, bottom_n, cost))

    records = []
    for win, alpha, top_n, bottom_n, cost in combinations:
        if mapping and fallback_use_regime:
            preds = walk_forward_predict_regime_specific(
                df,
                mapping,
                start_oos=start_oos,
                train_window_days=win,
                alpha=alpha,
            )
        else:
            preds = walk_forward_predict(
                df,
                factor_cols,
                start_oos=start_oos,
                train_window_days=win,
                alpha=alpha,
                use_regime=fallback_use_regime and ('regime' in df.columns),
            )
        if preds.empty:
            metrics = {k: np.nan for k in ['ic_mean', 'ic_ir', 'ic_win_rate', 'long_mean', 'short_mean', 'ls_mean', 'ls_ann', 'ls_ir']}
            metrics['n_days'] = 0
        else:
            metrics = evaluate_predictions(preds, top_n=top_n, bottom_n=bottom_n, cost_bps=cost)
        record = {
            'train_window': win,
            'alpha': alpha,
            'top_n': top_n,
            'bottom_n': bottom_n,
            'cost_bps': cost,
            **metrics,
        }
        records.append(record)

    res = pd.DataFrame(records)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path('data') / f'cost_sensitivity_grid_{timestamp}.csv'
    res.to_csv(out_path, index=False, encoding='utf-8-sig')
    return res, out_path


def main():
    parser = argparse.ArgumentParser(description='Cost sensitivity grid search for regime predictive model')
    parser.add_argument('--data-file', type=str, default=None,
                        help='Optional explicit volatility regime dataset (defaults to latest)')
    parser.add_argument('--mapping-file', type=str, default=None,
                        help='Optional regime-factor mapping JSON (defaults to latest)')
    parser.add_argument('--start-oos', type=str, default='2022-01-01')
    parser.add_argument('--train-windows', type=str, default='756,1008')
    parser.add_argument('--alphas', type=str, default='0.5,1.0,2.0')
    parser.add_argument('--top-ns', type=str, default='20,25,30')
    parser.add_argument('--cost-bps', type=str, default='0.0003,0.0005', help='Comma-separated costs in decimal form (e.g. 0.0003=3bp)')
    parser.add_argument('--bottom-n-ratio', type=float, default=1.0, help='Bottom bucket as proportion of top_n (default 1.0)')
    parser.add_argument('--industry-neutral', action='store_true')
    parser.add_argument('--neutral-shrink', type=float, default=1.0,
                        help='Shrink factor (0-1) when --industry-neutral is enabled')
    parser.add_argument('--neutral-industries', type=str, default=None,
                        help='Comma-separated industries to neutralise (default=all)')
    parser.add_argument('--no-regime', action='store_true', help='Force pooled model instead of regime-specific mapping')
    args = parser.parse_args()

    train_windows = [int(v) for v in _parse_list(args.train_windows, cast=float)]
    alphas = [float(v) for v in _parse_list(args.alphas, cast=float)]
    top_ns = [int(v) for v in _parse_list(args.top_ns, cast=float)]
    cost_bps_list = [float(v) for v in _parse_list(args.cost_bps, cast=float)]

    industries = None
    if args.neutral_industries:
        industries = {item.strip() for item in args.neutral_industries.split(',') if item.strip()}

    results, out_path = run_cost_grid(
        data_file=args.data_file,
        mapping_file=args.mapping_file,
        start_oos=args.start_oos,
        train_windows=train_windows,
        alphas=alphas,
        top_ns=top_ns,
        cost_bps_list=cost_bps_list,
        bottom_n_ratio=args.bottom_n_ratio,
        industry_neutral=args.industry_neutral,
        fallback_use_regime=not args.no_regime,
        neutral_shrink=args.neutral_shrink,
        neutral_industries=industries,
    )

    best_ir = results.sort_values('ls_ir', ascending=False).head(5)
    best_ann = results.sort_values('ls_ann', ascending=False).head(5)

    print('\n📊 Cost sensitivity grid complete')
    print(f'  Saved results to: {out_path}')
    print('\nTop 5 by LS_IR:')
    print(best_ir[['train_window', 'alpha', 'top_n', 'bottom_n', 'cost_bps', 'ls_ir', 'ls_ann', 'ic_mean']])
    print('\nTop 5 by LS_ann:')
    print(best_ann[['train_window', 'alpha', 'top_n', 'bottom_n', 'cost_bps', 'ls_ann', 'ls_ir', 'ic_mean']])


if __name__ == '__main__':
    main()
