#!/usr/bin/env python3
"""
Standardization comparison runner.

Re-computes regime-specific factor selection and predictive performance for
both z-score and rank-normalised factor sets. Produces a side-by-side report so
we can document how the standardisation choice affects selected factors and
out-of-sample metrics.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from predictive_model import (
    detect_factor_columns,
    walk_forward_predict_regime_specific,
    evaluate_predictions,
    find_latest_regime_file,
)
from regime_factor_selector import select_in_regime


@dataclass
class ModelMetrics:
    ic_mean: float
    ic_ir: float
    ic_win_rate: float
    ls_ann: float
    ls_ir: float
    long_mean: float
    short_mean: float
    n_days: int

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetrics":
        return cls(
            ic_mean=float(data.get('ic_mean', np.nan)),
            ic_ir=float(data.get('ic_ir', np.nan)),
            ic_win_rate=float(data.get('ic_win_rate', np.nan)),
            ls_ann=float(data.get('ls_ann', np.nan)),
            ls_ir=float(data.get('ls_ir', np.nan)),
            long_mean=float(data.get('long_mean', np.nan)),
            short_mean=float(data.get('short_mean', np.nan)),
            n_days=int(data.get('n_days', 0) or 0),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StandardizationResult:
    label: str
    metrics: ModelMetrics
    selected_factors: Dict[str, List[str]]
    factor_summary_path: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload['metrics'] = self.metrics.to_dict()
        return payload


def _load_regime_dataset(path: str | None = None) -> pd.DataFrame:
    if path:
        data_path = Path(path)
    else:
        latest = find_latest_regime_file()
        if latest is None:
            raise FileNotFoundError('No volatility_regime_data_*.csv file found in data/')
        data_path = Path(latest)
    if not data_path.exists():
        raise FileNotFoundError(f'Regime dataset missing: {data_path}')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df = df.sort_values(['date', 'stock_code']).reset_index(drop=True)
    return df


def _inject_rank_standardisation(regime_df: pd.DataFrame, rank_factor_file: str) -> Tuple[pd.DataFrame, int]:
    rank_path = Path(rank_factor_file)
    if not rank_path.exists():
        raise FileNotFoundError(f'Rank standardised factor file missing: {rank_factor_file}')
    rank_df = pd.read_csv(rank_path)
    rank_df['date'] = pd.to_datetime(rank_df['date'])
    rank_df['stock_code'] = rank_df['stock_code'].astype(str).str.zfill(6)
    std_cols = [c for c in rank_df.columns if c.endswith('_std')]
    subset = rank_df[['date', 'stock_code'] + std_cols].copy()
    rename_map = {c: f'{c}__rank' for c in std_cols}
    subset.rename(columns=rename_map, inplace=True)

    merged = regime_df.merge(subset, on=['date', 'stock_code'], how='left')
    replaced = 0
    for col in std_cols:
        rank_col = f'{col}__rank'
        if rank_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = np.nan
        mask = merged[rank_col].notna()
        replaced += int(mask.sum())
        merged.loc[mask, col] = merged.loc[mask, rank_col]
        merged.drop(columns=[rank_col], inplace=True)
    missing_rows = merged[std_cols].isna().all(axis=1).sum()
    return merged, missing_rows


def _select_factors(df: pd.DataFrame, label: str,
                    ic_thresh: float = 0.01,
                    corr_thresh: float = 0.8) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    factor_cols = detect_factor_columns(df)
    regimes = [r for r in df['regime'].dropna().unique().tolist() if r != '未分类']
    summaries = []
    mapping: Dict[str, List[str]] = {}
    for regime in regimes:
        sub = df[df['regime'] == regime].copy()
        selected, summary = select_in_regime(sub, factor_cols, ic_thresh=ic_thresh, corr_thresh=corr_thresh)
        mapping[regime] = selected
        summary = summary.assign(regime=regime, standardisation=label)
        summaries.append(summary)
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return mapping, summary_df


def _run_model(df: pd.DataFrame,
              mapping: Dict[str, List[str]],
              start_oos: str,
              train_window: int,
              alpha: float,
              top_n: int,
              bottom_n: int,
              cost_bps: float) -> ModelMetrics:
    preds = walk_forward_predict_regime_specific(
        df,
        mapping,
        start_oos=start_oos,
        train_window_days=train_window,
        alpha=alpha,
    )
    if preds.empty:
        return ModelMetrics(
            ic_mean=float('nan'),
            ic_ir=float('nan'),
            ic_win_rate=float('nan'),
            ls_ann=float('nan'),
            ls_ir=float('nan'),
            long_mean=float('nan'),
            short_mean=float('nan'),
            n_days=0,
        )
    metrics = evaluate_predictions(preds, top_n=top_n, bottom_n=bottom_n, cost_bps=cost_bps)
    return ModelMetrics.from_dict(metrics)


def _save_factor_summary(summary_df: pd.DataFrame, label: str, timestamp: str) -> str:
    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f'standardisation_{label}_factor_summary_{timestamp}.csv'
    if summary_df.empty:
        placeholder = pd.DataFrame(columns=['factor', 'ic_mean', 'ic_std', 't', 'p', 'win_rate', 'n_days', 'regime', 'standardisation'])
        placeholder.to_csv(path, index=False, encoding='utf-8-sig')
    else:
        summary_df.to_csv(path, index=False, encoding='utf-8-sig')
    return str(path)


def run_standardisation_comparison(
    zscore_regime_file: str | None = None,
    rank_factor_file: str = 'data/simple_factor_data_rank.csv',
    start_oos: str = '2022-01-01',
    train_window: int = 756,
    alpha: float = 1.0,
    top_n: int = 30,
    bottom_n: int = 30,
    cost_bps: float = 0.0005,
) -> Tuple[dict, Path]:
    base_df = _load_regime_dataset(zscore_regime_file)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Baseline (z-score)
    z_mapping, z_summary = _select_factors(base_df, 'zscore')
    z_metrics = _run_model(base_df, z_mapping, start_oos, train_window, alpha, top_n, bottom_n, cost_bps)
    z_summary_path = _save_factor_summary(z_summary, 'zscore', timestamp)
    z_result = StandardizationResult('zscore', z_metrics, z_mapping, z_summary_path)

    # Rank-standardised copy
    rank_df, missing_rows = _inject_rank_standardisation(base_df, rank_factor_file)
    if missing_rows:
        print(f'⚠️ Warning: {missing_rows} rows lack rank-standardised factors after merge.')
    r_mapping, r_summary = _select_factors(rank_df, 'rank')
    r_metrics = _run_model(rank_df, r_mapping, start_oos, train_window, alpha, top_n, bottom_n, cost_bps)
    r_summary_path = _save_factor_summary(r_summary, 'rank', timestamp)
    r_result = StandardizationResult('rank', r_metrics, r_mapping, r_summary_path)

    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'parameters': {
            'zscore_regime_file': zscore_regime_file or find_latest_regime_file(),
            'rank_factor_file': rank_factor_file,
            'start_oos': start_oos,
            'train_window': train_window,
            'alpha': alpha,
            'top_n': top_n,
            'bottom_n': bottom_n,
            'cost_bps': cost_bps,
        },
        'zscore': z_result.to_dict(),
        'rank': r_result.to_dict(),
        'metric_diff': {
            key: float(getattr(r_result.metrics, key) - getattr(z_result.metrics, key))
            for key in ['ic_mean', 'ic_ir', 'ic_win_rate', 'ls_ann', 'ls_ir', 'long_mean', 'short_mean']
        },
        'factor_diff': {
            regime: {
                'zscore': z_result.selected_factors.get(regime, []),
                'rank': r_result.selected_factors.get(regime, []),
                'only_zscore': sorted(list(set(z_result.selected_factors.get(regime, [])) - set(r_result.selected_factors.get(regime, [])))),
                'only_rank': sorted(list(set(r_result.selected_factors.get(regime, [])) - set(z_result.selected_factors.get(regime, [])))),
            }
            for regime in sorted(set(z_result.selected_factors) | set(r_result.selected_factors))
        },
    }

    prefix = Path('data') / f'standardisation_comparison_{timestamp}'
    with open(prefix.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    summary_rows = []
    for label, result in [('zscore', z_result), ('rank', r_result)]:
        row = {'label': label}
        row.update(result.metrics.to_dict())
        row['factor_summary'] = result.factor_summary_path
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(prefix.with_suffix('.csv'), index=False, encoding='utf-8-sig')

    payload['output_prefix'] = str(prefix)
    return payload, prefix


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare z-score vs rank standardisation results')
    parser.add_argument('--zscore-regime-file', type=str, default=None,
                        help='Optional explicit regime dataset (defaults to latest volatility_regime_data_*.csv)')
    parser.add_argument('--rank-factor-file', type=str, default='data/simple_factor_data_rank.csv')
    parser.add_argument('--start-oos', type=str, default='2022-01-01')
    parser.add_argument('--train-window', type=int, default=756)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--bottom-n', type=int, default=30)
    parser.add_argument('--cost-bps', type=float, default=0.0005)
    args = parser.parse_args()

    payload, prefix = run_standardisation_comparison(
        zscore_regime_file=args.zscore_regime_file,
        rank_factor_file=args.rank_factor_file,
        start_oos=args.start_oos,
        train_window=args.train_window,
        alpha=args.alpha,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        cost_bps=args.cost_bps,
    )

    print('\n📊 Standardisation comparison saved:')
    for label in ['zscore', 'rank']:
        metrics = payload[label]['metrics']
        print(f"  -> {label}: IC={metrics['ic_mean']:.4f}, LS_IR={metrics['ls_ir']:.3f}, LS_ann={metrics['ls_ann']:.2%}")
    print(f"  Outputs: {prefix.with_suffix('.json')} / {prefix.with_suffix('.csv')}")


if __name__ == '__main__':
    main()
