#!/usr/bin/env python3
"""
Model enhancement prototyper.

Benchmarks alternative modelling choices (Elastic Net, interaction features)
against the baseline ridge regime model. Outputs a JSON/CSV report capturing
key metrics for downstream documentation (execution plan item 1.4).
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
from typing import Callable, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from predictive_model import (
    detect_factor_columns,
    evaluate_predictions,
    walk_forward_predict_regime_specific,
    find_latest_regime_file,
)
from regime_model_grid_search import find_latest_mapping

try:
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:  # pragma: no cover - fallback if sklearn missing
    Ridge = None
    ElasticNet = None
    StandardScaler = None
    GradientBoostingRegressor = None


@dataclass
class ScenarioMetrics:
    label: str
    ic_mean: float
    ic_ir: float
    ic_win_rate: float
    ls_ann: float
    ls_ir: float
    long_mean: float
    short_mean: float
    n_days: int

    def to_dict(self) -> dict:
        return asdict(self)


def _load_dataset(path: str | None = None) -> pd.DataFrame:
    if path:
        data_path = Path(path)
    else:
        latest = find_latest_regime_file()
        if not latest:
            raise FileNotFoundError('No volatility_regime_data_*.csv found in data directory.')
        data_path = Path(latest)
    if not data_path.exists():
        raise FileNotFoundError(f'Dataset not found: {data_path}')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    return df.sort_values(['date', 'stock_code']).reset_index(drop=True)


def _neutralize_dataframe(df: pd.DataFrame,
                          factor_cols: List[str],
                          shrink: float,
                          industries: Optional[Set[str]]) -> pd.DataFrame:
    if shrink <= 0 or 'industry' not in df.columns:
        return df
    shrink = max(0.0, min(1.0, shrink))
    if shrink == 0:
        return df
    group_keys = ['date', 'industry']

    def adjust(group: pd.DataFrame) -> pd.DataFrame:
        ind = group.name[1]
        if industries and ind not in industries:
            return group
        local = group.copy()
        if factor_cols:
            means = local[factor_cols].mean()
            local[factor_cols] = local[factor_cols].sub(shrink * means)
        if 'forward_return_1d' in local.columns:
            mean_ret = local['forward_return_1d'].mean()
            local['forward_return_1d'] = local['forward_return_1d'] - shrink * mean_ret
        return local

    return df.groupby(group_keys, group_keys=False).apply(adjust).reset_index(drop=True)


def _load_mapping(path: str | None = None) -> dict[str, list[str]]:
    if path:
        mapping_path = Path(path)
    else:
        latest = find_latest_mapping()
        if not latest:
            raise FileNotFoundError('No selected_factors_by_regime_*.json mapping found in data directory.')
        mapping_path = Path(latest)
    if not mapping_path.exists():
        raise FileNotFoundError(f'Mapping file missing: {mapping_path}')
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return {k: list(v) for k, v in mapping.items()}


def _union_factor_columns(mapping: Dict[str, List[str]], fallback_cols: List[str]) -> List[str]:
    used = sorted({f for fs in mapping.values() for f in fs})
    return used if used else fallback_cols


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]):
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan


def _walk_forward_custom(df: pd.DataFrame,
                         mapping: Dict[str, List[str]],
                         model_factory: Callable[[], object],
                         start_oos: str,
                         train_window_days: int,
                         standardize: bool = True) -> pd.DataFrame:
    dates = sorted(df['date'].unique())
    start_dt = pd.to_datetime(start_oos)
    preds = []
    for t in dates:
        if t < start_dt:
            continue
        reg_today = df.loc[df['date'] == t, 'regime']
        if reg_today.empty:
            continue
        regime = reg_today.iloc[0]
        factors = mapping.get(regime, [])
        if not factors:
            continue
        cols = ['date', 'stock_code', 'forward_return_1d'] + factors
        train_start = t - pd.Timedelta(days=train_window_days)
        train_mask = (df['date'] >= train_start) & (df['date'] < t) & (df['regime'] == regime)
        test_mask = (df['date'] == t)
        train_df = df.loc[train_mask, cols].dropna()
        test_df = df.loc[test_mask, cols].dropna(subset=factors)
        if len(train_df) < max(800, len(factors) * 15) or test_df.empty:
            continue
        model = model_factory()
        X_train = train_df[factors].values
        y_train = train_df['forward_return_1d'].values
        X_test = test_df[factors].values
        y_test = test_df['forward_return_1d'].values
        if standardize and StandardScaler is not None:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        out = test_df[['date', 'stock_code']].copy()
        out['y_true'] = y_test
        out['y_pred'] = y_pred
        preds.append(out)
    if not preds:
        return pd.DataFrame()
    return pd.concat(preds, ignore_index=True)


def _add_interactions(df: pd.DataFrame,
                      mapping: Dict[str, List[str]],
                      max_pairs_per_regime: int = 3) -> tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    new_cols: Dict[str, tuple[str, str]] = {}
    updated_mapping: Dict[str, List[str]] = {}
    for regime, factors in mapping.items():
        updated = list(factors)
        top = factors[:max_pairs_per_regime]
        for i, f1 in enumerate(top):
            for f2 in top[i+1:]:
                col = f'{f1}__x__{f2}'
                new_cols[col] = (f1, f2)
                updated.append(col)
        updated_mapping[regime] = updated
    if not new_cols:
        return df, mapping, []
    df_new = df.copy()
    for col, (f1, f2) in new_cols.items():
        if f1 in df_new.columns and f2 in df_new.columns:
            df_new[col] = df_new[f1] * df_new[f2]
        else:
            df_new[col] = np.nan
    interaction_cols = list(new_cols.keys())
    df_new[interaction_cols] = df_new.groupby('date')[interaction_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    return df_new, updated_mapping, interaction_cols


def _evaluate(pred_df: pd.DataFrame, label: str, top_n: int, bottom_n: int, cost_bps: float) -> ScenarioMetrics:
    if pred_df.empty:
        return ScenarioMetrics(
            label=label,
            ic_mean=float('nan'),
            ic_ir=float('nan'),
            ic_win_rate=float('nan'),
            ls_ann=float('nan'),
            ls_ir=float('nan'),
            long_mean=float('nan'),
            short_mean=float('nan'),
            n_days=0,
        )
    metrics = evaluate_predictions(pred_df, top_n=top_n, bottom_n=bottom_n, cost_bps=cost_bps)
    n_days = metrics.get('n_days', 0)
    if isinstance(n_days, float) and (np.isnan(n_days) or np.isinf(n_days)):
        n_days = 0
    return ScenarioMetrics(
        label=label,
        ic_mean=float(metrics.get('ic_mean', np.nan)),
        ic_ir=float(metrics.get('ic_ir', np.nan)),
        ic_win_rate=float(metrics.get('ic_win_rate', np.nan)),
        ls_ann=float(metrics.get('ls_ann', np.nan)),
        ls_ir=float(metrics.get('ls_ir', np.nan)),
        long_mean=float(metrics.get('long_mean', np.nan)),
        short_mean=float(metrics.get('short_mean', np.nan)),
        n_days=int(n_days),
    )


def run_model_enhancements(data_file: str | None = None,
                           mapping_file: str | None = None,
                           start_oos: str = '2022-01-01',
                           train_window: int = 756,
                           alpha: float = 1.0,
                           elastic_net_l1: Iterable[float] = (0.3, 0.5, 0.7),
                           top_n: int = 30,
                           bottom_n: int = 30,
                           cost_bps: float = 0.0005,
                           interaction_pairs: int = 3,
                           neutral_shrink: float = 0.0,
                           neutral_industries: Optional[Set[str]] = None) -> tuple[dict, Path]:
    df = _load_dataset(data_file)
    mapping = _load_mapping(mapping_file)
    fallback_cols = detect_factor_columns(df)
    _ensure_columns(df, _union_factor_columns(mapping, fallback_cols))
    df = _neutralize_dataframe(df, _union_factor_columns(mapping, fallback_cols), neutral_shrink, neutral_industries)

    results: List[ScenarioMetrics] = []
    payload_details: dict[str, dict] = {}

    # Baseline ridge via existing helper
    baseline_preds = walk_forward_predict_regime_specific(
        df,
        mapping,
        start_oos=start_oos,
        train_window_days=train_window,
        alpha=alpha,
    )
    baseline_metrics = _evaluate(baseline_preds, 'ridge_baseline', top_n, bottom_n, cost_bps)
    results.append(baseline_metrics)
    payload_details['ridge_baseline'] = {'alpha': alpha}

    # Elastic Net variants
    if ElasticNet is not None:
        elastic_metrics = []
        for l1_ratio in elastic_net_l1:
            def factory(a=alpha, l1=l1_ratio):
                return ElasticNet(alpha=a, l1_ratio=l1, fit_intercept=True, max_iter=5000, random_state=0)
            preds = _walk_forward_custom(df, mapping, factory, start_oos, train_window, standardize=True)
            label = f'elastic_net_l1_{l1_ratio:.2f}'
            metrics = _evaluate(preds, label, top_n, bottom_n, cost_bps)
            results.append(metrics)
            elastic_metrics.append((l1_ratio, metrics.ls_ir))
        payload_details['elastic_net'] = {'l1_grid': list(elastic_net_l1), 'ls_ir': elastic_metrics}
    else:
        payload_details['elastic_net'] = {'error': 'sklearn not available'}

    # Interaction features with ridge
    df_inter, mapping_inter, interaction_cols = _add_interactions(df, mapping, max_pairs_per_regime=interaction_pairs)
    if interaction_cols and Ridge is not None:
        _ensure_columns(df_inter, interaction_cols)
        preds_inter = _walk_forward_custom(
            df_inter,
            mapping_inter,
            lambda a=alpha: Ridge(alpha=a, fit_intercept=True),
            start_oos,
            train_window,
            standardize=True,
        )
        metrics_inter = _evaluate(preds_inter, 'ridge_with_interactions', top_n, bottom_n, cost_bps)
        results.append(metrics_inter)
        payload_details['ridge_with_interactions'] = {
            'alpha': alpha,
            'added_features': interaction_cols,
        }
    elif interaction_cols:
        payload_details['ridge_with_interactions'] = {'error': 'sklearn Ridge unavailable'}

    # Elastic Net with interactions (optional if sklearn available)
    if ElasticNet is not None and interaction_cols:
        best_combo = None
        for l1_ratio in elastic_net_l1:
            def factory(a=alpha, l1=l1_ratio):
                return ElasticNet(alpha=a, l1_ratio=l1, fit_intercept=True, max_iter=5000, random_state=0)
            preds = _walk_forward_custom(df_inter, mapping_inter, factory, start_oos, train_window, standardize=True)
            label = f'elastic_net_interactions_l1_{l1_ratio:.2f}'
            metrics = _evaluate(preds, label, top_n, bottom_n, cost_bps)
            results.append(metrics)
            if best_combo is None or metrics.ls_ir > best_combo[1]:
                best_combo = (l1_ratio, metrics.ls_ir)
        if best_combo:
            payload_details['elastic_net_with_interactions'] = {
                'best_l1_ratio': best_combo[0],
                'best_ls_ir': best_combo[1],
            }

    # Gradient Boosting baseline
    if GradientBoostingRegressor is not None:
        def gb_factory():
            return GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=0,
                loss='squared_error',
            )
        preds_gb = _walk_forward_custom(df, mapping, gb_factory, start_oos, train_window, standardize=False)
        gb_metrics = _evaluate(preds_gb, 'gradient_boosting', top_n, bottom_n, cost_bps)
        results.append(gb_metrics)
        payload_details['gradient_boosting'] = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 3,
        }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = Path('data') / f'model_enhancement_summary_{timestamp}'

    metrics_df = pd.DataFrame([r.to_dict() for r in results])
    metrics_df.to_csv(prefix.with_suffix('.csv'), index=False, encoding='utf-8-sig')

    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'parameters': {
            'data_file': data_file or find_latest_regime_file(),
            'mapping_file': mapping_file or find_latest_mapping(),
            'start_oos': start_oos,
            'train_window': train_window,
            'alpha': alpha,
            'top_n': top_n,
            'bottom_n': bottom_n,
            'cost_bps': cost_bps,
            'interaction_pairs_per_regime': interaction_pairs,
            'neutral_shrink': neutral_shrink,
            'neutral_industries': sorted(neutral_industries) if neutral_industries else None,
        },
        'scenarios': metrics_df.to_dict(orient='records'),
        'details': payload_details,
    }

    with open(prefix.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    payload['output_prefix'] = str(prefix)
    return payload, prefix


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark Elastic Net and interaction feature enhancements')
    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--mapping-file', type=str, default=None)
    parser.add_argument('--start-oos', type=str, default='2022-01-01')
    parser.add_argument('--train-window', type=int, default=756)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--bottom-n', type=int, default=30)
    parser.add_argument('--cost-bps', type=float, default=0.0005)
    parser.add_argument('--interaction-pairs', type=int, default=3)
    parser.add_argument('--l1-grid', type=str, default='0.3,0.5,0.7')
    parser.add_argument('--neutral-shrink', type=float, default=0.0,
                        help='Shrink factor (0-1) to neutralise industries before modeling')
    parser.add_argument('--neutral-industries', type=str, default=None,
                        help='Comma-separated industries to neutralise; default all when shrink>0')
    args = parser.parse_args()

    l1_grid = [float(x.strip()) for x in args.l1_grid.split(',') if x.strip()]
    industries: Optional[Set[str]] = None
    if args.neutral_industries:
        industries = {item.strip() for item in args.neutral_industries.split(',') if item.strip()}

    payload, prefix = run_model_enhancements(
        data_file=args.data_file,
        mapping_file=args.mapping_file,
        start_oos=args.start_oos,
        train_window=args.train_window,
        alpha=args.alpha,
        elastic_net_l1=l1_grid,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        cost_bps=args.cost_bps,
        interaction_pairs=args.interaction_pairs,
        neutral_shrink=args.neutral_shrink,
        neutral_industries=industries,
    )

    print('\n📊 Model enhancement benchmarking saved:')
    for scenario in payload['scenarios']:
        print(f"  -> {scenario['label']}: IC={scenario['ic_mean']:.4f}, LS_IR={scenario['ls_ir']:.3f}, LS_ann={scenario['ls_ann']:.2%}")
    print(f"  Outputs: {prefix.with_suffix('.json')} / {prefix.with_suffix('.csv')}")


if __name__ == '__main__':
    main()
