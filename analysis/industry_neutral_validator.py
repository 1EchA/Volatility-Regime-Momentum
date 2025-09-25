#!/usr/bin/env python3
"""
Industry neutralization validation helper.

Loads a volatility regime dataset and compares model metrics before and after
applying cross-sectional industry neutralization. Results are saved to the
`data/` directory for later reference in reports.
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

import numpy as np
import pandas as pd

from predictive_model import (
    detect_factor_columns,
    walk_forward_predict,
    walk_forward_predict_regime_specific,
    evaluate_predictions,
    find_latest_regime_file,
)
from regime_model_grid_search import find_latest_mapping


@dataclass
class ScenarioResult:
    scenario: str
    ic_mean: float
    ic_ir: float
    ic_win_rate: float
    ls_ann: float
    ls_ir: float
    long_mean: float
    short_mean: float
    n_days: int

    @classmethod
    def from_metrics(cls, scenario: str, metrics: dict) -> "ScenarioResult":
        return cls(
            scenario=scenario,
            ic_mean=float(metrics.get("ic_mean", np.nan)),
            ic_ir=float(metrics.get("ic_ir", np.nan)),
            ic_win_rate=float(metrics.get("ic_win_rate", np.nan)),
            ls_ann=float(metrics.get("ls_ann", np.nan)),
            ls_ir=float(metrics.get("ls_ir", np.nan)),
            long_mean=float(metrics.get("long_mean", np.nan)),
            short_mean=float(metrics.get("short_mean", np.nan)),
            n_days=int(metrics.get("n_days", 0) or 0),
        )


def _load_regime_dataframe(data_file: str | None = None) -> pd.DataFrame:
    if data_file:
        path = Path(data_file)
        if not path.exists():
            raise FileNotFoundError(f"Regime data file not found: {data_file}")
        df = pd.read_csv(path)
    else:
        latest = find_latest_regime_file()
        if latest is None:
            raise FileNotFoundError("No volatility_regime_data_*.csv found under data/.")
        df = pd.read_csv(latest)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'stock_code']).reset_index(drop=True)
    return df


def _neutralize_by_industry(df: pd.DataFrame,
                            factor_cols: list[str],
                            shrink: float = 1.0,
                            industries: set[str] | None = None) -> pd.DataFrame:
    if 'industry' not in df.columns:
        raise ValueError('DataFrame lacks required industry column for neutralization')
    shrink = max(0.0, min(1.0, shrink))
    group_keys = ['date', 'industry']
    df_copy = df.copy()

    if shrink == 0.0:
        return df_copy

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

    df_copy = df_copy.groupby(group_keys, group_keys=False).apply(adjust)
    return df_copy


def _load_regime_mapping(mapping_file: str | None = None) -> dict[str, list[str]] | None:
    path = Path(mapping_file) if mapping_file else None
    if path and path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    latest = find_latest_mapping()
    if latest:
        with open(latest, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _evaluate(df: pd.DataFrame,
              scenario_name: str,
              factor_cols: list[str],
              regime_mapping: dict[str, list[str]] | None,
              start_oos: str,
              train_window: int,
              alpha: float,
              top_n: int,
              bottom_n: int,
              cost_bps: float) -> ScenarioResult:
    if regime_mapping:
        preds = walk_forward_predict_regime_specific(
            df,
            regime_mapping,
            start_oos=start_oos,
            train_window_days=train_window,
            alpha=alpha,
        )
    else:
        preds = walk_forward_predict(
            df,
            factor_cols,
            start_oos=start_oos,
            train_window_days=train_window,
            alpha=alpha,
            use_regime='regime' in df.columns,
        )
    if preds.empty:
        return ScenarioResult(
            scenario=scenario_name,
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
    return ScenarioResult.from_metrics(scenario_name, metrics)


def _industry_stability_stats(df: pd.DataFrame) -> dict[str, float]:
    if 'industry' not in df.columns:
        return {
            'industry_coverage': 0.0,
            'industry_stability_ratio': 0.0,
            'unique_industries': 0,
        }
    valid = df['industry'].fillna('未分类')
    coverage = float((valid != '未分类').mean())
    dominant = df.groupby('stock_code')['industry'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else '未分类')
    dominant_map = dominant.to_dict()
    stability = float((df['stock_code'].map(dominant_map) == df['industry']).mean())
    return {
        'industry_coverage': coverage,
        'industry_stability_ratio': stability,
        'unique_industries': int(valid.nunique()),
    }


def run_industry_validation(data_file: str | None = None,
                             mapping_file: str | None = None,
                             start_oos: str = '2022-01-01',
                             train_window: int = 756,
                             alpha: float = 1.0,
                             top_n: int = 30,
                             bottom_n: int = 30,
                             cost_bps: float = 0.0005,
                             neutral_shrink: float = 1.0,
                             neutral_industries: set[str] | None = None) -> tuple[dict, Path]:
    df = _load_regime_dataframe(data_file)
    factor_cols = detect_factor_columns(df)
    regime_mapping = _load_regime_mapping(mapping_file)

    scenario_results: list[ScenarioResult] = []
    scenario_results.append(_evaluate(df, 'baseline', factor_cols, regime_mapping,
                                      start_oos, train_window, alpha, top_n, bottom_n, cost_bps))

    used_cols = factor_cols
    if regime_mapping:
        used_union = sorted({col for cols in regime_mapping.values() for col in cols})
        if used_union:
            used_cols = used_union
    if 'industry' in df.columns:
        neutral_df = _neutralize_by_industry(df, used_cols, shrink=neutral_shrink, industries=neutral_industries)
        scenario_results.append(_evaluate(neutral_df, 'industry_neutral', factor_cols, regime_mapping,
                                          start_oos, train_window, alpha, top_n, bottom_n, cost_bps))

    coverage_stats = _industry_stability_stats(df)

    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'data_file': data_file or find_latest_regime_file(),
        'mapping_file': mapping_file or find_latest_mapping(),
        'start_oos': start_oos,
        'train_window': train_window,
        'alpha': alpha,
        'top_n': top_n,
        'bottom_n': bottom_n,
        'cost_bps': cost_bps,
        'neutral_shrink': neutral_shrink,
        'neutral_industries': sorted(neutral_industries) if neutral_industries else None,
        'industry_stats': coverage_stats,
        'scenarios': [asdict(r) for r in scenario_results],
    }

    if len(scenario_results) == 2 and scenario_results[0].n_days and scenario_results[1].n_days:
        payload['diff'] = {
            k: float(getattr(scenario_results[1], k) - getattr(scenario_results[0], k))
            for k in ['ic_mean', 'ic_ir', 'ic_win_rate', 'ls_ann', 'ls_ir', 'long_mean', 'short_mean']
        }

    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = out_dir / f'industry_neutral_comparison_{ts}'
    json_path = base.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_rows = []
    for res in payload['scenarios']:
        row = res.copy()
        row.update({'start_oos': start_oos, 'train_window': train_window, 'alpha': alpha, 'cost_bps': cost_bps})
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(base.with_suffix('.csv'), index=False, encoding='utf-8-sig')

    payload['output_prefix'] = str(base)
    return payload, base


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare baseline vs industry-neutral modeling metrics')
    parser.add_argument('--data-file', type=str, default=None,
                        help='Specify regime dataset; defaults to latest volatility_regime_data_*.csv')
    parser.add_argument('--mapping-file', type=str, default=None,
                        help='Optional regime factor mapping JSON')
    parser.add_argument('--start-oos', type=str, default='2022-01-01')
    parser.add_argument('--train-window', type=int, default=756)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--bottom-n', type=int, default=30)
    parser.add_argument('--cost-bps', type=float, default=0.0005)
    parser.add_argument('--neutral-shrink', type=float, default=1.0,
                        help='Shrink factor (0-1) for industry adjustment; 1=full neutralisation')
    parser.add_argument('--neutral-industries', type=str, default=None,
                        help='Comma-separated industry names to neutralise; default=all')
    args = parser.parse_args()

    industries = None
    if args.neutral_industries:
        industries = {item.strip() for item in args.neutral_industries.split(',') if item.strip()}

    payload, base_path = run_industry_validation(
        data_file=args.data_file,
        mapping_file=args.mapping_file,
        start_oos=args.start_oos,
        train_window=args.train_window,
        alpha=args.alpha,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        cost_bps=args.cost_bps,
        neutral_shrink=args.neutral_shrink,
        neutral_industries=industries,
    )

    print("\n📊 Industry neutralization comparison saved.")
    for scenario in payload['scenarios']:
        ic = scenario['ic_mean']
        ls_ir = scenario['ls_ir']
        ls_ann = scenario['ls_ann']
        print(f"  -> {scenario['scenario']}: IC={ic:.4f}, LS_IR={ls_ir:.3f}, LS_ann={ls_ann:.2%}")
    print(f"  Outputs: {base_path.with_suffix('.json')} / {base_path.with_suffix('.csv')}")


if __name__ == '__main__':
    main()
