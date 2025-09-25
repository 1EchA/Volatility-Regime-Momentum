#!/usr/bin/env python3
"""
End-to-end execution helper for the volatility-regime momentum project.

Workflow:
1. (Optional) recompute factor dataset with specified standardisation.
2. (Optional) rerun volatility regime classification on the factor dataset.
3. Perform regime-wise factor selection.
4. Train the regime-specific predictive model and save predictions/metrics.
5. (Optional) run the cost sensitivity grid search and performance reporter.

Outputs are saved under data/ with timestamped filenames for traceability.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np

from simple_factor_calculator import SimpleFactorCalculator
from volatility_regime_analyzer import VolatilityRegimeAnalyzer
from regime_factor_selector import select_in_regime
from predictive_model import (
    detect_factor_columns,
    walk_forward_predict_regime_specific,
    evaluate_predictions,
    save_outputs,
)
from predictive_model import find_latest_regime_file
from regime_model_grid_search import find_latest_mapping

# Reuse helper functions from analysis scripts
from analysis.performance_reporter import (
    load_predictions,
    compute_ic_series,
    compute_portfolio_timeseries,
    compute_summary_metrics,
)

from analysis.performance_reporter import compute_regime_contributions
from analysis.execution_strategies import (
    baseline_daily,
    hysteresis_bands,
    ema_hysteresis_combo,
    low_freq_rebalance,
    swap_cap_limited,
    compute_ic_series_with_score,
)
import subprocess


DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)


def recompute_factors(standardisation: str, universe: str | None) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_name = f'simple_factor_data_{standardisation}_{ts}.csv'
    calc = SimpleFactorCalculator(
        data_dir=str(DATA_DIR),
        universe_file=universe,
        standardization=standardisation,
        output_filename=output_name,
    )
    calc.compute()
    return DATA_DIR / output_name


def recompute_regime(factor_file: Path) -> Path:
    analyzer = VolatilityRegimeAnalyzer(data_file=str(factor_file))
    regime_df = analyzer.run_full_analysis()
    if regime_df is None:
        raise RuntimeError('Regime analysis failed')
    latest_regime = find_latest_regime_file()
    if not latest_regime:
        raise RuntimeError('No regime file produced')
    return Path(latest_regime)


def run_factor_selection(regime_file: Path, ic_thresh: float, corr_thresh: float,
                         neutral_shrink: float, neutral_industries: Optional[Set[str]]) -> tuple[Dict[str, List[str]], Path]:
    import pandas as pd
    df = pd.read_csv(regime_file)
    df['date'] = pd.to_datetime(df['date'])
    factors = detect_factor_columns(df)
    regimes = [r for r in df['regime'].dropna().unique().tolist() if r != '未分类']
    mapping: Dict[str, List[str]] = {}
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    for reg in regimes:
        sub = df[df['regime'] == reg]
        selected, summary = select_in_regime(sub, factors, ic_thresh=ic_thresh, corr_thresh=corr_thresh)
        mapping[reg] = selected
        summary.to_csv(DATA_DIR / f'pipeline_factor_summary_{reg}_{ts}.csv', index=False, encoding='utf-8-sig')

    out_json = DATA_DIR / f'pipeline_selected_factors_{ts}.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    return mapping, out_json


def neutralize_dataframe(df, factor_cols: List[str], shrink: float, industries: Optional[Set[str]]):
    if shrink <= 0 or 'industry' not in df.columns:
        return df
    shrink = max(0.0, min(1.0, shrink))
    if shrink == 0:
        return df

    group_keys = ['date', 'industry']

    def adjust(group):
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

    df_neutral = df.groupby(group_keys, group_keys=False).apply(adjust)
    return df_neutral.reset_index(drop=True)


def run_predictive_model(regime_file: Path,
                         mapping: Dict[str, List[str]],
                         start_oos: str,
                         train_window: int,
                         alpha: float,
                         top_n: int,
                         bottom_n: int,
                         cost_bps: float,
                         neutral_shrink: float,
                         neutral_industries: Optional[Set[str]]) -> dict:
    import pandas as pd

    df = pd.read_csv(regime_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'stock_code']).reset_index(drop=True)

    factor_cols = sorted({f for cols in mapping.values() for f in cols})
    df = neutralize_dataframe(df, factor_cols, neutral_shrink, neutral_industries)

    preds = walk_forward_predict_regime_specific(
        df,
        mapping,
        start_oos=start_oos,
        train_window_days=train_window,
        alpha=alpha,
    )
    if preds.empty:
        raise RuntimeError('Predictive model produced no predictions (insufficient data)')
    # 丰富预测：合并行业信息并计算“行业内分位”
    if 'industry' in df.columns:
        import pandas as _pd
        meta_cols = ['date', 'stock_code', 'industry']
        try:
            preds = preds.merge(df[meta_cols].drop_duplicates(), on=['date', 'stock_code'], how='left')
            # 行业内分位（按 y_pred 降序，越高越靠前；返回0-1）
            preds['ind_count'] = preds.groupby(['date', 'industry'])['stock_code'].transform('count')
            preds['ind_rank'] = preds.groupby(['date', 'industry'])['y_pred'].rank(ascending=False, method='min')
            preds['ind_rank_pct'] = 1.0 - (preds['ind_rank'] - 1) / preds['ind_count'].clip(lower=1)
        except Exception:
            # 安全降级：忽略行业分位计算
            pass
    metrics = evaluate_predictions(preds, top_n=top_n, bottom_n=bottom_n, cost_bps=cost_bps)
    pred_file, report_file = save_outputs(preds, metrics)

    # Additional summary metrics using performance reporter helpers
    pred_df = load_predictions(Path(pred_file))
    ic_series = compute_ic_series(pred_df)
    ts = compute_portfolio_timeseries(pred_df, top_n, bottom_n, cost_bps)
    perf_metrics = compute_summary_metrics(ts, ic_series)
    regime_contrib = compute_regime_contributions(pred_df, df[['date', 'regime']], top_n, bottom_n, cost_bps)
    regime_contrib.to_csv(Path(report_file).with_name(Path(report_file).stem + '_regime.csv'), index=False, encoding='utf-8-sig')

    summary = {
        'metrics': metrics,
        'performance_metrics': perf_metrics,
        'prediction_file': pred_file,
        'report_file': report_file,
    }
    summary_path = Path(report_file).with_suffix('.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _compute_half_life_from_sets(daily_sets: list[dict]) -> dict:
    # Track survival ages per side
    def side_half_life(key: str) -> float:
        ages: dict[str, int] = {}
        finished: list[int] = []
        for item in daily_sets:
            curr: set[str] = item[key]
            # increment ages for existing holdings
            for code in list(ages.keys()):
                if code in curr:
                    ages[code] += 1
                else:
                    finished.append(ages.pop(code))
            # add new holdings
            for code in curr:
                if code not in ages:
                    ages[code] = 1
        # finalise remaining
        finished.extend(ages.values())
        if not finished:
            return float('nan')
        arr = sorted(finished)
        mid = len(arr) // 2
        if len(arr) % 2 == 1:
            return float(arr[mid])
        return float((arr[mid - 1] + arr[mid]) / 2.0)

    return {
        'half_life_long_days': side_half_life('long'),
        'half_life_short_days': side_half_life('short'),
    }


def run_execution_layer(predictions_path: Path,
                        strategy: str,
                        top_n: int,
                        bottom_n: int,
                        cost_bps: float,
                        delta: int = 15,
                        ema_span: int = 4,
                        k: int = 5,
                        swap_cap: float = 0.2) -> tuple[Path, Path]:
    """Apply execution strategy on predictions and save timeseries + metrics.
    Returns (timeseries_csv, metrics_json).
    """
    import pandas as pd
    preds = load_predictions(predictions_path)
    ts = None
    ic_series = None
    param_desc = None
    details = None
    if strategy == 'baseline':
        ts, details = baseline_daily(preds, top_n=top_n, bottom_n=bottom_n, cost_bps=cost_bps, return_details=True)
        ic_series = compute_ic_series_with_score(preds, 'y_pred')
        param_desc = None
    elif strategy == 'hysteresis':
        ts, details = hysteresis_bands(preds, top_n=top_n, bottom_n=bottom_n, cost_bps=cost_bps, delta=delta, return_details=True)
        ic_series = compute_ic_series_with_score(preds, 'y_pred')
        param_desc = f'delta={delta}'
    elif strategy == 'ema_hysteresis':
        ts, scored, details = ema_hysteresis_combo(preds, top_n=top_n, bottom_n=bottom_n, cost_bps=cost_bps, ema_span=ema_span, delta=delta, return_details=True)
        ic_series = compute_ic_series_with_score(scored, 'y_score')
        param_desc = f'ema_span={ema_span},delta={delta}'
    elif strategy == 'lowfreq':
        ts, details = low_freq_rebalance(preds, top_n=top_n, bottom_n=bottom_n, cost_bps=cost_bps, k=k, return_details=True)
        ic_series = compute_ic_series_with_score(preds, 'y_pred')
        param_desc = f'k={k}'
    elif strategy == 'swapcap':
        ts, details = swap_cap_limited(preds, top_n=top_n, bottom_n=bottom_n, cost_bps=cost_bps, swap_cap_ratio=swap_cap, return_details=True)
        ic_series = compute_ic_series_with_score(preds, 'y_pred')
        param_desc = f'swap_cap={swap_cap}'
    else:
        raise ValueError(f'Unsupported execution strategy: {strategy}')

    if ts is None or ts.empty:
        raise RuntimeError('Execution layer produced empty timeseries — check predictions and parameters')

    metrics = compute_summary_metrics(ts, ic_series)
    # Enrich with execution profile
    enrich = {
        'avg_turnover': float(ts['turnover'].mean()),
        'avg_overlap_long': float(ts.get('overlap_long', pd.Series([np.nan])).mean()),
        'avg_overlap_short': float(ts.get('overlap_short', pd.Series([np.nan])).mean()),
        'avg_added_long': float(ts.get('added_long', pd.Series([np.nan])).mean()),
        'avg_added_short': float(ts.get('added_short', pd.Series([np.nan])).mean()),
    }
    if details:
        half_life = _compute_half_life_from_sets(details)
    else:
        half_life = {}
    ts_path = Path('data') / f"pipeline_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}_timeseries.csv"
    ts.to_csv(ts_path, index=False, encoding='utf-8-sig')
    metrics_path = ts_path.with_name(ts_path.name.replace('_timeseries.csv', '_metrics.json'))
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'strategy': strategy, 'param': param_desc, 'metrics': metrics, 'execution_profile': {**enrich, **half_life}}, f, ensure_ascii=False, indent=2)
    return ts_path, metrics_path


def maybe_run_turnover_grid(run: bool, predictions_path: Path, logger) -> Path | None:
    if not run:
        return None
    cmd = [
        'python3', 'analysis/turnover_strategy_grid.py',
        '--predictions', str(predictions_path),
        '--fine-tune'
    ]
    logger(f'Running turnover strategy grid: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)
    latest = sorted(Path('data').glob('turnover_strategy_grid_*.csv'), key=lambda p: p.stat().st_mtime)
    return latest[-1] if latest else None


def maybe_run_cost_grid(run: bool, regime_file: Path, mapping_file: Path,
                        start_oos: str, top_n_list: str, logger,
                        neutral_shrink: float, neutral_industries: Optional[Set[str]]) -> Path | None:
    if not run:
        return None
    import subprocess
    cmd = [
        'python3', 'analysis/cost_sensitivity_grid.py',
        '--data-file', str(regime_file),
        '--mapping-file', str(mapping_file),
        '--start-oos', start_oos,
        '--top-ns', top_n_list,
    ]
    if neutral_shrink > 0:
        cmd.extend(['--industry-neutral', '--neutral-shrink', str(neutral_shrink)])
        if neutral_industries:
            cmd.extend(['--neutral-industries', ','.join(neutral_industries)])
    logger(f'Running cost sensitivity grid: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)
    latest = sorted(DATA_DIR.glob('cost_sensitivity_grid_*.csv'), key=lambda p: p.stat().st_mtime)
    return latest[-1] if latest else None


def main():
    parser = argparse.ArgumentParser(description='Run the full volatility-regime momentum pipeline')
    parser.add_argument('--standardisation', type=str, default='zscore', choices=['zscore', 'rank'])
    # 默认使用全量500只股票的股票池
    parser.add_argument('--universe', type=str, default='stock_universe.csv')
    parser.add_argument('--recompute-factors', action='store_true')
    parser.add_argument('--recompute-regime', action='store_true')
    parser.add_argument('--ic-thresh', type=float, default=0.01)
    parser.add_argument('--corr-thresh', type=float, default=0.8)
    parser.add_argument('--start-oos', type=str, default='2022-01-01')
    parser.add_argument('--train-window', type=int, default=756)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--bottom-n', type=int, default=30)
    parser.add_argument('--cost-bps', type=float, default=0.0005)
    parser.add_argument('--run-cost-grid', action='store_true')
    parser.add_argument('--cost-grid-top-ns', type=str, default='20,25,30')
    parser.add_argument('--neutral-shrink', type=float, default=0.0,
                        help='Shrink factor (0-1) for industry neutralisation; 0=disabled')
    parser.add_argument('--neutral-industries', type=str, default=None,
                        help='Comma-separated industry names to neutralise (default: all when shrink>0)')
    # Execution layer (optional)
    parser.add_argument('--execution-strategy', type=str, default='none',
                        choices=['none', 'baseline', 'hysteresis', 'ema_hysteresis', 'lowfreq', 'swapcap'])
    parser.add_argument('--delta', type=int, default=15)
    parser.add_argument('--ema-span', type=int, default=4)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--swap-cap', type=float, default=0.2)
    parser.add_argument('--run-turnover-grid', action='store_true', help='After predictions, run turnover grid on the predictions file')
    args = parser.parse_args()

    def log(msg: str):
        print(f'[PIPELINE] {msg}')

    industries: Optional[Set[str]] = None
    if args.neutral_industries:
        industries = {item.strip() for item in args.neutral_industries.split(',') if item.strip()}

    # Step 1: factor dataset
    if args.recompute_factors:
        log('Recomputing factor dataset...')
        factor_file = recompute_factors(args.standardisation, args.universe)
    else:
        default_path = DATA_DIR / ('simple_factor_data_rank.csv' if args.standardisation == 'rank' else 'simple_factor_data.csv')
        if not default_path.exists():
            log('Requested reuse of factor data but default file missing; recomputing instead.')
            factor_file = recompute_factors(args.standardisation, args.universe)
        else:
            factor_file = default_path
    log(f'Using factor file: {factor_file}')

    # Step 2: regime classification
    if args.recompute_regime:
        log('Recomputing volatility regimes...')
        regime_file = recompute_regime(factor_file)
    else:
        latest_regime = find_latest_regime_file()
        if latest_regime is None:
            log('No existing regime file found; recomputing.')
            regime_file = recompute_regime(factor_file)
        else:
            regime_file = Path(latest_regime)
    log(f'Using regime file: {regime_file}')

    # Step 3: factor selection
    log('Selecting factors by regime...')
    mapping, mapping_path = run_factor_selection(regime_file, args.ic_thresh, args.corr_thresh,
                                                args.neutral_shrink, industries)
    log(f'Selected factors saved to: {mapping_path}')

    # Step 4: predictive model
    log('Running predictive model...')
    summary = run_predictive_model(
        regime_file,
        mapping,
        start_oos=args.start_oos,
        train_window=args.train_window,
        alpha=args.alpha,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        cost_bps=args.cost_bps,
        neutral_shrink=args.neutral_shrink,
        neutral_industries=industries,
    )
    log(f"Predictions: {summary['prediction_file']}")
    log(f"Report: {summary['report_file']}")

    # Optional: execution layer
    if args.execution_strategy != 'none':
        log(f"Applying execution strategy: {args.execution_strategy}")
        ts_path, metrics_path = run_execution_layer(
            Path(summary['prediction_file']),
            strategy=args.execution_strategy,
            top_n=args.top_n,
            bottom_n=args.bottom_n,
            cost_bps=args.cost_bps,
            delta=args.delta,
            ema_span=args.ema_span,
            k=args.k,
            swap_cap=args.swap_cap,
        )
        log(f'Execution timeseries saved to: {ts_path}')
        log(f'Execution metrics saved to: {metrics_path}')
    
    # Optional: turnover grid on predictions
    grid2 = maybe_run_turnover_grid(args.run_turnover_grid, Path(summary['prediction_file']), log)
    if grid2:
        log(f'Turnover grid saved to: {grid2}')

    # Step 5: optional cost grid
    grid_path = maybe_run_cost_grid(
        args.run_cost_grid,
        regime_file,
        mapping_path,
        args.start_oos,
        args.cost_grid_top_ns,
        log,
        args.neutral_shrink,
        industries,
    )
    if grid_path:
        log(f'Cost grid saved to: {grid_path}')

    log('Pipeline complete.')


if __name__ == '__main__':
    main()
