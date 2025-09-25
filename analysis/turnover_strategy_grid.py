#!/usr/bin/env python3
"""
在固定预测明细上，对低换手执行策略进行并行网格测试：
- 方案A：低频再平衡（k）
- 方案B：进出场滞后带（delta）
- 方案C：EMA 信号平滑（ema_span）
- 方案D：换手上限（swap_cap_ratio）

每个方案跑 3×3×3（TopN×成本×策略参数）共27组；四方案合计108组。
输出：data/turnover_strategy_grid_<ts>.csv
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

from analysis.performance_reporter import compute_summary_metrics
from analysis.execution_strategies import (
    baseline_daily,
    low_freq_rebalance,
    hysteresis_bands,
    ema_smoothed,
    swap_cap_limited,
    ema_hysteresis_combo,
    compute_ic_series_with_score,
)


def _parse_list(text: str, cast=float) -> list:
    items = []
    for chunk in str(text).split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(cast(chunk))
    return items


def _find_latest_predictions() -> Path | None:
    data_dir = ROOT / 'data'
    cands = sorted(data_dir.glob('predictions_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    return df.sort_values(['date', 'stock_code']).reset_index(drop=True)


def run_grid(pred_df: pd.DataFrame,
             strategies: list[str],
             top_ns: Iterable[int],
             cost_bps_list: Iterable[float],
             k_vals: Iterable[int],
             delta_vals: Iterable[int],
             ema_spans: Iterable[int],
             swap_caps: Iterable[float],
             combo_ema_spans: Iterable[int] = None,
             combo_deltas: Iterable[int] = None) -> pd.DataFrame:
    records = []
    
    # 设置组合策略默认参数
    if combo_ema_spans is None:
        combo_ema_spans = [3, 4, 6]
    if combo_deltas is None:
        combo_deltas = [12, 15, 18]

    for strat in strategies:
        if strat == 'A':  # 低频再平衡
            for top_n, cost, k in product(top_ns, cost_bps_list, k_vals):
                ts = low_freq_rebalance(pred_df, top_n=top_n, bottom_n=top_n, cost_bps=cost, k=int(k))
                if ts.empty:
                    continue
                ic_series = compute_ic_series_with_score(pred_df, score_col='y_pred')
                metrics = compute_summary_metrics(ts, ic_series)
                records.append({
                    'strategy': 'A_lowfreq', 'param': int(k), 'param_name': 'k',
                    'top_n': int(top_n), 'bottom_n': int(top_n), 'cost_bps': float(cost),
                    **metrics,
                })

        elif strat == 'B':  # 滞后带
            for top_n, cost, delta in product(top_ns, cost_bps_list, delta_vals):
                ts = hysteresis_bands(pred_df, top_n=top_n, bottom_n=top_n, cost_bps=cost, delta=int(delta))
                if ts.empty:
                    continue
                ic_series = compute_ic_series_with_score(pred_df, score_col='y_pred')
                metrics = compute_summary_metrics(ts, ic_series)
                records.append({
                    'strategy': 'B_hysteresis', 'param': int(delta), 'param_name': 'delta',
                    'top_n': int(top_n), 'bottom_n': int(top_n), 'cost_bps': float(cost),
                    **metrics,
                })

        elif strat == 'C':  # EMA 平滑
            for top_n, cost, span in product(top_ns, cost_bps_list, ema_spans):
                ts, df_scored = ema_smoothed(pred_df, top_n=top_n, bottom_n=top_n, cost_bps=cost, ema_span=int(span))
                if ts.empty:
                    continue
                ic_series = compute_ic_series_with_score(df_scored.rename(columns={'y_score': 'score'}), score_col='score')
                metrics = compute_summary_metrics(ts, ic_series)
                records.append({
                    'strategy': 'C_ema', 'param': int(span), 'param_name': 'ema_span',
                    'top_n': int(top_n), 'bottom_n': int(top_n), 'cost_bps': float(cost),
                    **metrics,
                })

        elif strat == 'D':  # 换手上限
            for top_n, cost, cap in product(top_ns, cost_bps_list, swap_caps):
                ts = swap_cap_limited(pred_df, top_n=top_n, bottom_n=top_n, cost_bps=cost, swap_cap_ratio=float(cap))
                if ts.empty:
                    continue
                ic_series = compute_ic_series_with_score(pred_df, score_col='y_pred')
                metrics = compute_summary_metrics(ts, ic_series)
                records.append({
                    'strategy': 'D_swapcap', 'param': float(cap), 'param_name': 'swap_cap_ratio',
                    'top_n': int(top_n), 'bottom_n': int(top_n), 'cost_bps': float(cost),
                    **metrics,
                })
                
        elif strat == 'E':  # EMA+滞后带组合策略
            for top_n, cost, ema_span, delta in product(top_ns, cost_bps_list, combo_ema_spans, combo_deltas):
                ts, df_scored = ema_hysteresis_combo(
                    pred_df, top_n=top_n, bottom_n=top_n, cost_bps=cost, 
                    ema_span=int(ema_span), delta=int(delta)
                )
                if ts.empty:
                    continue
                ic_series = compute_ic_series_with_score(df_scored, score_col='y_score')
                metrics = compute_summary_metrics(ts, ic_series)
                records.append({
                    'strategy': 'E_combo', 'param': f'{int(ema_span)}_{int(delta)}', 
                    'param_name': 'ema_span_delta',
                    'ema_span': int(ema_span), 'delta': int(delta),
                    'top_n': int(top_n), 'bottom_n': int(top_n), 'cost_bps': float(cost),
                    **metrics,
                })

        elif strat == 'BASE':
            for top_n, cost in product(top_ns, cost_bps_list):
                ts = baseline_daily(pred_df, top_n=top_n, bottom_n=top_n, cost_bps=cost)
                if ts.empty:
                    continue
                ic_series = compute_ic_series_with_score(pred_df, score_col='y_pred')
                metrics = compute_summary_metrics(ts, ic_series)
                records.append({
                    'strategy': 'BASE_daily', 'param': None, 'param_name': None,
                    'top_n': int(top_n), 'bottom_n': int(top_n), 'cost_bps': float(cost),
                    **metrics,
                })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description='Turnover-optimised execution strategies grid on fixed predictions')
    parser.add_argument('--predictions', type=str, default=None,
                        help='预测明细CSV；默认取 data/ 下最新 predictions_*.csv')
    parser.add_argument('--strategies', type=str, default='A,B,C,D,E,BASE',
                        help='逗号分隔：A,B,C,D,E,BASE')
    parser.add_argument('--top-ns', type=str, default='20,30,40')
    parser.add_argument('--cost-bps', type=str, default='0.0003,0.0005,0.0008')
    parser.add_argument('--k-values', type=str, default='5,10,20')
    parser.add_argument('--delta-values', type=str, default='0,10,20')
    parser.add_argument('--ema-spans', type=str, default='3,5,10')
    parser.add_argument('--swap-caps', type=str, default='0.1,0.2,0.3')
    parser.add_argument('--combo-ema-spans', type=str, default='3,4,6',
                        help='EMA+滞后带组合策略的EMA参数')
    parser.add_argument('--combo-deltas', type=str, default='12,15,18',
                        help='EMA+滞后带组合策略的delta参数')
    # 精细扫描模式
    parser.add_argument('--fine-tune', action='store_true',
                        help='启用精细扫描模式（滞后带5×4×2=40组）')
    parser.add_argument('--combo-only', action='store_true',
                        help='只运行EMA+滞后带组合策略')
    args = parser.parse_args()

    pred_path = Path(args.predictions) if args.predictions else _find_latest_predictions()
    if not pred_path or not pred_path.exists():
        raise FileNotFoundError('未找到预测文件，请通过 --predictions 指定，或先运行预测流水线。')

    strategies = [s.strip().upper() for s in args.strategies.split(',') if s.strip()]
    top_ns = [int(v) for v in _parse_list(args.top_ns, cast=float)]
    cost_bps_list = [float(v) for v in _parse_list(args.cost_bps, cast=float)]
    k_vals = [int(v) for v in _parse_list(args.k_values, cast=float)]
    delta_vals = [int(v) for v in _parse_list(args.delta_values, cast=float)]
    ema_spans = [int(v) for v in _parse_list(args.ema_spans, cast=float)]
    swap_caps = [float(v) for v in _parse_list(args.swap_caps, cast=float)]
    combo_ema_spans = [int(v) for v in _parse_list(args.combo_ema_spans, cast=float)]
    combo_deltas = [int(v) for v in _parse_list(args.combo_deltas, cast=float)]
    
    # 精细扫描模式调整参数
    if args.fine_tune:
        strategies = ['B']  # 只运行滞后带精细扫描
        top_ns = [30, 35, 40, 45]  # 4个点
        delta_vals = [10, 12, 15, 18, 20]  # 5个点  
        cost_bps_list = [0.0003, 0.0005]  # 2个点
        print(f"✓ 精细扫描模式：滞后带策略 {len(delta_vals)}×{len(top_ns)}×{len(cost_bps_list)} = {len(delta_vals)*len(top_ns)*len(cost_bps_list)}组")
    
    if args.combo_only:
        strategies = ['E']  # 只运行组合策略
        top_ns = [35, 40]  # 聚焦最优TopN
        cost_bps_list = [0.0003, 0.0005]
        print(f"✓ 组合策略模式：EMA+滞后带 {len(combo_ema_spans)}×{len(combo_deltas)}×{len(top_ns)}×{len(cost_bps_list)} = {len(combo_ema_spans)*len(combo_deltas)*len(top_ns)*len(cost_bps_list)}组")

    pred_df = load_predictions(pred_path)
    results = run_grid(
        pred_df, strategies, top_ns, cost_bps_list, k_vals, delta_vals, 
        ema_spans, swap_caps, combo_ema_spans, combo_deltas
    )
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = ROOT / 'data' / f'turnover_strategy_grid_{ts}.csv'
    results.to_csv(out_path, index=False, encoding='utf-8-sig')

    # 简要汇总
    if not results.empty:
        print('✅ Turnover strategy grid complete')
        print(f'  Saved: {out_path}')
        by_ir = results.sort_values('ls_ir', ascending=False).head(10)
        by_ann = results.sort_values('ls_ann', ascending=False).head(10)
        print('\nTop 10 by LS_IR:')
        print(by_ir[['strategy','param_name','param','top_n','cost_bps','ls_ir','ls_ann','avg_turnover','max_drawdown']])
        print('\nTop 10 by LS_Ann:')
        print(by_ann[['strategy','param_name','param','top_n','cost_bps','ls_ann','ls_ir','avg_turnover','max_drawdown']])
    else:
        print('No results produced (empty predictions or parameter mismatch).')


if __name__ == '__main__':
    main()

