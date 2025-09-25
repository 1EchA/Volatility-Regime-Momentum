#!/usr/bin/env python3
"""
稳健性验证脚本：
- 多个样本外起点（start_oos）
- 多个训练窗口（train_window）
- 固定执行策略（滞后带或EMA+滞后组合）与参数

流程：
1) 载入最新（或指定）制度标注数据与因子映射；
2) 运行分制度滚动预测，得到 predictions DataFrame；
3) 在给定执行策略与参数下计算组合时序；
4) 输出汇总指标到 data/robustness_summary_<ts>.csv。
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from datetime import datetime
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from predictive_model import (
    detect_factor_columns,
    walk_forward_predict,
    walk_forward_predict_regime_specific,
    find_latest_regime_file,
)
from regime_model_grid_search import find_latest_mapping
from analysis.execution_strategies import (
    baseline_daily, hysteresis_bands, ema_hysteresis_combo, compute_ic_series_with_score
)
from analysis.performance_reporter import compute_summary_metrics


def _load_dataframe(path: Path | None) -> pd.DataFrame:
    if path is None:
        latest = find_latest_regime_file()
        if not latest:
            raise FileNotFoundError('未找到制度标注数据，请先运行主流水线生成。')
        path = Path(latest)
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    return df.sort_values(['date', 'stock_code']).reset_index(drop=True)


def _load_mapping(path: Path | None) -> dict | None:
    if path is None:
        latest = find_latest_mapping()
        if latest:
            path = Path(latest)
        else:
            return None
    with open(path, 'r', encoding='utf-8') as f:
        import json
        return json.load(f)


def run_once(df: pd.DataFrame,
             mapping: dict | None,
             start_oos: str,
             train_window: int,
             top_n: int,
             cost_bps: float,
             strategy: str,
             delta: int | None,
             ema_span: int | None) -> dict:
    # 1) 预测
    if mapping:
        preds = walk_forward_predict_regime_specific(
            df, mapping, start_oos=start_oos, train_window_days=train_window, alpha=1.0
        )
    else:
        fcols = detect_factor_columns(df)
        preds = walk_forward_predict(
            df, fcols, start_oos=start_oos, train_window_days=train_window, alpha=1.0, use_regime=('regime' in df.columns)
        )
    if preds.empty:
        return {
            'start_oos': start_oos, 'train_window': train_window, 'strategy': strategy,
            'param': None, 'top_n': top_n, 'cost_bps': cost_bps,
            'ic_mean': np.nan, 'ic_ir': np.nan, 'ls_ann': np.nan, 'ls_ir': np.nan,
            'avg_turnover': np.nan, 'max_drawdown': np.nan
        }

    # 2) 执行策略 & IC
    if strategy == 'hysteresis':
        ts = hysteresis_bands(preds, top_n=top_n, bottom_n=top_n, cost_bps=cost_bps, delta=int(delta or 15))
        ic_series = compute_ic_series_with_score(preds, 'y_pred')
        param = f'delta={int(delta or 15)}'
    elif strategy == 'combo':
        ts, scored = ema_hysteresis_combo(preds, top_n=top_n, bottom_n=top_n, cost_bps=cost_bps,
                                          ema_span=int(ema_span or 4), delta=int(delta or 15))
        ic_series = compute_ic_series_with_score(scored, 'y_score')
        param = f'ema={int(ema_span or 4)},delta={int(delta or 15)}'
    else:  # baseline
        ts = baseline_daily(preds, top_n=top_n, bottom_n=top_n, cost_bps=cost_bps)
        ic_series = compute_ic_series_with_score(preds, 'y_pred')
        param = None

    if ts.empty:
        return {
            'start_oos': start_oos, 'train_window': train_window, 'strategy': strategy,
            'param': param, 'top_n': top_n, 'cost_bps': cost_bps,
            'ic_mean': np.nan, 'ic_ir': np.nan, 'ls_ann': np.nan, 'ls_ir': np.nan,
            'avg_turnover': np.nan, 'max_drawdown': np.nan
        }

    # 3) 指标
    metrics = compute_summary_metrics(ts, ic_series)
    return {
        'start_oos': start_oos,
        'train_window': train_window,
        'strategy': strategy,
        'param': param,
        'top_n': top_n,
        'cost_bps': cost_bps,
        **metrics
    }


def main():
    parser = argparse.ArgumentParser(description='稳健性验证：多OOS起点/训练窗口 + 执行策略参数')
    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--mapping-file', type=str, default=None)
    parser.add_argument('--start-oos', type=str, default='2021-01-01,2022-01-01,2022-07-01')
    parser.add_argument('--train-windows', type=str, default='756,900,1008')
    parser.add_argument('--strategy', type=str, default='hysteresis', choices=['baseline', 'hysteresis', 'combo'])
    parser.add_argument('--top-n', type=int, default=40)
    parser.add_argument('--cost-bps', type=float, default=0.0005)
    parser.add_argument('--delta', type=int, default=15)
    parser.add_argument('--ema-span', type=int, default=4)
    args = parser.parse_args()

    df = _load_dataframe(Path(args.data_file) if args.data_file else None)
    mapping = _load_mapping(Path(args.mapping_file) if args.mapping_file else None)

    starts = [s.strip() for s in args.start_oos.split(',') if s.strip()]
    wins = [int(float(w)) for w in args.train_windows.split(',') if w.strip()]

    records = []
    for start_oos, train_window in product(starts, wins):
        rec = run_once(
            df, mapping, start_oos, train_window, args.top_n, args.cost_bps,
            args.strategy, args.delta, args.ema_span
        )
        records.append(rec)

    res = pd.DataFrame(records)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'data' / f'robustness_summary_{ts}.csv'
    res.to_csv(out, index=False, encoding='utf-8-sig')
    print('✅ Robustness validation done')
    print('Saved:', out)
    if not res.empty:
        print(res[['start_oos','train_window','strategy','param','ls_ir','ls_ann','avg_turnover','max_drawdown']])


if __name__ == '__main__':
    main()

