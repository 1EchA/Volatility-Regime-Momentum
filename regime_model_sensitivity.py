#!/usr/bin/env python3
"""
分制度岭回归敏感性分析

组合：
- top_n / bottom_n ∈ {20, 30}
- cost_bps ∈ {0.0003, 0.0005, 0.0010}
- 使用指定的 train_window (默认756) 与 alpha (默认1.0)

输出：
- data/regime_model_sensitivity_YYYYMMDD_HHMMSS.csv
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json

from predictive_model import load_data, walk_forward_predict_regime_specific, evaluate_predictions


def find_latest_mapping() -> str | None:
    cands = sorted(Path('data').glob('selected_factors_by_regime_*.json'))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])


def main():
    import argparse
    parser = argparse.ArgumentParser(description='分制度岭回归敏感性分析')
    parser.add_argument('--data-file', type=str, default='data/simple_factor_data.csv')
    parser.add_argument('--mapping', type=str, default=None)
    parser.add_argument('--start-oos', type=str, default='2022-01-01')
    parser.add_argument('--train-window', type=int, default=756)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--industry-neutral', action='store_true')
    args = parser.parse_args()

    df = load_data(args.data_file, use_regime=True)
    if args.industry_neutral:
        if 'industry' not in df.columns:
            print('⚠️ 数据缺少 industry 列，无法行业中性化，忽略该选项')
            args.industry_neutral = False
        else:
            factor_cols = [c for c in df.columns if c.endswith('_std')]
            df[factor_cols] = df.groupby(['date', 'industry'])[factor_cols].transform(lambda x: x - x.mean())
            if 'forward_return_1d' in df.columns:
                df['forward_return_1d'] = df.groupby(['date', 'industry'])['forward_return_1d'].transform(lambda x: x - x.mean())
    mapping_file = args.mapping or find_latest_mapping()
    if not mapping_file:
        print('❌ 未找到制度因子映射')
        return
    with open(mapping_file, 'r', encoding='utf-8') as f:
        regime_mapping = json.load(f)

    top_bottom_options = [(20, 20), (30, 30)]
    cost_options = [0.0003, 0.0005, 0.0010]

    pred_cache = walk_forward_predict_regime_specific(
        df, regime_mapping,
        start_oos=args.start_oos,
        train_window_days=args.train_window,
        alpha=args.alpha
    )
    if pred_cache.empty:
        print('❌ 无预测结果')
        return

    rows = []
    for top_n, bottom_n in top_bottom_options:
        for cost in cost_options:
            metrics = evaluate_predictions(pred_cache, top_n=top_n, bottom_n=bottom_n, cost_bps=cost)
            rows.append({
                'top_n': top_n,
                'bottom_n': bottom_n,
                'cost_bps': cost,
                **metrics
            })

    res = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path('data') / f'regime_model_sensitivity_{ts}.csv'
    res.to_csv(out, index=False, encoding='utf-8-sig')
    print(res)
    print(f'💾 保存: {out}')


if __name__ == '__main__':
    main()
