#!/usr/bin/env python3
"""
制度内因子筛选器

步骤：
1) 自动读取最新的制度数据文件 data/volatility_regime_data_*.csv
2) 识别 *_std 因子列
3) 按制度分别计算每日Spearman IC → ic_mean, t, p, 胜率
4) 按阈值筛选（默认 |IC|≥0.01 且 p<0.05），并在制度内按相关性<0.8去重
5) 输出：
   - data/selected_factors_by_regime_YYYYMMDD_HHMMSS.json（制度→因子列表）
   - 各制度的txt清单与csv汇总
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import json


def find_latest_regime_file() -> str | None:
    data_dir = Path('data')
    cands = sorted(data_dir.glob('volatility_regime_data_*.csv'))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])


def detect_factors(df: pd.DataFrame) -> list[str]:
    exclude = {'date', 'stock_code', 'close', 'daily_return',
               'forward_return_1d', 'forward_return_5d', 'forward_return_10d',
               'regime', 'volatility', 'market_return'}
    return [c for c in df.columns if c.endswith('_std') and c not in exclude]


def daily_ic(df: pd.DataFrame, factor: str, return_col: str = 'forward_return_1d') -> pd.Series:
    ics = []
    dates = []
    for dt, g in df.groupby('date'):
        x = g[factor]
        y = g[return_col]
        mask = x.notna() & y.notna()
        if mask.sum() >= 10:
            ic, _ = stats.spearmanr(x[mask], y[mask])
            if not np.isnan(ic):
                ics.append(ic)
                dates.append(dt)
    return pd.Series(ics, index=dates)


def select_in_regime(df_reg: pd.DataFrame, factor_cols: list[str], ic_thresh=0.01, corr_thresh=0.8):
    rows = []
    for f in factor_cols:
        ics = daily_ic(df_reg, f)
        if len(ics) < 60:
            continue
        ic_mean = ics.mean()
        ic_std = ics.std()
        t_stat, p_val = stats.ttest_1samp(ics, 0)
        win_rate = (ics > 0).mean()
        rows.append({'factor': f, 'ic_mean': ic_mean, 'ic_std': ic_std, 't': t_stat, 'p': p_val, 'win_rate': win_rate, 'n_days': len(ics)})
    summary = pd.DataFrame(rows).sort_values('ic_mean', key=np.abs, ascending=False)
    pool = summary[(summary['p'] < 0.05) & (summary['ic_mean'].abs() >= ic_thresh)].copy()
    selected = []
    if not pool.empty:
        sub = df_reg[['date', 'stock_code'] + pool['factor'].tolist()].dropna()
        corr = sub[pool['factor'].tolist()].corr().abs()
        for f in pool['factor']:
            if not selected:
                selected.append(f)
                continue
            high_corr = any(corr.loc[f, s] >= corr_thresh for s in selected if f in corr.index and s in corr.columns)
            if not high_corr:
                selected.append(f)
    return selected, summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='制度内因子筛选器')
    parser.add_argument('--regime-file', type=str, default=None)
    parser.add_argument('--ic-thresh', type=float, default=0.01)
    parser.add_argument('--corr-thresh', type=float, default=0.8)
    args = parser.parse_args()

    regime_file = args.regime_file or find_latest_regime_file()
    if not regime_file:
        print('❌ 未找到制度数据文件')
        return
    df = pd.read_csv(regime_file)
    df['date'] = pd.to_datetime(df['date'])
    factors = detect_factors(df)
    regimes = [r for r in df['regime'].dropna().unique().tolist() if r != '未分类']
    print(f'制度: {regimes}\n候选因子: {len(factors)}')

    mapping = {}
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('data')
    for r in regimes:
        sub = df[df['regime'] == r]
        sel, summary = select_in_regime(sub, factors, ic_thresh=args.ic_thresh, corr_thresh=args.corr_thresh)
        mapping[r] = sel
        # 保存各制度报告
        out_txt = out_dir / f'selected_factors_{r}_{ts}.txt'
        with open(out_txt, 'w', encoding='utf-8') as f:
            for s in sel:
                f.write(s + '\n')
        summary.to_csv(out_dir / f'factor_selection_{r}_{ts}.csv', index=False, encoding='utf-8-sig')
        print(f'[{r}] 选中{len(sel)}个: {sel}')

    # 汇总JSON
    out_json = out_dir / f'selected_factors_by_regime_{ts}.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f'💾 制度因子映射: {out_json}')


if __name__ == '__main__':
    main()

