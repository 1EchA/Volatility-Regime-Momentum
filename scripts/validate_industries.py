#!/usr/bin/env python3
from __future__ import annotations

import glob
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def main():
    # 1) latest factor/regime files if exist
    latest_reg = sorted(DATA.glob('volatility_regime_data_*.csv'))
    latest_fac = sorted(DATA.glob('simple_factor_data*_*.csv'))
    for label, paths in [('regime', latest_reg), ('factor', latest_fac)]:
        if not paths:
            print(f'no {label} file found');
            continue
        p = paths[-1]
        df = pd.read_csv(p, usecols=['stock_code','industry'])
        df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        nunq = df['stock_code'].nunique()
        null_ratio = df['industry'].isna().mean()
        unclassified = (df['industry'].astype(str) == '未分类').mean()
        print(f'{label}: {p.name}  stocks={nunq}  industry_null={null_ratio:.2%}  未分类={unclassified:.2%}')
        print('示例行业：', df['industry'].dropna().unique()[:10])

    # 2) from stock_universe.csv
    up = ROOT / 'stock_universe.csv'
    if up.exists():
        u = pd.read_csv(up, dtype={'code':str})
        ind_col = 'industry' if 'industry' in u.columns else ('行业' if '行业' in u.columns else None)
        if ind_col:
            null_ratio = u[ind_col].isna().mean()
            unclassified = (u[ind_col].astype(str) == '未分类').mean()
            print(f'universe: rows={len(u)}  industry_null={null_ratio:.2%}  未分类={unclassified:.2%}')
        else:
            print('universe: no industry column')


if __name__ == '__main__':
    main()

