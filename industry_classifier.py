#!/usr/bin/env python3
"""
行业分类更新脚本

读取股票池（默认 stock_universe_selected.csv），调用 AkShare 获取所属行业，
更新 CSV 中的 industry 列，并输出 data/industry_mapping.csv 备后续使用。
"""

import akshare as ak
import pandas as pd
import time
from pathlib import Path


def load_bulk_classification() -> dict:
    mapping = {}
    try:
        df = ak.stock_industry_classification_cninfo()
        if df is not None and not df.empty:
            code_col = None
            for col in ['公司代码', '证券代码', '股票代码']:
                if col in df.columns:
                    code_col = col
                    break
            if code_col is None:
                return mapping
            industry_col = None
            for col in ['行业名称', '一级行业', '行业']:
                if col in df.columns:
                    industry_col = col
                    break
            if industry_col is None:
                return mapping
            temp = df[[code_col, industry_col]].dropna()
            for _, row in temp.iterrows():
                mapping[str(row[code_col]).zfill(6)] = str(row[industry_col])
    except Exception:
        pass
    return mapping


def fetch_industry_individual(code: str) -> str:
    try:
        info = ak.stock_individual_info_em(code)
        if info is not None and not info.empty:
            row = info[info['item'] == '所属行业']
            if not row.empty:
                return row['value'].iloc[0]
    except Exception:
        return '未分类'
    return '未分类'


def main():
    import argparse
    parser = argparse.ArgumentParser(description='更新股票行业分类')
    parser.add_argument('--universe', type=str, default='stock_universe_selected.csv')
    parser.add_argument('--output-mapping', type=str, default='data/industry_mapping.csv')
    parser.add_argument('--delay', type=float, default=0.3, help='API 调用间隔秒')
    args = parser.parse_args()

    universe_path = Path(args.universe)
    if not universe_path.exists():
        print(f'❌ 股票池文件不存在: {universe_path}')
        return

    df = pd.read_csv(universe_path, dtype={'code': str})
    bulk_map = load_bulk_classification()
    cons_map = {}
    try:
        board_df = ak.stock_board_industry_name_em()
        for _, row in board_df.iterrows():
            board_code = row.get('板块代码')
            board_name = row.get('板块名称')
            if not board_code:
                continue
            try:
                constituents = ak.stock_board_industry_cons_em(board_code)
                for _, crow in constituents.iterrows():
                    c = str(crow.get('代码', '')).zfill(6)
                    if c:
                        cons_map[c] = board_name
            except Exception:
                continue
            time.sleep(args.delay)
    except Exception:
        pass

    industries = []
    for i, code in enumerate(df['code'].astype(str).str.zfill(6), 1):
        ind = cons_map.get(code) or bulk_map.get(code)
        if not ind or ind in ['nan', 'None']:
            ind = fetch_industry_individual(code)
            time.sleep(args.delay)
        industries.append(ind if ind else '未分类')
        print(f'[{i}/{len(df)}] {code}: {industries[-1]}')

    df['industry'] = industries
    df.to_csv(universe_path, index=False, encoding='utf-8-sig')
    mapping_path = Path(args.output_mapping)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    df[['code', 'industry']].to_csv(mapping_path, index=False, encoding='utf-8-sig')
    print(f'💾 已更新股票池行业列，并保存映射: {mapping_path}')


if __name__ == '__main__':
    main()
