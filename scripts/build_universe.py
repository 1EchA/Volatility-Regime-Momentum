#!/usr/bin/env python3
"""
Build a 500-stock universe and attach industry classification.

Logic (data-driven + robust fallbacks):
- Candidates: union of (a) local data/*.csv codes, (b) AkShare全A代码（可选），(c) 既有 stock_universe.csv
- Quality filters (last N years):
  * trading_days >= --min-days (default 504)
  * median_daily_amount >= --min-amt (default 5e7 CNY) when有 amount 列
  * exclude ST/*ST by name 标记（若有）
- Ranking: prefer 流通市值 (from AkShare spot) → 次选 median_daily_amount → 次选有效交易天数
- Industry mapping: 优先 data/industry_mapping.csv → 其次 stock_universe.csv 的列 → AkShare 板块/个股接口 → 名称关键词兜底

Outputs: stock_universe.csv with columns [code,name,行业,市值排名,总市值_亿元,流通市值_亿元,样本标识,更新时间]
and data/industry_mapping.csv for downstream use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def _infer_industry_from_name(name: str | None) -> str:
    if not name:
        return '未分类'
    n = str(name)
    rules = [
        ('银行', '银行'), ('证券', '证券'), ('保险', '保险'), ('信托', '非银金融'),
        ('白酒', '食品饮料'), ('啤酒', '食品饮料'), ('饮料', '食品饮料'), ('乳业', '食品饮料'),
        ('家电', '家用电器'), ('电器', '家用电器'),
        ('半导体', '半导体'), ('芯片', '半导体'), ('集成电路', '半导体'), ('电子', '电子'),
        ('汽车', '汽车'), ('整车', '汽车'), ('零部件', '汽车零部件'),
        ('化工', '化工'), ('医药', '医药生物'), ('制药', '医药生物'), ('生物', '医药生物'),
        ('计算机', '计算机'), ('软件', '计算机'), ('通信', '通信'), ('互联网', '计算机'),
        ('煤', '煤炭'), ('钢铁', '钢铁'), ('有色', '有色金属'), ('黄金', '有色金属'),
        ('电力', '公用事业'), ('公用', '公用事业'), ('水务', '公用事业'), ('燃气', '公用事业'),
        ('建筑', '建筑装饰'), ('建材', '建筑材料'), ('地产', '房地产'), ('物业', '房地产'),
        ('航运', '交通运输'), ('机场', '交通运输'), ('港口', '交通运输'), ('高速', '交通运输'),
        ('军工', '国防军工'), ('航天', '国防军工'), ('航空', '国防军工'),
        ('石油', '石油石化'), ('石化', '石油石化'), ('新能源', '电力设备'), ('光伏', '电力设备'), ('风电', '电力设备')
    ]
    for kw, ind in rules:
        if kw in n:
            return ind
    return '未分类'


def load_local_candidates() -> pd.DataFrame:
    """Scan data/*.csv and build basic stats for each code from local OHLCV files."""
    rows = []
    for p in DATA.glob('*.csv'):
        stem = p.stem
        if not stem.isdigit() or len(stem) not in (6, 7):
            continue
        code = stem.zfill(6)
        try:
            df = pd.read_csv(p)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
            amt = pd.to_numeric(df.get('amount', pd.Series([0]*len(df))), errors='coerce')
            med_amt = float(amt.median(skipna=True)) if len(amt) else 0.0
            rows.append({'code': code, 'days': len(df), 'median_amount': med_amt})
        except Exception:
            continue
    return pd.DataFrame(rows)


def load_spot_info() -> pd.DataFrame:
    """Try to load spot info from AkShare; return empty DF on failure."""
    try:
        import akshare as ak
        spot = ak.stock_zh_a_spot_em()
        # 兼容列名
        code_col = next((c for c in ['代码','股票代码','证券代码','code'] if c in spot.columns), None)
        name_col = next((c for c in ['名称','股票名称','证券名称','name'] if c in spot.columns), None)
        ffmv_col = next((c for c in ['流通市值','流通市值-亿','流通市值(亿)','流通市值_亿元','流通值'] if c in spot.columns), None)
        mv_col = next((c for c in ['总市值','总市值-亿','总市值(亿)','总市值_亿元'] if c in spot.columns), None)
        out = pd.DataFrame()
        out['code'] = spot[code_col].astype(str).str.zfill(6)
        out['name'] = spot[name_col].astype(str)
        out['流通市值_亿元'] = pd.to_numeric(spot.get(ffmv_col, pd.NA), errors='coerce')
        out['总市值_亿元'] = pd.to_numeric(spot.get(mv_col, pd.NA), errors='coerce')
        return out
    except Exception:
        return pd.DataFrame(columns=['code','name','流通市值_亿元','总市值_亿元'])


def load_existing_universe() -> pd.DataFrame:
    """Load existing stock_universe.csv if present to expand candidate pool."""
    up = ROOT / 'stock_universe.csv'
    if not up.exists():
        return pd.DataFrame(columns=['code','name','行业'])
    try:
        df = pd.read_csv(up, dtype={'code': str})
        df['code'] = df['code'].astype(str).str.zfill(6)
        name_col = 'name' if 'name' in df.columns else ('名称' if '名称' in df.columns else None)
        ind_col = 'industry' if 'industry' in df.columns else ('行业' if '行业' in df.columns else None)
        out = pd.DataFrame()
        out['code'] = df['code']
        if name_col:
            out['name'] = df[name_col]
        if ind_col:
            out['行业'] = df[ind_col]
        return out
    except Exception:
        return pd.DataFrame(columns=['code','name','行业'])


def load_industry_mapping() -> Dict[str,str]:
    mapping: Dict[str,str] = {}
    # 1) existing mapping file
    m = DATA / 'industry_mapping.csv'
    if m.exists():
        try:
            df = pd.read_csv(m, dtype={'code':str})
            ind_col = 'industry' if 'industry' in df.columns else ('行业' if '行业' in df.columns else None)
            if ind_col:
                for _, r in df[['code', ind_col]].dropna().iterrows():
                    mapping[str(r['code']).zfill(6)] = str(r[ind_col])
        except Exception:
            pass
    # 2) fallback to AkShare (best-effort)
    try:
        import akshare as ak
        board_df = ak.stock_board_industry_name_em()
        for _, row in board_df.iterrows():
            board_code = row.get('板块代码')
            board_name = row.get('板块名称')
            if not board_code:
                continue
            try:
                cons = ak.stock_board_industry_cons_em(board_code)
                for _, cr in cons.iterrows():
                    c = str(cr.get('代码','')).zfill(6)
                    if c:
                        mapping[c] = board_name
            except Exception:
                continue
    except Exception:
        pass
    return mapping


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Build 500-stock universe with industry')
    ap.add_argument('--target-size', type=int, default=500)
    ap.add_argument('--min-days', type=int, default=504)
    ap.add_argument('--min-amt', type=float, default=5e7)
    ap.add_argument('--download-missing', action='store_true', help='use DataCollector to fetch missing price CSVs')
    args = ap.parse_args()

    # 1) candidates from local CSVs
    local_df = load_local_candidates()
    if local_df.empty and args.download-missing:
        try:
            from data_collector import DataCollector
            # Download top names first via spot
            spot = load_spot_info()
            codes = spot['code'].tolist() if not spot.empty else []
            dc = DataCollector(data_dir=str(DATA), cache_dir=str(ROOT/'cache'))
            if codes:
                dc.download_stock_batch(codes[:args.target_size*2], max_workers=5, batch_size=20)
        except Exception:
            pass
        local_df = load_local_candidates()

    if local_df.empty:
        print('❌ 未找到本地价格数据（data/*.csv），请先下载或提供 CSV 再运行。')
        sys.exit(1)

    # 2) spot 基本信息（名称、市值）
    spot = load_spot_info()

    # 3) 合并、过滤
    uni = local_df.copy()
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    uni = uni.merge(spot, on='code', how='left')
    # 额外加入已有股票池的代码（可能无本地CSV），后续用于补齐数量
    existing = load_existing_universe()
    # ST 剔除（靠名称关键词，若无名称则保留）
    uni['is_st'] = uni['name'].astype(str).str.contains('ST', case=False, na=False)
    uni = uni[~uni['is_st']]
    # 质量门槛
    uni = uni[uni['days'] >= args.min_days]
    if 'median_amount' in uni.columns and uni['median_amount'].notna().any():
        uni = uni[uni['median_amount'].fillna(0) >= args.min_amt]
    # 排名：流通市值 → 中位成交额 → 交易天数
    uni = uni.sort_values(['流通市值_亿元','median_amount','days'], ascending=[False, False, False])
    uni = uni.drop_duplicates('code').reset_index(drop=True)

    # 若数量不足，用 existing 补齐（不再套用质量门槛，仅去重）
    if len(uni) < args.target_size and not existing.empty:
        add = existing[~existing['code'].isin(uni['code'])].copy()
        add = add.merge(spot, on='code', how='left')
        uni = pd.concat([uni, add], ignore_index=True).drop_duplicates('code').reset_index(drop=True)

    # 4) 行业匹配
    ind_map = load_industry_mapping()
    industries = []
    for _, r in uni.iterrows():
        code = str(r['code']).zfill(6)
        ind = ind_map.get(code)
        if not ind or ind in ['未分类', 'None', 'nan', '']:
            ind = _infer_industry_from_name(r.get('name'))
        industries.append(ind if ind else '未分类')
    uni['行业'] = industries

    # 5) 选取 target-size，并补充元数据
    out = uni.head(args.target_size).copy()
    out.insert(0, 'code', out.pop('code'))
    out['样本标识'] = f'扩展{args.target_size}只'
    out['更新时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 简化列
    keep_cols = ['code','name','行业','总市值_亿元','流通市值_亿元','样本标识','更新时间']
    for c in keep_cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[keep_cols]
    out.to_csv(ROOT/'stock_universe.csv', index=False, encoding='utf-8-sig')

    # 6) industry mapping 输出
    mp = out[['code','行业']].rename(columns={'行业':'industry'}).dropna()
    DATA.mkdir(exist_ok=True)
    mp.to_csv(DATA/'industry_mapping.csv', index=False, encoding='utf-8-sig')

    print(f'✅ 已生成股票池: stock_universe.csv (rows={len(out)})')
    print(f'✅ 已生成行业映射: data/industry_mapping.csv')


if __name__ == '__main__':
    main()
