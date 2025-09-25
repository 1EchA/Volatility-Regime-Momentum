#!/usr/bin/env python3
"""
自动股票池筛选（基于市值与流动性）

功能：
- 从 AkShare 实时A股行情拉取全市场快照
- 过滤：剔除ST/退市、PE异常、流动性不足
- 排序与分行业约束抽样，构建目标规模（如100/150/200）股票池
- 输出 CSV（不覆盖现有 stock_universe.csv，默认写入 selected_universe.csv）

说明：
- 依赖 akshare.stock_zh_a_spot_em()
- 流动性指标采用当前快照字段（成交额、换手率），并提供可选的“近250日均成交额”校验占位（如需，后续可为候选集批量拉取日线做进一步筛选）
"""

import akshare as ak
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class UniverseSelector:
    def __init__(self, output_path: str = 'stock_universe_selected.csv'):
        self.output_path = Path(output_path)

    def fetch_spot(self) -> pd.DataFrame:
        df = ak.stock_zh_a_spot_em()
        # 统一列名（尽可能鲁棒）
        rename_map = {
            '代码': 'code', '名称': 'name', '总市值': 'total_market_cap', '流通市值': 'float_market_cap',
            '市盈率-动态': 'pe_ttm', '市净率': 'pb', '换手率': 'turnover_rate', '成交额': 'amount',
            '量比': 'volume_ratio', '行业': 'industry'
        }
        for k, v in rename_map.items():
            if k in df.columns:
                df = df.rename(columns={k: v})
        # 类型转换
        for col in ['total_market_cap', 'float_market_cap', 'pe_ttm', 'pb', 'turnover_rate', 'amount', 'volume_ratio']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _basic_filters(self, df: pd.DataFrame,
                        min_float_mcap: float = 3e10,  # 300亿（单位：元）
                        min_amount: float = 5e8,       # 当日成交额≥5亿
                        pe_low: float = 0.0,
                        pe_high: float = 100.0) -> pd.DataFrame:
        df = df.copy()
        # 剔除 ST / *ST
        if 'name' in df.columns:
            df = df[~df['name'].astype(str).str.contains('ST', case=False, na=False)]
        # 流通市值过滤
        if 'float_market_cap' in df.columns:
            df = df[df['float_market_cap'] >= min_float_mcap]
        # 成交额过滤
        if 'amount' in df.columns:
            df = df[df['amount'] >= min_amount]
        # PE区间
        if 'pe_ttm' in df.columns:
            df = df[(df['pe_ttm'] >= pe_low) & (df['pe_ttm'] <= pe_high)]
        return df

    def _balanced_sample(self, df: pd.DataFrame, target_n: int = 150, max_per_industry: int = 25) -> pd.DataFrame:
        df = df.copy()
        if 'industry' not in df.columns:
            df['industry'] = '未分类'
        # 按流通市值降序
        df = df.sort_values(['float_market_cap', 'amount'], ascending=[False, False])
        # 若行业信息不可用（仅1类），直接取TOP N
        if df['industry'].nunique() <= 1:
            return df.head(target_n).copy()
        selected = []
        industry_counts = {}
        for _, row in df.iterrows():
            ind = row['industry'] if pd.notna(row['industry']) else '未分类'
            cnt = industry_counts.get(ind, 0)
            if cnt >= max_per_industry:
                continue
            selected.append(row)
            industry_counts[ind] = cnt + 1
            if len(selected) >= target_n:
                break
        return pd.DataFrame(selected)

    def build_universe(self, target_n: int = 150,
                        min_float_mcap: float = 3e10,
                        min_amount: float = 5e8,
                        pe_low: float = 0.0, pe_high: float = 100.0,
                        max_per_industry: int = 25) -> pd.DataFrame:
        print('📥 拉取A股快照...')
        spot = self.fetch_spot()
        print(f'   总数: {len(spot)}')

        print('🔍 基础过滤（市值/流动性/PE）...')
        filt = self._basic_filters(spot, min_float_mcap=min_float_mcap, min_amount=min_amount,
                                   pe_low=pe_low, pe_high=pe_high)
        print(f'   过滤后: {len(filt)}')

        print('🔀 分行业均衡抽样...')
        selected = self._balanced_sample(filt, target_n=target_n, max_per_industry=max_per_industry)
        print(f'   入选: {len(selected)}')

        # 精简输出列
        keep = ['code', 'name', 'float_market_cap', 'total_market_cap', 'pe_ttm', 'pb',
                'turnover_rate', 'amount', 'volume_ratio', 'industry']
        for col in keep:
            if col not in selected.columns:
                selected[col] = np.nan

        selected = selected[keep]
        selected['selected'] = 1
        selected['selection_timestamp'] = datetime.now().isoformat(timespec='seconds')

        # 保存
        selected.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f'💾 已保存股票池: {self.output_path}')
        return selected


def main():
    import argparse
    parser = argparse.ArgumentParser(description='自动股票池筛选')
    parser.add_argument('--target-n', type=int, default=150)
    parser.add_argument('--output', type=str, default='stock_universe_selected.csv')
    parser.add_argument('--min-float-mcap', type=float, default=3e10)
    parser.add_argument('--min-amount', type=float, default=5e8)
    parser.add_argument('--max-per-industry', type=int, default=25)
    parser.add_argument('--pe-low', type=float, default=0.0)
    parser.add_argument('--pe-high', type=float, default=100.0)
    args = parser.parse_args()

    selector = UniverseSelector(output_path=args.output)
    selector.build_universe(
        target_n=args.target_n,
        min_float_mcap=args.min_float_mcap,
        min_amount=args.min_amount,
        pe_low=args.pe_low,
        pe_high=args.pe_high,
        max_per_industry=args.max_per_industry,
    )


if __name__ == '__main__':
    main()
