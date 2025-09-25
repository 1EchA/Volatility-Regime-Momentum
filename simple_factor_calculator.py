#!/usr/bin/env python3
"""
简单因子计算器（单因子筛选准备）

目标：生成用于单因子IC/回归筛选的数据集：
- 因变量：日收益率（close-to-close）与前瞻收益（t+1, t+5, t+10）
- 简单可解释因子：
  * 动量：5d/21d/60d/250d（均跳过t-1，使用t-2相对更早窗口，防前瞻偏差）
  * 波动率：21日历史波动率（年化）
  * 量比：当日成交量 / 过去20日均量
  * 换手率：直接使用日换手率字段（若命名不统一，自动适配）
  * ILLIQ：Amihud 非流动性 |ret_{t-1}| / amount_{t-1}

输出：data/simple_factor_data.csv（包含原始与横截面标准化后的 *_std 列）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import norm


class SimpleFactorCalculator:
    def __init__(self, data_dir: str = 'data', universe_file: str | None = None,
                 standardization: str = 'zscore', output_filename: str = 'simple_factor_data.csv'):
        self.data_dir = Path(data_dir)
        self.universe_file = Path(universe_file) if universe_file else None
        self.standardization_method = standardization.lower()
        self.output_filename = output_filename
        self.industry_map = self._load_industry_map()

    def _load_codes(self) -> list[str]:
        if self.universe_file and self.universe_file.exists():
            df = pd.read_csv(self.universe_file, dtype={'code': str})
            if 'code' in df.columns:
                return df['code'].astype(str).str.zfill(6).unique().tolist()
        # 否则从 data 目录下已有CSV推断
        codes = []
        for p in self.data_dir.glob('*.csv'):
            # 排除非个股数据文件
            if p.name.startswith(('baseline_', 'factors_', 'processed_', 'market_', 'regression_', 'volatility_')):
                continue
            if p.stem.isdigit() and len(p.stem) in (6, 7):
                codes.append(p.stem[:6])
        return sorted(set(codes))

    def _load_industry_map(self) -> dict:
        industry_map = {}
        candidates = []
        if self.universe_file and self.universe_file.exists():
            candidates.append(self.universe_file)
        mapping_file = self.data_dir / 'industry_mapping.csv'
        if mapping_file.exists():
            candidates.append(mapping_file)
        for file in candidates:
            try:
                df = pd.read_csv(file, dtype={'code': str})
                if 'code' in df.columns and 'industry' in df.columns:
                    tmp = df[['code', 'industry']].dropna()
                    for _, row in tmp.iterrows():
                        industry_map[row['code'].zfill(6)] = row['industry']
            except Exception:
                continue
        return industry_map

    @staticmethod
    def _compute_daily_return(df: pd.DataFrame) -> pd.DataFrame:
        df['daily_return'] = df['close'].pct_change()
        return df

    @staticmethod
    def _compute_momentum(df: pd.DataFrame, period: int, colname: str) -> pd.DataFrame:
        # t-2 vs t-(period+1)
        df[colname] = df['close'].shift(2) / df['close'].shift(period + 1) - 1
        return df

    @staticmethod
    def _compute_volatility_21d(df: pd.DataFrame) -> pd.DataFrame:
        df['volatility_21d'] = df['daily_return'].rolling(window=21, min_periods=15).std() * np.sqrt(252)
        return df

    @staticmethod
    def _compute_vol_ratio(df: pd.DataFrame) -> pd.DataFrame:
        # 使用滞后收益构造滚动波动率，避免前瞻
        ret_lag = df['daily_return'].shift(1)
        vol21 = ret_lag.rolling(21, min_periods=15).std()
        vol63 = ret_lag.rolling(63, min_periods=30).std()
        df['vol_ratio_21_63'] = (vol21 / (vol63 + 1e-12)) - 1
        return df

    @staticmethod
    def _compute_atr14(df: pd.DataFrame) -> pd.DataFrame:
        # 需要 high/low/close
        if all(c in df.columns for c in ['high', 'low', 'close']):
            prev_close = df['close'].shift(1)
            tr1 = (df['high'] - df['low']).abs()
            tr2 = (df['high'] - prev_close).abs()
            tr3 = (df['low'] - prev_close).abs()
            tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)
            atr = pd.Series(tr).rolling(14, min_periods=10).mean().values
            df['atr14'] = atr / (df['close'].replace(0, np.nan))
        else:
            df['atr14'] = np.nan
        return df

    @staticmethod
    def _compute_gk_vol(df: pd.DataFrame) -> pd.DataFrame:
        # Garman-Klass 日波动估计 + 21日滚动均值（年化）
        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            with np.errstate(divide='ignore', invalid='ignore'):
                log_hl = np.log(df['high'] / df['low']).replace([np.inf, -np.inf], np.nan)
                log_co = np.log(df['close'] / df['open']).replace([np.inf, -np.inf], np.nan)
                gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
            gk_vol = gk_var.rolling(21, min_periods=15).mean().clip(lower=0).pow(0.5) * np.sqrt(252)
            df['gk_vol_21d'] = gk_vol
        else:
            df['gk_vol_21d'] = np.nan
        return df

    @staticmethod
    def _compute_amplitude(df: pd.DataFrame) -> pd.DataFrame:
        if all(c in df.columns for c in ['high', 'low', 'close']):
            df['amplitude_hl'] = (df['high'] - df['low']) / (df['close'].replace(0, np.nan))
        else:
            df['amplitude_hl'] = np.nan
        return df

    @staticmethod
    def _compute_breakout(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        # 使用滞后窗口的滚动高/低，避免前瞻
        roll_max = df['close'].shift(1).rolling(window, min_periods=int(window*0.6)).max()
        roll_min = df['close'].shift(1).rolling(window, min_periods=int(window*0.6)).min()
        df[f'breakout_high_{window}'] = (df['close'] > roll_max).astype(float)
        df[f'breakout_low_{window}'] = (df['close'] < roll_min).astype(float)
        return df

    @staticmethod
    def _compute_distance_to_extrema(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        max_win = df['close'].shift(1).rolling(window, min_periods=int(window*0.6)).max()
        min_win = df['close'].shift(1).rolling(window, min_periods=int(window*0.6)).min()
        df[f'dist_to_high_{window}'] = df['close'] / (max_win + 1e-12) - 1
        df[f'dist_to_low_{window}'] = df['close'] / (min_win + 1e-12) - 1
        return df

    @staticmethod
    def _compute_volume_trend(df: pd.DataFrame) -> pd.DataFrame:
        vol = df['volume'] if 'volume' in df.columns else pd.Series(np.nan, index=df.index)
        ma20 = vol.shift(1).rolling(20, min_periods=10).mean()
        ma60 = vol.shift(1).rolling(60, min_periods=30).mean()
        df['volume_trend_20_60'] = (ma20 / (ma60 + 1e-12)) - 1
        return df

    @staticmethod
    def _compute_obv(df: pd.DataFrame) -> pd.DataFrame:
        if 'volume' in df.columns:
            sign = np.sign(df['close'].diff()).fillna(0)
            obv = (sign * df['volume']).fillna(0).cumsum()
            df['obv'] = obv
        else:
            df['obv'] = np.nan
        return df

    @staticmethod
    def _compute_zero_ret_ratio(df: pd.DataFrame) -> pd.DataFrame:
        zr = (df['daily_return'].shift(1).fillna(0).abs() < 1e-10).astype(float)
        df['zero_ret_ratio_21'] = zr.rolling(21, min_periods=10).mean()
        return df

    @staticmethod
    def _compute_roll_spread(df: pd.DataFrame) -> pd.DataFrame:
        r = df['daily_return'].shift(1)
        # 计算滞后1期自协方差（滚动窗口）
        def cov_lag1(x):
            x = pd.Series(x).dropna()
            if len(x) < 10:
                return np.nan
            x0 = x[:-1]
            x1 = x[1:]
            return np.cov(x0, x1, ddof=0)[0, 1]
        cov1 = r.rolling(21, min_periods=15).apply(cov_lag1, raw=True)
        # Roll spread 近似：2 * sqrt(-cov1)，仅当cov1<0
        spread = 2 * np.sqrt(np.clip(-cov1, 0, None))
        df['roll_spread_21'] = spread
        return df

    @staticmethod
    def _compute_volume_ratio(df: pd.DataFrame) -> pd.DataFrame:
        vol_ma20 = df['volume'].rolling(window=20, min_periods=15).mean()
        df['volume_ratio'] = df['volume'] / vol_ma20
        return df

    @staticmethod
    def _compute_turnover_rate(df: pd.DataFrame) -> pd.DataFrame:
        # 兼容不同列名：turnover / turnover_rate
        if 'turnover' in df.columns:
            df['turnover_rate'] = pd.to_numeric(df['turnover'], errors='coerce')
        elif 'turnover_rate' in df.columns:
            df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce')
        else:
            df['turnover_rate'] = np.nan
        return df

    @staticmethod
    def _compute_illiq(df: pd.DataFrame) -> pd.DataFrame:
        df['illiq'] = (df['daily_return'].shift(1).abs() / (df['amount'].shift(1) + 1)) * 1e9
        return df

    @staticmethod
    def _forward_returns(df: pd.DataFrame) -> pd.DataFrame:
        df['forward_return_1d'] = df['daily_return'].shift(-1)
        df['forward_return_5d'] = df['close'].shift(-5) / df['close'] - 1
        df['forward_return_10d'] = df['close'].shift(-10) / df['close'] - 1
        return df

    def _load_stock_df(self, code: str) -> pd.DataFrame | None:
        f = self.data_dir / f'{code}.csv'
        if not f.exists():
            return None
        try:
            df = pd.read_csv(f)
            # 标准列处理
            col_map = {
                '日期': 'date', '收盘': 'close', '开盘': 'open', '最高': 'high', '最低': 'low',
                '成交量': 'volume', '成交额': 'amount', '换手率': 'turnover', '股票代码': 'code'
            }
            for k, v in col_map.items():
                if k in df.columns:
                    df = df.rename(columns={k: v})
            # 类型
            df['date'] = pd.to_datetime(df['date'])
            for c in ['open', 'close', 'high', 'low', 'volume', 'amount']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            # 股票代码列
            if 'code' in df.columns:
                df['stock_code'] = df['code'].astype(str).str.zfill(6)
            else:
                df['stock_code'] = code
            df = df.sort_values('date').reset_index(drop=True)
            return df
        except Exception:
            return None

    def compute(self) -> pd.DataFrame | None:
        codes = self._load_codes()
        if not codes:
            print('❌ 未找到股票数据/股票池')
            return None
        print(f'📋 计算因子：股票数={len(codes)}')

        all_list = []
        for i, code in enumerate(codes, 1):
            if i % 25 == 0:
                print(f'   进度 {i}/{len(codes)}')
            df = self._load_stock_df(code)
            if df is None or len(df) < 40:
                continue
            # 必要列校验
            if not set(['date', 'close']).issubset(df.columns):
                continue
            # 因子
            df = self._compute_daily_return(df)
            df = self._compute_momentum(df, 5, 'momentum_5d')
            df = self._compute_momentum(df, 21, 'momentum_21d')
            df = self._compute_momentum(df, 60, 'momentum_60d')
            df = self._compute_momentum(df, 250, 'momentum_250d')
            df = self._compute_volatility_21d(df)
            df = self._compute_vol_ratio(df)
            if 'volume' in df.columns:
                df = self._compute_volume_ratio(df)
                df = self._compute_volume_trend(df)
            df = self._compute_turnover_rate(df)
            if 'amount' in df.columns:
                df = self._compute_illiq(df)
            df = self._compute_atr14(df)
            df = self._compute_gk_vol(df)
            df = self._compute_amplitude(df)
            df = self._compute_breakout(df, 63)
            df = self._compute_distance_to_extrema(df, 252)
            df = self._compute_obv(df)
            df = self._compute_zero_ret_ratio(df)
            df = self._compute_roll_spread(df)
            df = self._forward_returns(df)

            df['industry'] = self.industry_map.get(code, '未分类')

            keep = ['date', 'stock_code', 'industry', 'close', 'daily_return', 'forward_return_1d',
                    'forward_return_5d', 'forward_return_10d',
                    'momentum_5d', 'momentum_21d', 'momentum_60d', 'momentum_250d',
                    'volatility_21d', 'vol_ratio_21_63', 'volume_ratio', 'volume_trend_20_60',
                    'turnover_rate', 'illiq', 'atr14', 'gk_vol_21d', 'amplitude_hl',
                    'breakout_high_63', 'breakout_low_63', 'dist_to_high_252', 'dist_to_low_252',
                    'obv', 'zero_ret_ratio_21', 'roll_spread_21']
            for k in keep:
                if k not in df.columns:
                    df[k] = np.nan
            all_list.append(df[keep])

        if not all_list:
            print('❌ 没有可用数据')
            return None

        data = pd.concat(all_list, ignore_index=True)
        data = data.sort_values(['date', 'stock_code']).reset_index(drop=True)

        # 横截面标准化（对因子列）
        factor_cols = ['momentum_5d', 'momentum_21d', 'momentum_60d', 'momentum_250d',
                       'volatility_21d', 'vol_ratio_21_63', 'volume_ratio', 'volume_trend_20_60',
                       'turnover_rate', 'illiq', 'atr14', 'gk_vol_21d', 'amplitude_hl',
                       'breakout_high_63', 'breakout_low_63', 'dist_to_high_252', 'dist_to_low_252',
                       'obv', 'zero_ret_ratio_21', 'roll_spread_21']
        def standardize_day(group: pd.DataFrame) -> pd.DataFrame:
            for col in factor_cols:
                if col not in group.columns:
                    continue
                vals = group[col].astype(float)
                if self.standardization_method == 'rank':
                    ranks = vals.rank(method='average', na_option='keep')
                    n = ranks.count()
                    if n >= 5:
                        scaled = (ranks - 0.5) / n
                        scaled = scaled.clip(1e-6, 1-1e-6)
                        group[f'{col}_std'] = norm.ppf(scaled)
                    else:
                        group[f'{col}_std'] = np.nan
                else:  # zscore
                    mean_val = vals.mean(skipna=True)
                    std_val = vals.std(skipna=True)
                    if std_val and std_val > 1e-12:
                        group[f'{col}_std'] = (vals - mean_val) / std_val
                    else:
                        group[f'{col}_std'] = np.nan
            return group

        data = data.groupby('date', group_keys=False).apply(standardize_day)

        # 丢弃因滞后导致的前期行（最长窗口252，保守丢前260行）
        data = data.groupby('stock_code', as_index=False, group_keys=False).apply(lambda d: d.iloc[260:])

        out = Path(self.output_filename)
        if not out.is_absolute():
            out = self.data_dir / out
        data.to_csv(out, index=False, encoding='utf-8-sig')
        print(f'💾 已保存: {out}，记录数={len(data):,}')
        return data


def main():
    import argparse
    parser = argparse.ArgumentParser(description='简单因子计算器')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--universe', type=str, default='stock_universe_selected.csv')
    parser.add_argument('--standardization', type=str, default='zscore', choices=['zscore', 'rank'])
    parser.add_argument('--output', type=str, default='simple_factor_data.csv')
    args = parser.parse_args()

    calc = SimpleFactorCalculator(data_dir=args.data_dir, universe_file=args.universe,
                                 standardization=args.standardization,
                                 output_filename=args.output)
    calc.compute()


if __name__ == '__main__':
    main()
