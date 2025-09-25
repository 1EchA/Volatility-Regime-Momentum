#!/usr/bin/env python3
"""
波动率制度分类与条件动量分析
构建基于市场波动率的三制度分类体系，研究条件动量效应
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VolatilityRegimeAnalyzer:
    def __init__(self, data_file='data/baseline_factor_data.csv',
                 vol_window: int = 21,
                 vol_metric: str = 'composite',  # 'ts', 'xsec', 'composite'
                 regime_scheme: str = 'q50_90'   # 'q33_67' | 'q50_90'
                 ):
        """
        初始化波动率制度分析器

        Args:
            data_file: 基线因子数据文件路径
        """
        self.data_file = data_file
        self.data = None
        self.market_data = None
        self.regime_data = None
        self.vol_window = vol_window
        self.vol_metric = vol_metric
        self.regime_scheme = regime_scheme

        print("🔍 波动率制度分析器初始化完成")

    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("\n📊 加载基线数据...")

        try:
            self.data = pd.read_csv(self.data_file)
            self.data['date'] = pd.to_datetime(self.data['date'])

            print(f"   ✅ 数据加载成功")
            print(f"   📈 观测值: {len(self.data):,} 条")
            print(f"   🏢 股票数量: {self.data['stock_code'].nunique()} 只")
            print(f"   📅 时间跨度: {self.data['date'].min().date()} 至 {self.data['date'].max().date()}")

            return True

        except Exception as e:
            print(f"   ❌ 数据加载失败: {e}")
            return False

    def construct_market_index(self):
        """构建市场指数（等权重）"""
        print("\n📈 构建市场指数...")

        # 计算每日等权重市场收益率与横截面波动率
        grouped = self.data.groupby('date')
        market_return = grouped['daily_return'].mean()
        xsec_vol = grouped['daily_return'].std()
        counts = grouped.size()

        self.market_data = pd.DataFrame({
            'date': market_return.index,
            'market_return': market_return.values,
            'xsec_vol': xsec_vol.values,
            'n_stocks': counts.values
        }).sort_values('date')

        print(f"   ✅ 市场指数构建完成")
        print(f"   📅 交易日数量: {len(self.market_data)}")
        print(f"   📊 平均股票数/日: {self.market_data['n_stocks'].mean():.1f}")

        return True

    def calculate_market_volatility(self, window=None):
        """计算市场波动率（时间序列/横截面/复合）"""
        if window is None:
            window = self.vol_window
        print(f"\n📊 计算市场波动率 (窗口={window}天, 度量={self.vol_metric})...")

        # 时间序列滚动波动率（等权收益）
        self.market_data['vol_ts'] = (
            self.market_data['market_return']
            .rolling(window=window, min_periods=window//2)
            .std() * np.sqrt(252)
        )

        # 横截面波动率平滑
        self.market_data['vol_xsec'] = (
            self.market_data['xsec_vol']
            .rolling(window=max(5, window//2), min_periods=3)
            .mean()
        )

        # 标准化后合成
        if self.vol_metric == 'ts':
            self.market_data['volatility'] = self.market_data['vol_ts']
        elif self.vol_metric == 'xsec':
            self.market_data['volatility'] = self.market_data['vol_xsec']
        else:
            # composite: 0.5*Z(ts) + 0.5*Z(xsec)
            z_ts = (self.market_data['vol_ts'] - self.market_data['vol_ts'].mean()) / (self.market_data['vol_ts'].std() + 1e-9)
            z_xs = (self.market_data['vol_xsec'] - self.market_data['vol_xsec'].mean()) / (self.market_data['vol_xsec'].std() + 1e-9)
            self.market_data['volatility'] = 0.5 * z_ts + 0.5 * z_xs

        # 清洗
        self.market_data = self.market_data.dropna(subset=['volatility'])

        vol_stats = self.market_data['volatility'].describe()
        print(f"   📊 波动率统计:")
        print(f"   - 均值: {vol_stats['mean']:.4f}")
        print(f"   - 标准差: {vol_stats['std']:.4f}")
        print(f"   - 最小值: {vol_stats['min']:.4f}")
        print(f"   - 最大值: {vol_stats['max']:.4f}")

        return True

    def define_volatility_regimes(self):
        """定义波动率制度：支持 q33_67 或 q50_90，将名称改为：正常/高波动/极高波动"""
        print(f"\n🎯 定义波动率制度 (方案: {self.regime_scheme})...")

        if self.regime_scheme == 'q33_67':
            low_threshold = self.market_data['volatility'].quantile(0.33)
            high_threshold = self.market_data['volatility'].quantile(0.67)
        else:
            # q50_90: 正常(<=P50), 高波动(P50-P90], 极高波动(>P90)
            low_threshold = self.market_data['volatility'].quantile(0.50)
            high_threshold = self.market_data['volatility'].quantile(0.90)

        conditions = [
            self.market_data['volatility'] <= low_threshold,
            (self.market_data['volatility'] > low_threshold) & (self.market_data['volatility'] <= high_threshold),
            self.market_data['volatility'] > high_threshold
        ]
        choices = ['正常', '高波动', '极高波动']
        self.market_data['regime'] = np.select(conditions, choices, default='未分类')

        # 统计各制度分布
        regime_counts = self.market_data['regime'].value_counts()
        regime_props = self.market_data['regime'].value_counts(normalize=True)

        print(f"   📊 制度分布:")
        for regime in ['正常', '高波动', '极高波动']:
            count = regime_counts.get(regime, 0)
            prop = regime_props.get(regime, 0)
            print(f"   - {regime}: {count} 天 ({prop:.1%})")

        print(f"   🎯 制度划分阈值:")
        print(f"   - 低波动上限: {low_threshold:.4f}")
        print(f"   - 高波动下限: {high_threshold:.4f}")

        return True

    def merge_regime_data(self):
        """将制度信息合并到主数据"""
        print("\n🔗 合并制度信息到主数据...")

        # 选择需要的列
        regime_info = self.market_data[['date', 'regime', 'volatility', 'market_return']].copy()

        # 合并到主数据
        self.regime_data = pd.merge(
            self.data,
            regime_info,
            on='date',
            how='inner'
        )

        print(f"   ✅ 数据合并完成")
        print(f"   📈 合并后观测值: {len(self.regime_data):,} 条")
        print(f"   📊 各制度观测值分布:")

        regime_obs = self.regime_data['regime'].value_counts()
        for regime, count in regime_obs.items():
            prop = count / len(self.regime_data)
            print(f"   - {regime}: {count:,} 条 ({prop:.1%})")

        return True

    def calculate_conditional_momentum(self):
        """计算条件动量因子"""
        print("\n🚀 计算条件动量因子...")

        # 为每只股票计算前瞻收益率
        conditional_data = []

        for stock_code in self.regime_data['stock_code'].unique():
            stock_data = self.regime_data[self.regime_data['stock_code'] == stock_code].copy()
            stock_data = stock_data.sort_values('date')

            # 计算1日前瞻收益率
            stock_data['forward_return_1d'] = stock_data['daily_return'].shift(-1)

            # 重新计算动量因子（确保没有前瞻偏差）
            stock_data['momentum_21d'] = (
                stock_data['close'].shift(2) / stock_data['close'].shift(22) - 1
            )

            conditional_data.append(stock_data)

        self.regime_data = pd.concat(conditional_data, ignore_index=True)

        # 移除缺失值
        original_len = len(self.regime_data)
        self.regime_data = self.regime_data.dropna(subset=['forward_return_1d', 'momentum_21d'])

        print(f"   ✅ 条件动量因子计算完成")
        print(f"   📊 有效观测值: {len(self.regime_data):,} 条 (原{original_len:,}条)")

        return True

    def analyze_regime_characteristics(self):
        """分析各制度特征"""
        print("\n📋 分析各制度特征...")

        regime_stats = {}

        for regime in ['正常', '高波动', '极高波动']:
            regime_subset = self.regime_data[self.regime_data['regime'] == regime]

            if len(regime_subset) > 0:
                stats_dict = {
                    'n_obs': len(regime_subset),
                    'avg_market_vol': regime_subset['volatility'].mean(),
                    'avg_market_return': regime_subset['market_return'].mean(),
                    'avg_momentum': regime_subset['momentum_21d'].mean(),
                    'std_momentum': regime_subset['momentum_21d'].std(),
                    'avg_forward_return': regime_subset['forward_return_1d'].mean(),
                    'n_trading_days': regime_subset['date'].nunique(),
                    'n_stocks_avg': regime_subset.groupby('date')['stock_code'].count().mean()
                }

                regime_stats[regime] = stats_dict

        # 展示结果
        print("\n   📊 各制度特征对比:")
        print("   " + "="*80)
        print(f"   {'指标':<20} {'低波动':<15} {'中波动':<15} {'高波动':<15}")
        print("   " + "-"*80)

        metrics = [
            ('观测值数量', 'n_obs', '{:,}'),
            ('平均市场波动率', 'avg_market_vol', '{:.4f}'),
            ('平均市场收益率', 'avg_market_return', '{:.4f}'),
            ('平均动量因子', 'avg_momentum', '{:.4f}'),
            ('动量因子标准差', 'std_momentum', '{:.4f}'),
            ('平均前瞻收益率', 'avg_forward_return', '{:.4f}'),
            ('交易日数量', 'n_trading_days', '{:.0f}'),
            ('平均股票数/日', 'n_stocks_avg', '{:.1f}')
        ]

        for metric_name, metric_key, fmt in metrics:
            row = f"   {metric_name:<20}"
            for regime in ['正常', '高波动', '极高波动']:
                if regime in regime_stats:
                    value = regime_stats[regime][metric_key]
                    row += f" {fmt.format(value):<15}"
                else:
                    row += f" {'N/A':<15}"
            print(row)

        self.regime_stats = regime_stats
        return regime_stats

    def test_regime_differences(self):
        """检验制度间差异显著性"""
        print("\n🔬 检验制度间差异显著性...")

        regimes = ['低波动', '中波动', '高波动']
        test_results = {}

        # 检验动量因子的制度差异
        momentum_data = []
        forward_return_data = []

        for regime in regimes:
            regime_subset = self.regime_data[self.regime_data['regime'] == regime]
            momentum_data.append(regime_subset['momentum_21d'].dropna())
            forward_return_data.append(regime_subset['forward_return_1d'].dropna())

        # ANOVA检验 - 动量因子
        if all(len(data) > 10 for data in momentum_data):
            f_stat_mom, p_val_mom = stats.f_oneway(*momentum_data)
            test_results['momentum_anova'] = {
                'f_statistic': f_stat_mom,
                'p_value': p_val_mom,
                'significant': p_val_mom < 0.05
            }

        # ANOVA检验 - 前瞻收益率
        if all(len(data) > 10 for data in forward_return_data):
            f_stat_ret, p_val_ret = stats.f_oneway(*forward_return_data)
            test_results['return_anova'] = {
                'f_statistic': f_stat_ret,
                'p_value': p_val_ret,
                'significant': p_val_ret < 0.05
            }

        print(f"   📊 ANOVA检验结果:")
        if 'momentum_anova' in test_results:
            mom_result = test_results['momentum_anova']
            sig_text = "显著" if mom_result['significant'] else "不显著"
            print(f"   - 动量因子制度差异: F={mom_result['f_statistic']:.4f}, p={mom_result['p_value']:.4f} ({sig_text})")

        if 'return_anova' in test_results:
            ret_result = test_results['return_anova']
            sig_text = "显著" if ret_result['significant'] else "不显著"
            print(f"   - 前瞻收益率制度差异: F={ret_result['f_statistic']:.4f}, p={ret_result['p_value']:.4f} ({sig_text})")

        self.test_results = test_results
        return test_results

    def save_regime_data(self):
        """保存制度分类数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存完整制度数据
        regime_file = f"data/volatility_regime_data_{timestamp}.csv"
        self.regime_data.to_csv(regime_file, index=False, encoding='utf-8-sig')

        # 保存市场数据
        market_file = f"data/market_volatility_data_{timestamp}.csv"
        self.market_data.to_csv(market_file, index=False, encoding='utf-8-sig')

        print(f"\n💾 制度数据已保存:")
        print(f"   📄 制度分类数据: {regime_file}")
        print(f"   📈 市场波动率数据: {market_file}")

        return regime_file, market_file

    def run_full_analysis(self):
        """运行完整的制度分析"""
        print("🚀 开始波动率制度分类分析")
        print("="*60)

        # 1. 数据准备
        if not self.load_and_prepare_data():
            return None

        # 2. 构建市场指数
        if not self.construct_market_index():
            return None

        # 3. 计算市场波动率
        if not self.calculate_market_volatility():
            return None

        # 4. 定义波动率制度
        if not self.define_volatility_regimes():
            return None

        # 5. 合并制度信息
        if not self.merge_regime_data():
            return None

        # 6. 计算条件动量因子
        if not self.calculate_conditional_momentum():
            return None

        # 7. 分析制度特征
        self.analyze_regime_characteristics()

        # 8. 检验制度差异
        self.test_regime_differences()

        # 9. 保存数据
        self.save_regime_data()

        print("\n🎉 波动率制度分析完成！")
        print("="*60)

        return self.regime_data

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='波动率制度分析')
    parser.add_argument('--data-file', type=str, default='data/baseline_factor_data.csv')
    parser.add_argument('--vol-window', type=int, default=21)
    parser.add_argument('--vol-metric', type=str, default='composite', choices=['ts', 'xsec', 'composite'])
    parser.add_argument('--scheme', type=str, default='q50_90', choices=['q33_67', 'q50_90'])
    args = parser.parse_args()

    analyzer = VolatilityRegimeAnalyzer(
        data_file=args.data_file,
        vol_window=args.vol_window,
        vol_metric=args.vol_metric,
        regime_scheme=args.scheme,
    )
    regime_data = analyzer.run_full_analysis()

    if regime_data is not None:
        print("\n📊 制度分类成功完成")
        print(f"   🎯 为条件动量分析准备了 {len(regime_data):,} 条观测值")
    else:
        print("❌ 制度分析失败")

if __name__ == "__main__":
    main()
