#!/usr/bin/env python3
"""
条件Fama-MacBeth分析
分制度检验波动率条件下的动量效应
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path


class ConditionalFamaMacBethAnalyzer:
    def __init__(self, regime_data_file: str | None = None):
        """
        初始化条件Fama-MacBeth分析器

        Args:
            regime_data_file: 制度分类数据文件路径
        """
        self.regime_data_file = regime_data_file
        self.regime_data = None
        self.results = {}

        print("🔍 条件Fama-MacBeth分析器初始化完成")

    def _find_latest_regime_file(self) -> str | None:
        data_dir = Path('data')
        candidates = sorted(data_dir.glob('volatility_regime_data_*.csv'))
        if candidates:
            # 取修改时间最新
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(candidates[0])
        return None

    def load_regime_data(self):
        """加载制度分类数据"""
        print("\n📊 加载制度分类数据...")

        try:
            # 自动探测最新制度文件
            regime_file = self.regime_data_file
            if regime_file is None:
                regime_file = self._find_latest_regime_file()
                if regime_file is None:
                    raise FileNotFoundError('未找到 volatility_regime_data_*.csv')
            self.regime_data_file = regime_file
            self.regime_data = pd.read_csv(regime_file)
            self.regime_data['date'] = pd.to_datetime(self.regime_data['date'])

            print(f"   ✅ 数据加载成功")
            print(f"   📈 观测值: {len(self.regime_data):,} 条")
            print(f"   🏢 股票数量: {self.regime_data['stock_code'].nunique()} 只")
            print(f"   📅 时间跨度: {self.regime_data['date'].min().date()} 至 {self.regime_data['date'].max().date()}")

            # 显示制度分布
            regime_counts = self.regime_data['regime'].value_counts()
            print(f"   📊 制度分布:")
            for regime, count in regime_counts.items():
                prop = count / len(self.regime_data)
                print(f"   - {regime}: {count:,} 条 ({prop:.1%})")

            return True

        except Exception as e:
            print(f"   ❌ 数据加载失败: {e}")
            return False

    def run_regime_specific_regression(self, regime_name):
        """运行特定制度下的Fama-MacBeth回归"""
        print(f"\n🔍 分析 {regime_name} 制度...")

        # 筛选特定制度的数据
        regime_subset = self.regime_data[self.regime_data['regime'] == regime_name].copy()

        if len(regime_subset) < 100:  # 数据量过少
            print(f"   ⚠️ {regime_name} 数据量不足: {len(regime_subset)} 条")
            return None

        # 第一步：横截面回归
        cross_section_results = []
        valid_dates = []

        for date in regime_subset['date'].unique():
            daily_data = regime_subset[regime_subset['date'] == date].copy()

            # 确保有足够的观测值
            if len(daily_data) >= 10:
                # 准备数据
                momentum_values = daily_data['momentum_21d'].values
                return_values = daily_data['forward_return_1d'].values

                # 检查缺失值
                valid_idx = ~(np.isnan(momentum_values) | np.isnan(return_values))

                if valid_idx.sum() >= 10:
                    X = momentum_values[valid_idx]
                    y = return_values[valid_idx]

                    # 添加常数项
                    X_with_const = sm.add_constant(X)

                    try:
                        # OLS回归
                        model = sm.OLS(y, X_with_const).fit()

                        result = {
                            'date': date,
                            'regime': regime_name,
                            'n_obs': len(y),
                            'alpha': model.params[0],
                            'beta_momentum': model.params[1],
                            'tstat_momentum': model.tvalues[1],
                            'pvalue_momentum': model.pvalues[1],
                            'r_squared': model.rsquared,
                            'adj_r_squared': model.rsquared_adj
                        }

                        cross_section_results.append(result)
                        valid_dates.append(date)

                    except Exception:
                        continue  # 跳过失败的回归

        if len(cross_section_results) == 0:
            print(f"   ❌ {regime_name} 没有成功的横截面回归")
            return None

        cross_section_df = pd.DataFrame(cross_section_results)

        print(f"   📊 成功回归日数: {len(cross_section_df)} / {regime_subset['date'].nunique()}")

        # 第二步：时间序列平均和检验
        beta_series = cross_section_df['beta_momentum'].dropna()
        alpha_series = cross_section_df['alpha'].dropna()

        if len(beta_series) > 5:
            # 动量系数统计
            beta_mean = beta_series.mean()
            beta_std = beta_series.std()
            beta_tstat = beta_mean / (beta_std / np.sqrt(len(beta_series)))
            beta_pvalue = 2 * (1 - stats.t.cdf(abs(beta_tstat), len(beta_series) - 1))

            # Alpha统计
            alpha_mean = alpha_series.mean()
            alpha_std = alpha_series.std()
            alpha_tstat = alpha_mean / (alpha_std / np.sqrt(len(alpha_series)))
            alpha_pvalue = 2 * (1 - stats.t.cdf(abs(alpha_tstat), len(alpha_series) - 1))

            # 模型统计
            avg_r_squared = cross_section_df['r_squared'].mean()
            median_r_squared = cross_section_df['r_squared'].median()

            regime_results = {
                'regime': regime_name,
                'n_regressions': len(cross_section_df),
                'avg_obs_per_day': cross_section_df['n_obs'].mean(),
                'momentum_beta': {
                    'mean': beta_mean,
                    'std': beta_std,
                    't_statistic': beta_tstat,
                    'p_value': beta_pvalue,
                    'significance': '***' if beta_pvalue < 0.01 else
                                  '**' if beta_pvalue < 0.05 else
                                  '*' if beta_pvalue < 0.1 else ''
                },
                'alpha': {
                    'mean': alpha_mean,
                    'std': alpha_std,
                    't_statistic': alpha_tstat,
                    'p_value': alpha_pvalue
                },
                'model_fit': {
                    'avg_r_squared': avg_r_squared,
                    'median_r_squared': median_r_squared
                },
                'cross_section_data': cross_section_df
            }

            print(f"   📊 {regime_name} 结果:")
            print(f"   - 动量系数: {beta_mean:.6f} {regime_results['momentum_beta']['significance']}")
            print(f"   - t统计量: {beta_tstat:.4f}")
            print(f"   - p值: {beta_pvalue:.4f}")
            print(f"   - 平均R²: {avg_r_squared:.4f}")

            return regime_results

        else:
            print(f"   ❌ {regime_name} 有效观测值不足")
            return None

    def compare_regime_effects(self):
        """比较不同制度下的动量效应"""
        print("\n🔍 运行分制度Fama-MacBeth分析...")

        regime_results = {}
        # 自动获取制度标签并按波动率均值排序
        regimes = list(self.regime_data['regime'].dropna().unique())
        regime_order = (
            self.regime_data.groupby('regime')['volatility'].mean().sort_values().index.tolist()
            if 'volatility' in self.regime_data.columns else regimes
        )

        # 分别分析各制度
        for regime in regime_order:
            result = self.run_regime_specific_regression(regime)
            if result is not None:
                regime_results[regime] = result

        self.results['regime_analysis'] = regime_results

        # 比较分析
        if len(regime_results) >= 2:
            print(f"\n📊 制度比较分析:")
            print("   " + "="*70)
            print(f"   {'制度':<8} {'动量系数':<12} {'t统计量':<10} {'p值':<10} {'显著性':<8} {'R²':<8}")
            print("   " + "-"*70)

            for regime, results in regime_results.items():
                momentum_info = results['momentum_beta']
                model_info = results['model_fit']

                print(f"   {regime:<8} {momentum_info['mean']:<12.6f} "
                      f"{momentum_info['t_statistic']:<10.4f} "
                      f"{momentum_info['p_value']:<10.4f} "
                      f"{momentum_info['significance']:<8} "
                      f"{model_info['avg_r_squared']:<8.4f}")

        return regime_results

    def test_regime_difference_significance(self):
        """检验制度间动量效应差异显著性"""
        print("\n🔬 检验制度间动量效应差异...")

        if 'regime_analysis' not in self.results:
            print("   ❌ 需要先运行制度分析")
            return None

        regime_results = self.results['regime_analysis']
        regimes = list(regime_results.keys())

        if len(regimes) < 2:
            print("   ❌ 至少需要两个制度的结果")
            return None

        # 提取各制度的动量系数序列
        regime_betas = {}
        for regime in regimes:
            if 'cross_section_data' in regime_results[regime]:
                beta_series = regime_results[regime]['cross_section_data']['beta_momentum']
                regime_betas[regime] = beta_series.dropna()

        # 两两比较
        comparison_results = {}

        for i, regime1 in enumerate(regimes):
            for j, regime2 in enumerate(regimes):
                if i < j and regime1 in regime_betas and regime2 in regime_betas:
                    data1 = regime_betas[regime1]
                    data2 = regime_betas[regime2]

                    if len(data1) > 5 and len(data2) > 5:
                        # 两样本t检验
                        t_stat, p_value = stats.ttest_ind(data1, data2)

                        # 计算效应大小 (Cohen's d)
                        pooled_std = np.sqrt(((len(data1) - 1) * data1.var() +
                                            (len(data2) - 1) * data2.var()) /
                                           (len(data1) + len(data2) - 2))
                        cohens_d = (data1.mean() - data2.mean()) / pooled_std

                        comparison_key = f"{regime1}_vs_{regime2}"
                        comparison_results[comparison_key] = {
                            'regime1': regime1,
                            'regime2': regime2,
                            'mean1': data1.mean(),
                            'mean2': data2.mean(),
                            'difference': data1.mean() - data2.mean(),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'cohens_d': cohens_d,
                            'significant': p_value < 0.05
                        }

        # 显示比较结果
        print(f"   📊 制度间动量效应比较:")
        print("   " + "="*80)
        print(f"   {'比较':<20} {'差异':<12} {'t统计量':<10} {'p值':<10} {'效应大小':<10} {'显著性':<8}")
        print("   " + "-"*80)

        for comp_name, comp_result in comparison_results.items():
            significance = "是" if comp_result['significant'] else "否"
            print(f"   {comp_name:<20} {comp_result['difference']:<12.6f} "
                  f"{comp_result['t_statistic']:<10.4f} "
                  f"{comp_result['p_value']:<10.4f} "
                  f"{comp_result['cohens_d']:<10.4f} "
                  f"{significance:<8}")

        self.results['regime_comparisons'] = comparison_results
        return comparison_results

    def calculate_conditional_momentum_strength(self):
        """计算条件动量效应强度"""
        print("\n📈 计算条件动量效应强度...")

        if 'regime_analysis' not in self.results:
            print("   ❌ 需要先运行制度分析")
            return None

        regime_results = self.results['regime_analysis']
        strength_analysis = {}

        for regime, results in regime_results.items():
            momentum_info = results['momentum_beta']

            # 计算效应强度指标
            strength_metrics = {
                'momentum_coefficient': momentum_info['mean'],
                'absolute_t_stat': abs(momentum_info['t_statistic']),
                'significance_level': 1 if momentum_info['p_value'] < 0.01 else
                                    2 if momentum_info['p_value'] < 0.05 else
                                    3 if momentum_info['p_value'] < 0.1 else 4,
                'economic_significance': abs(momentum_info['mean']) > 0.001,  # 经济显著性阈值
                'statistical_power': 1 - momentum_info['p_value'] if momentum_info['p_value'] < 0.5 else 0.5
            }

            strength_analysis[regime] = strength_metrics

        # 排序和评级
        sorted_regimes = sorted(strength_analysis.items(),
                              key=lambda x: x[1]['absolute_t_stat'], reverse=True)

        print(f"   📊 各制度动量效应强度排名:")
        print("   " + "="*60)
        print(f"   {'排名':<4} {'制度':<8} {'系数':<12} {'|t|':<8} {'显著性等级':<10}")
        print("   " + "-"*60)

        for rank, (regime, metrics) in enumerate(sorted_regimes, 1):
            sig_level_text = {1: "***", 2: "**", 3: "*", 4: ""}[metrics['significance_level']]
            print(f"   {rank:<4} {regime:<8} {metrics['momentum_coefficient']:<12.6f} "
                  f"{metrics['absolute_t_stat']:<8.4f} {sig_level_text:<10}")

        self.results['strength_analysis'] = strength_analysis
        return strength_analysis

    def generate_conditional_report(self):
        """生成条件动量效应分析报告"""
        print("\n📝 生成条件动量效应分析报告...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data/conditional_momentum_report_{timestamp}.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("波动率条件动量效应分析报告\n")
            f.write("="*60 + "\n\n")

            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据期间: {self.regime_data['date'].min().date()} 至 {self.regime_data['date'].max().date()}\n")
            f.write(f"观测值总数: {len(self.regime_data):,} 条\n\n")

            # 1. 分制度结果
            f.write("一、分制度动量效应分析\n")
            f.write("-"*40 + "\n")

            if 'regime_analysis' in self.results:
                for regime, results in self.results['regime_analysis'].items():
                    momentum_info = results['momentum_beta']
                    model_info = results['model_fit']

                    f.write(f"\n{regime}制度:\n")
                    f.write(f"  回归次数: {results['n_regressions']}\n")
                    f.write(f"  平均观测值/日: {results['avg_obs_per_day']:.1f}\n")
                    f.write(f"  动量系数: {momentum_info['mean']:.6f} {momentum_info['significance']}\n")
                    f.write(f"  t统计量: {momentum_info['t_statistic']:.4f}\n")
                    f.write(f"  p值: {momentum_info['p_value']:.4f}\n")
                    f.write(f"  平均R²: {model_info['avg_r_squared']:.4f}\n")
                    f.write(f"  中位数R²: {model_info['median_r_squared']:.4f}\n")

            # 2. 制度比较
            f.write("\n二、制度间比较分析\n")
            f.write("-"*40 + "\n")

            if 'regime_comparisons' in self.results:
                for comp_name, comp_result in self.results['regime_comparisons'].items():
                    significance = "显著" if comp_result['significant'] else "不显著"
                    f.write(f"\n{comp_name}:\n")
                    f.write(f"  动量系数差异: {comp_result['difference']:.6f}\n")
                    f.write(f"  t统计量: {comp_result['t_statistic']:.4f}\n")
                    f.write(f"  p值: {comp_result['p_value']:.4f}\n")
                    f.write(f"  效应大小(Cohen's d): {comp_result['cohens_d']:.4f}\n")
                    f.write(f"  差异显著性: {significance}\n")

            # 3. 效应强度分析
            f.write("\n三、条件动量效应强度分析\n")
            f.write("-"*40 + "\n")

            if 'strength_analysis' in self.results:
                sorted_regimes = sorted(self.results['strength_analysis'].items(),
                                      key=lambda x: x[1]['absolute_t_stat'], reverse=True)

                f.write("按效应强度排序:\n")
                for rank, (regime, metrics) in enumerate(sorted_regimes, 1):
                    f.write(f"{rank}. {regime}: 动量系数={metrics['momentum_coefficient']:.6f}, ")
                    f.write(f"|t|={metrics['absolute_t_stat']:.4f}\n")

            # 4. 核心发现
            f.write("\n四、核心发现与结论\n")
            f.write("-"*40 + "\n")

            if 'regime_analysis' in self.results:
                regime_results = self.results['regime_analysis']

                # 找出最强和最弱的制度
                if len(regime_results) >= 2:
                    regime_strengths = {}
                    for regime, results in regime_results.items():
                        regime_strengths[regime] = abs(results['momentum_beta']['t_statistic'])

                    strongest_regime = max(regime_strengths, key=regime_strengths.get)
                    weakest_regime = min(regime_strengths, key=regime_strengths.get)

                    f.write(f"1. 动量效应最强制度: {strongest_regime}\n")
                    f.write(f"2. 动量效应最弱制度: {weakest_regime}\n")

                    # 检查是否符合理论预期
                    if strongest_regime == '高波动':
                        f.write("3. 结果符合理论预期: 高波动期动量效应增强\n")
                    else:
                        f.write("3. 结果与理论预期不完全一致，需进一步分析\n")

            f.write(f"\n注: *** p<0.01, ** p<0.05, * p<0.1\n")

        print(f"   📄 报告已生成: {report_file}")
        return report_file

    def save_results(self):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"data/conditional_momentum_results_{timestamp}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)

        print(f"   💾 结果已保存: {results_file}")
        return results_file

    def run_full_analysis(self):
        """运行完整的条件动量分析"""
        print("🚀 开始条件动量效应分析")
        print("="*60)

        # 1. 加载数据
        if not self.load_regime_data():
            return None

        # 2. 分制度回归分析
        self.compare_regime_effects()

        # 3. 制度差异显著性检验
        self.test_regime_difference_significance()

        # 4. 效应强度分析
        self.calculate_conditional_momentum_strength()

        # 5. 保存结果
        self.save_results()

        # 6. 生成报告
        self.generate_conditional_report()

        print("\n🎉 条件动量效应分析完成！")
        print("="*60)

        return self.results

def main():
    """主函数"""
    analyzer = ConditionalFamaMacBethAnalyzer()
    results = analyzer.run_full_analysis()

    if results:
        print("\n📊 条件动量分析成功完成")
    else:
        print("❌ 条件动量分析失败")

if __name__ == "__main__":
    main()
