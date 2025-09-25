#!/usr/bin/env python3
"""
Fama-MacBeth多因子回归分析
基于5个基线因子构建Fama-MacBeth两步法回归模型
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FamaMacBethAnalyzer:
    def __init__(self, data_file='data/baseline_factor_data.csv'):
        """
        初始化Fama-MacBeth分析器

        Args:
            data_file: 基线因子数据文件路径
        """
        self.data_file = data_file
        self.data = None
        self.factor_columns = [
            'momentum_21d_std',
            'volatility_21d_std',
            'volume_ratio_std',
            'pe_percentile_std',
            'relative_return_std'
        ]
        self.results = {}
        self.cross_section_results = []

        print("🔍 Fama-MacBeth分析器初始化完成")

    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("\n📊 加载并准备数据...")

        try:
            self.data = pd.read_csv(self.data_file)
            self.data['date'] = pd.to_datetime(self.data['date'])

            # 按日期和股票代码排序
            self.data = self.data.sort_values(['date', 'stock_code'])

            # 计算前瞻1日收益率
            forward_returns = []
            for stock_code in self.data['stock_code'].unique():
                stock_data = self.data[self.data['stock_code'] == stock_code].copy()
                stock_data = stock_data.sort_values('date')
                stock_data['forward_return_1d'] = stock_data['daily_return'].shift(-1)
                forward_returns.append(stock_data)

            self.data = pd.concat(forward_returns, ignore_index=True)

            # 移除无前瞻收益率和因子缺失的观测值
            self.data = self.data.dropna(subset=['forward_return_1d'] + self.factor_columns)

            print(f"   ✅ 数据准备完成")
            print(f"   📈 有效观测值: {len(self.data):,} 条")
            print(f"   📅 时间跨度: {self.data['date'].min().date()} 至 {self.data['date'].max().date()}")
            print(f"   🏢 股票数量: {self.data['stock_code'].nunique()} 只")
            print(f"   📊 交易日数量: {self.data['date'].nunique()} 个")

            return True

        except Exception as e:
            print(f"   ❌ 数据准备失败: {e}")
            return False

    def run_cross_sectional_regressions(self):
        """第一步：运行横截面回归"""
        print("\n🔍 第一步：运行每日横截面回归...")

        valid_dates = []
        cross_section_results = []

        for date in self.data['date'].unique():
            daily_data = self.data[self.data['date'] == date].copy()

            # 确保有足够的观测值
            if len(daily_data) >= 10:  # 至少10个观测值
                # 准备因子矩阵
                factor_matrix = daily_data[self.factor_columns].values
                returns = daily_data['forward_return_1d'].values

                # 检查是否有缺失值
                valid_idx = ~(np.isnan(factor_matrix).any(axis=1) | np.isnan(returns))

                if valid_idx.sum() >= 10:  # 有效观测值足够
                    X = factor_matrix[valid_idx]
                    y = returns[valid_idx]

                    # 添加常数项
                    X_with_const = sm.add_constant(X)

                    try:
                        # OLS回归
                        model = sm.OLS(y, X_with_const).fit()

                        # 保存结果
                        result = {
                            'date': date,
                            'n_obs': len(y),
                            'alpha': model.params[0],  # 常数项
                            'r_squared': model.rsquared,
                            'adj_r_squared': model.rsquared_adj
                        }

                        # 保存各因子系数
                        for i, factor in enumerate(self.factor_columns):
                            result[f'beta_{factor}'] = model.params[i + 1]
                            result[f'tstat_{factor}'] = model.tvalues[i + 1]
                            result[f'pvalue_{factor}'] = model.pvalues[i + 1]

                        cross_section_results.append(result)
                        valid_dates.append(date)

                    except Exception as e:
                        print(f"   ⚠️ {date.date()} 回归失败: {e}")

        self.cross_section_results = pd.DataFrame(cross_section_results)

        print(f"   ✅ 横截面回归完成")
        print(f"   📊 成功回归日数: {len(self.cross_section_results)} / {self.data['date'].nunique()}")
        print(f"   📈 成功率: {len(self.cross_section_results) / self.data['date'].nunique():.1%}")

        if len(self.cross_section_results) == 0:
            print("   ❌ 没有成功的横截面回归")
            return False

        return True

    def calculate_time_series_averages(self):
        """第二步：计算时间序列平均值和检验"""
        print("\n📊 第二步：计算时间序列平均值和统计检验...")

        if len(self.cross_section_results) == 0:
            print("   ❌ 无横截面回归结果")
            return False

        # 计算平均系数和检验
        time_series_results = {}

        # Alpha（截距项）
        alpha_series = self.cross_section_results['alpha'].dropna()
        if len(alpha_series) > 0:
            alpha_mean = alpha_series.mean()
            alpha_std = alpha_series.std()
            alpha_tstat = alpha_mean / (alpha_std / np.sqrt(len(alpha_series)))
            alpha_pvalue = 2 * (1 - stats.t.cdf(abs(alpha_tstat), len(alpha_series) - 1))

            time_series_results['alpha'] = {
                'mean': alpha_mean,
                'std': alpha_std,
                't_statistic': alpha_tstat,
                'p_value': alpha_pvalue,
                'n_obs': len(alpha_series)
            }

        # 各因子Beta
        for factor in self.factor_columns:
            beta_col = f'beta_{factor}'
            if beta_col in self.cross_section_results.columns:
                beta_series = self.cross_section_results[beta_col].dropna()

                if len(beta_series) > 0:
                    beta_mean = beta_series.mean()
                    beta_std = beta_series.std()
                    beta_tstat = beta_mean / (beta_std / np.sqrt(len(beta_series)))
                    beta_pvalue = 2 * (1 - stats.t.cdf(abs(beta_tstat), len(beta_series) - 1))

                    time_series_results[factor] = {
                        'mean': beta_mean,
                        'std': beta_std,
                        't_statistic': beta_tstat,
                        'p_value': beta_pvalue,
                        'n_obs': len(beta_series),
                        'significance': '***' if beta_pvalue < 0.01 else
                                      '**' if beta_pvalue < 0.05 else
                                      '*' if beta_pvalue < 0.1 else ''
                    }

                    print(f"   📊 {factor}: β={beta_mean:.4f}, t={beta_tstat:.2f}, p={beta_pvalue:.4f} {time_series_results[factor]['significance']}")

        # 计算模型整体统计量
        r_squared_series = self.cross_section_results['r_squared'].dropna()
        adj_r_squared_series = self.cross_section_results['adj_r_squared'].dropna()

        model_stats = {
            'avg_r_squared': r_squared_series.mean(),
            'median_r_squared': r_squared_series.median(),
            'avg_adj_r_squared': adj_r_squared_series.mean(),
            'median_adj_r_squared': adj_r_squared_series.median(),
            'avg_n_obs': self.cross_section_results['n_obs'].mean()
        }

        print(f"\n   📈 模型整体表现:")
        print(f"   - 平均R²: {model_stats['avg_r_squared']:.4f}")
        print(f"   - 中位数R²: {model_stats['median_r_squared']:.4f}")
        print(f"   - 平均调整R²: {model_stats['avg_adj_r_squared']:.4f}")
        print(f"   - 平均观测值数/日: {model_stats['avg_n_obs']:.1f}")

        self.results = {
            'time_series_results': time_series_results,
            'model_stats': model_stats,
            'cross_section_summary': {
                'total_regressions': len(self.cross_section_results),
                'date_range': [
                    str(self.cross_section_results['date'].min().date()),
                    str(self.cross_section_results['date'].max().date())
                ]
            }
        }

        return True

    def run_diagnostic_tests(self):
        """运行模型诊断检验"""
        print("\n🔬 运行模型诊断检验...")

        diagnostics = {}

        # 1. R²分布分析
        r_squared_series = self.cross_section_results['r_squared']
        diagnostics['r_squared_distribution'] = {
            'mean': r_squared_series.mean(),
            'std': r_squared_series.std(),
            'min': r_squared_series.min(),
            'max': r_squared_series.max(),
            'q25': r_squared_series.quantile(0.25),
            'q75': r_squared_series.quantile(0.75),
            'low_r2_days': (r_squared_series < 0.05).sum(),  # R²<5%的天数
            'high_r2_days': (r_squared_series > 0.3).sum()   # R²>30%的天数
        }

        # 2. 因子系数稳定性检验
        factor_stability = {}
        for factor in self.factor_columns:
            beta_col = f'beta_{factor}'
            if beta_col in self.cross_section_results.columns:
                beta_series = self.cross_section_results[beta_col].dropna()

                factor_stability[factor] = {
                    'std_dev': beta_series.std(),
                    'coeff_var': abs(beta_series.std() / beta_series.mean()) if beta_series.mean() != 0 else np.inf,
                    'positive_days': (beta_series > 0).sum(),
                    'negative_days': (beta_series < 0).sum(),
                    'sign_consistency': max((beta_series > 0).sum(), (beta_series < 0).sum()) / len(beta_series)
                }

        diagnostics['factor_stability'] = factor_stability

        # 3. 样本量充足性检验
        n_obs_series = self.cross_section_results['n_obs']
        diagnostics['sample_adequacy'] = {
            'avg_obs_per_day': n_obs_series.mean(),
            'min_obs_per_day': n_obs_series.min(),
            'days_insufficient_sample': (n_obs_series < 15).sum(),  # 样本量<15的天数
            'obs_to_var_ratio': n_obs_series.mean() / (len(self.factor_columns) + 1)  # 观测值/变量比
        }

        # 4. 残差分析 (基于最近100个回归)
        recent_data = self.cross_section_results.tail(100)
        alpha_series = recent_data['alpha'].dropna()

        if len(alpha_series) > 10:
            # Ljung-Box检验序列相关性
            from statsmodels.stats.diagnostic import acorr_ljungbox
            try:
                lb_stat, lb_pvalue = acorr_ljungbox(alpha_series, lags=10, return_df=False)
                diagnostics['residual_analysis'] = {
                    'ljung_box_stat': float(lb_stat.iloc[-1]) if hasattr(lb_stat, 'iloc') else float(lb_stat),
                    'ljung_box_pvalue': float(lb_pvalue.iloc[-1]) if hasattr(lb_pvalue, 'iloc') else float(lb_pvalue),
                    'serial_correlation': 'Yes' if (lb_pvalue.iloc[-1] if hasattr(lb_pvalue, 'iloc') else lb_pvalue) < 0.05 else 'No'
                }
            except:
                diagnostics['residual_analysis'] = {'note': 'Ljung-Box test failed'}

        self.results['diagnostics'] = diagnostics

        print("   ✅ 诊断检验完成")
        print(f"   📊 R²分布: 均值={diagnostics['r_squared_distribution']['mean']:.4f}, 低R²天数={diagnostics['r_squared_distribution']['low_r2_days']}")
        print(f"   🔍 样本充足性: 平均观测值/日={diagnostics['sample_adequacy']['avg_obs_per_day']:.1f}")

        return True

    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n📝 生成Fama-MacBeth分析报告...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data/fama_macbeth_report_{timestamp}.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Fama-MacBeth多因子回归分析报告\n")
            f.write("="*60 + "\n\n")

            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据期间: {self.results['cross_section_summary']['date_range'][0]} 至 {self.results['cross_section_summary']['date_range'][1]}\n")
            f.write(f"回归次数: {self.results['cross_section_summary']['total_regressions']}\n")
            f.write(f"股票数量: {self.data['stock_code'].nunique()} 只\n\n")

            f.write("一、Fama-MacBeth两步法结果\n")
            f.write("-"*40 + "\n")

            # Alpha结果
            if 'alpha' in self.results['time_series_results']:
                alpha_result = self.results['time_series_results']['alpha']
                f.write(f"Alpha (截距项): {alpha_result['mean']:.6f}\n")
                f.write(f"  t统计量: {alpha_result['t_statistic']:.4f}\n")
                f.write(f"  p值: {alpha_result['p_value']:.4f}\n")
                f.write(f"  标准误: {alpha_result['std'] / np.sqrt(alpha_result['n_obs']):.6f}\n\n")

            # 因子结果
            f.write("因子风险溢价 (Factor Risk Premiums):\n")
            for factor, result in self.results['time_series_results'].items():
                if factor != 'alpha':
                    sig = result.get('significance', '')
                    f.write(f"\n{factor}:\n")
                    f.write(f"  系数: {result['mean']:.6f} {sig}\n")
                    f.write(f"  t统计量: {result['t_statistic']:.4f}\n")
                    f.write(f"  p值: {result['p_value']:.4f}\n")
                    f.write(f"  标准误: {result['std'] / np.sqrt(result['n_obs']):.6f}\n")

            f.write("\n二、模型整体表现\n")
            f.write("-"*40 + "\n")
            f.write(f"平均R²: {self.results['model_stats']['avg_r_squared']:.4f}\n")
            f.write(f"中位数R²: {self.results['model_stats']['median_r_squared']:.4f}\n")
            f.write(f"平均调整R²: {self.results['model_stats']['avg_adj_r_squared']:.4f}\n")
            f.write(f"平均观测值数/日: {self.results['model_stats']['avg_n_obs']:.1f}\n")

            f.write("\n三、诊断检验结果\n")
            f.write("-"*40 + "\n")

            # R²分布
            r2_dist = self.results['diagnostics']['r_squared_distribution']
            f.write(f"R²分布统计:\n")
            f.write(f"  均值: {r2_dist['mean']:.4f}\n")
            f.write(f"  标准差: {r2_dist['std']:.4f}\n")
            f.write(f"  范围: [{r2_dist['min']:.4f}, {r2_dist['max']:.4f}]\n")
            f.write(f"  低解释力天数 (R²<5%): {r2_dist['low_r2_days']}\n")
            f.write(f"  高解释力天数 (R²>30%): {r2_dist['high_r2_days']}\n\n")

            # 样本充足性
            sample_adeq = self.results['diagnostics']['sample_adequacy']
            f.write(f"样本充足性:\n")
            f.write(f"  观测值/变量比: {sample_adeq['obs_to_var_ratio']:.1f}\n")
            f.write(f"  样本不足天数: {sample_adeq['days_insufficient_sample']}\n\n")

            # 因子稳定性
            f.write("因子稳定性:\n")
            for factor, stability in self.results['diagnostics']['factor_stability'].items():
                f.write(f"  {factor}: 符号一致性={stability['sign_consistency']:.2%}\n")

            f.write(f"\n注: *** p<0.01, ** p<0.05, * p<0.1\n")

        print(f"   📄 报告已生成: {report_file}")
        return report_file

    def save_results(self):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果
        results_file = f"data/fama_macbeth_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)

        # 保存横截面回归结果
        cross_section_file = f"data/cross_section_results_{timestamp}.csv"
        self.cross_section_results.to_csv(cross_section_file, index=False, encoding='utf-8-sig')

        print(f"\n💾 结果已保存:")
        print(f"   📄 综合结果: {results_file}")
        print(f"   📊 横截面结果: {cross_section_file}")

        return results_file, cross_section_file

    def run_full_analysis(self):
        """运行完整的Fama-MacBeth分析"""
        print("🚀 开始Fama-MacBeth多因子回归分析")
        print("="*60)

        # 1. 数据准备
        if not self.load_and_prepare_data():
            return None

        # 2. 横截面回归
        if not self.run_cross_sectional_regressions():
            return None

        # 3. 时间序列平均和检验
        if not self.calculate_time_series_averages():
            return None

        # 4. 诊断检验
        self.run_diagnostic_tests()

        # 5. 保存结果
        self.save_results()

        # 6. 生成报告
        self.generate_comprehensive_report()

        print("\n🎉 Fama-MacBeth分析完成！")
        print("="*60)

        return self.results

def main():
    """主函数"""
    analyzer = FamaMacBethAnalyzer()
    results = analyzer.run_full_analysis()

    if results:
        print("\n📊 分析成功完成")

        # 显示关键结果
        if 'time_series_results' in results:
            print("\n🎯 关键因子显著性:")
            for factor, result in results['time_series_results'].items():
                if factor != 'alpha':
                    sig = result.get('significance', '')
                    print(f"   {factor}: β={result['mean']:.4f} {sig}")
    else:
        print("❌ 分析失败")

if __name__ == "__main__":
    main()