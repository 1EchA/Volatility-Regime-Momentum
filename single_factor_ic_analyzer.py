#!/usr/bin/env python3
"""
单因子IC分析和显著性检验
计算每个因子的信息系数(Information Coefficient)并进行统计显著性检验
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


class SingleFactorICAnalyzer:
    def __init__(
        self,
        data_file="data/baseline_factor_data.csv",
        factor_columns: list[str] | None = None,
    ):
        """
        初始化IC分析器

        Args:
            data_file: 基线因子数据文件路径
        """
        self.data_file = data_file
        self.data = None
        # 若未显式提供因子列，将在加载后自动识别 *_std 列
        self.factor_columns = factor_columns
        self.results = {}

        print("🔍 单因子IC分析器初始化完成")

    def load_data(self):
        """加载基线因子数据"""
        print("\n📊 加载基线因子数据...")

        try:
            self.data = pd.read_csv(self.data_file)
            self.data["date"] = pd.to_datetime(self.data["date"])

            # 按日期和股票代码排序
            self.data = self.data.sort_values(["date", "stock_code"])

            print(f"   ✅ 数据加载成功")
            print(f"   📈 数据规模: {len(self.data):,} 条观测值")
            print(
                f"   📅 时间跨度: {self.data['date'].min().date()} 至 {self.data['date'].max().date()}"
            )
            print(f"   🏢 股票数量: {self.data['stock_code'].nunique()} 只")

            # 自动识别因子列
            if self.factor_columns is None:
                exclude = {
                    "date",
                    "stock_code",
                    "close",
                    "daily_return",
                    "forward_return_1d",
                    "forward_return_5d",
                    "forward_return_10d",
                }
                std_cols = [
                    c
                    for c in self.data.columns
                    if c.endswith("_std") and c not in exclude
                ]
                # 回退到基线5因子
                if std_cols:
                    self.factor_columns = std_cols
                else:
                    self.factor_columns = [
                        "momentum_21d_std",
                        "volatility_21d_std",
                        "volume_ratio_std",
                        "pe_percentile_std",
                        "relative_return_std",
                    ]

            return True

        except Exception as e:
            print(f"   ❌ 数据加载失败: {e}")
            return False

    def calculate_forward_returns(self):
        """计算前瞻收益率"""
        print("\n📈 计算前瞻收益率...")

        # 按股票分组计算1日、5日、10日前瞻收益率
        forward_returns = []

        for stock_code in self.data["stock_code"].unique():
            stock_data = self.data[self.data["stock_code"] == stock_code].copy()
            stock_data = stock_data.sort_values("date")

            # 1日前瞻收益率
            stock_data["forward_return_1d"] = stock_data["daily_return"].shift(-1)

            # 5日前瞻收益率
            stock_data["forward_return_5d"] = (
                stock_data["close"].shift(-5) / stock_data["close"] - 1
            )

            # 10日前瞻收益率
            stock_data["forward_return_10d"] = (
                stock_data["close"].shift(-10) / stock_data["close"] - 1
            )

            forward_returns.append(stock_data)

        self.data = pd.concat(forward_returns, ignore_index=True)

        # 移除无前瞻收益率的观测值
        self.data = self.data.dropna(subset=["forward_return_1d"])

        print(f"   ✅ 前瞻收益率计算完成")
        print(f"   📊 有效观测值: {len(self.data):,} 条")

        return True

    def calculate_ic_by_date(self, factor_col, return_col):
        """
        按日期计算IC值

        Args:
            factor_col: 因子列名
            return_col: 收益率列名

        Returns:
            Series: 每日IC值
        """
        daily_ics = []
        dates = []

        for date in self.data["date"].unique():
            daily_data = self.data[self.data["date"] == date]

            # 确保有足够的观测值
            if len(daily_data) >= 5:
                factor_values = daily_data[factor_col].dropna()
                return_values = daily_data[return_col].dropna()

                # 找到因子和收益率都有值的观测
                common_idx = factor_values.index.intersection(return_values.index)

                if len(common_idx) >= 5:
                    factor_clean = factor_values.loc[common_idx]
                    return_clean = return_values.loc[common_idx]

                    # 计算Spearman相关系数作为IC
                    ic, _ = stats.spearmanr(factor_clean, return_clean)

                    if not np.isnan(ic):
                        daily_ics.append(ic)
                        dates.append(date)

        return pd.Series(daily_ics, index=dates)

    def analyze_single_factor(self, factor_col):
        """
        分析单个因子的IC表现

        Args:
            factor_col: 因子列名

        Returns:
            dict: 因子分析结果
        """
        print(f"   🔍 分析因子: {factor_col}")

        results = {}

        # 对不同期限的前瞻收益率计算IC
        return_periods = ["1d", "5d", "10d"]

        for period in return_periods:
            return_col = f"forward_return_{period}"

            if return_col in self.data.columns:
                # 计算每日IC
                daily_ics = self.calculate_ic_by_date(factor_col, return_col)

                if len(daily_ics) > 0:
                    # 基本统计量
                    ic_mean = daily_ics.mean()
                    ic_std = daily_ics.std()
                    ic_median = daily_ics.median()

                    # t检验
                    t_stat, p_value = stats.ttest_1samp(daily_ics, 0)

                    # IR (Information Ratio)
                    ir = ic_mean / ic_std if ic_std != 0 else 0

                    # 胜率 (IC>0的比例)
                    win_rate = (daily_ics > 0).mean()

                    # 绝对IC均值
                    abs_ic_mean = daily_ics.abs().mean()

                    results[period] = {
                        "ic_mean": ic_mean,
                        "ic_std": ic_std,
                        "ic_median": ic_median,
                        "ir": ir,
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "win_rate": win_rate,
                        "abs_ic_mean": abs_ic_mean,
                        "ic_count": len(daily_ics),
                        "daily_ics": daily_ics.tolist(),
                    }

                    print(
                        f"     📊 {period} IC均值: {ic_mean:.4f}, t统计量: {t_stat:.4f}, p值: {p_value:.4f}"
                    )

        return results

    def run_ic_analysis(self):
        """运行完整的IC分析"""
        print("\n🚀 开始单因子IC分析")
        print("=" * 60)

        # 加载数据
        if not self.load_data():
            return None

        # 计算前瞻收益率
        if not self.calculate_forward_returns():
            return None

        print(f"\n🔍 开始分析 {len(self.factor_columns)} 个因子...")

        # 分析每个因子
        for factor_col in self.factor_columns:
            factor_results = self.analyze_single_factor(factor_col)
            self.results[factor_col] = factor_results

        return self.results

    def generate_summary_report(self):
        """生成汇总报告"""
        print("\n📋 生成IC分析汇总报告...")

        # 创建汇总表
        summary_data = []

        for factor_name, factor_results in self.results.items():
            for period, period_results in factor_results.items():
                summary_data.append(
                    {
                        "factor": factor_name,
                        "period": period,
                        "ic_mean": period_results["ic_mean"],
                        "ic_std": period_results["ic_std"],
                        "ir": period_results["ir"],
                        "t_stat": period_results["t_statistic"],
                        "p_value": period_results["p_value"],
                        "win_rate": period_results["win_rate"],
                        "abs_ic_mean": period_results["abs_ic_mean"],
                        "significance": "***"
                        if period_results["p_value"] < 0.01
                        else "**"
                        if period_results["p_value"] < 0.05
                        else "*"
                        if period_results["p_value"] < 0.1
                        else "",
                    }
                )

        summary_df = pd.DataFrame(summary_data)

        print("\n📊 IC分析结果汇总:")
        print(summary_df.round(4))

        # 识别最佳因子
        print("\n🏆 因子排名 (按1日IC_IR排序):")
        ranking_1d = summary_df[summary_df["period"] == "1d"].sort_values(
            "ir", ascending=False
        )
        for idx, row in ranking_1d.iterrows():
            print(
                f"   {row['factor']}: IR={row['ir']:.4f}, IC均值={row['ic_mean']:.4f}, 胜率={row['win_rate']:.2%} {row['significance']}"
            )

        # 显著性统计
        significant_factors = summary_df[summary_df["p_value"] < 0.05]
        print(f"\n✅ 显著因子数量 (p<0.05): {len(significant_factors)} / {len(summary_df)}")

        return summary_df

    def save_results(self):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果
        results_file = f"data/ic_analysis_results_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)

        # 保存汇总报告
        summary_df = self.generate_summary_report()
        summary_file = f"data/ic_analysis_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

        print(f"\n💾 结果已保存:")
        print(f"   📄 详细结果: {results_file}")
        print(f"   📊 汇总报告: {summary_file}")

        return results_file, summary_file

    def generate_academic_report(self):
        """生成学术格式报告"""
        print("\n📝 生成学术格式IC分析报告...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data/ic_analysis_report_{timestamp}.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("单因子IC分析报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"数据期间: {self.data['date'].min().date()} 至 {self.data['date'].max().date()}\n"
            )
            f.write(f"股票数量: {self.data['stock_code'].nunique()} 只\n")
            f.write(f"观测值数量: {len(self.data):,} 条\n\n")

            f.write("一、IC分析方法说明\n")
            f.write("-" * 30 + "\n")
            f.write("1. IC计算方法: Spearman相关系数\n")
            f.write("2. 前瞻收益率: 1日、5日、10日\n")
            f.write("3. 显著性水平: α=0.05, 0.01\n")
            f.write("4. 信息比率: IR = IC均值 / IC标准差\n\n")

            f.write("二、各因子IC表现\n")
            f.write("-" * 30 + "\n")

            for factor_name, factor_results in self.results.items():
                f.write(f"\n{factor_name}:\n")

                for period, results in factor_results.items():
                    sig_level = (
                        "***"
                        if results["p_value"] < 0.01
                        else "**"
                        if results["p_value"] < 0.05
                        else "*"
                        if results["p_value"] < 0.1
                        else ""
                    )

                    f.write(f"  {period}期: IC={results['ic_mean']:.4f} {sig_level}, ")
                    f.write(
                        f"IR={results['ir']:.4f}, t统计量={results['t_statistic']:.4f}, "
                    )
                    f.write(
                        f"p值={results['p_value']:.4f}, 胜率={results['win_rate']:.2%}\n"
                    )

            f.write("\n三、统计显著性总结\n")
            f.write("-" * 30 + "\n")

            # 统计显著因子
            total_tests = sum(
                len(factor_results) for factor_results in self.results.values()
            )
            significant_01 = 0
            significant_05 = 0

            for factor_results in self.results.values():
                for results in factor_results.values():
                    if results["p_value"] < 0.01:
                        significant_01 += 1
                    elif results["p_value"] < 0.05:
                        significant_05 += 1

            f.write(f"总测试数量: {total_tests}\n")
            f.write(f"1%水平显著: {significant_01} ({significant_01/total_tests:.1%})\n")
            f.write(f"5%水平显著: {significant_05} ({significant_05/total_tests:.1%})\n")
            f.write(
                f"总显著数量: {significant_01 + significant_05} ({(significant_01 + significant_05)/total_tests:.1%})\n"
            )

            f.write("\n四、因子有效性结论\n")
            f.write("-" * 30 + "\n")

            # 找出最佳因子
            best_factors = {}
            for factor_name, factor_results in self.results.items():
                if "1d" in factor_results:
                    ir_1d = factor_results["1d"]["ir"]
                    p_val_1d = factor_results["1d"]["p_value"]
                    best_factors[factor_name] = {"ir": ir_1d, "p_value": p_val_1d}

            sorted_factors = sorted(
                best_factors.items(), key=lambda x: x[1]["ir"], reverse=True
            )

            f.write("按1日期IR排名:\n")
            for i, (factor_name, metrics) in enumerate(sorted_factors, 1):
                sig_level = (
                    "***"
                    if metrics["p_value"] < 0.01
                    else "**"
                    if metrics["p_value"] < 0.05
                    else "*"
                    if metrics["p_value"] < 0.1
                    else ""
                )
                f.write(f"{i}. {factor_name}: IR={metrics['ir']:.4f} {sig_level}\n")

            f.write(f"\n注: *** p<0.01, ** p<0.05, * p<0.1\n")

        print(f"   📄 学术报告已生成: {report_file}")
        return report_file


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="单因子IC分析器")
    parser.add_argument(
        "--data-file", type=str, default="data/baseline_factor_data.csv"
    )
    parser.add_argument(
        "--factors", type=str, nargs="*", default=None, help="指定因子列（如不指定，将自动识别 *_std 列）"
    )
    args = parser.parse_args()

    analyzer = SingleFactorICAnalyzer(
        data_file=args.data_file, factor_columns=args.factors
    )

    # 运行IC分析
    results = analyzer.run_ic_analysis()

    if results:
        # 保存结果
        analyzer.save_results()

        # 生成学术报告
        analyzer.generate_academic_report()

        print("\n🎉 单因子IC分析完成！")
        print("=" * 60)
    else:
        print("❌ IC分析失败")


if __name__ == "__main__":
    main()
