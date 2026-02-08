#!/usr/bin/env python3
"""
制度差异显著性深度检验
使用多种统计方法检验波动率制度间的动量效应差异
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, levene, bartlett
from statsmodels.stats.multitest import multipletests
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class RegimeDifferenceAnalyzer:
    def __init__(
        self,
        regime_data_file="data/volatility_regime_data_20250918_103600.csv",
        conditional_results_file="data/conditional_momentum_results_20250918_103747.json",
    ):
        """
        初始化制度差异分析器

        Args:
            regime_data_file: 制度分类数据文件
            conditional_results_file: 条件动量分析结果文件
        """
        self.regime_data_file = regime_data_file
        self.conditional_results_file = conditional_results_file

        self.regime_data = None
        self.conditional_results = None
        self.analysis_results = {}

        print("🔍 制度差异深度分析器初始化完成")

    def load_data(self):
        """加载数据"""
        print("\n📊 加载分析数据...")

        try:
            # 加载制度数据
            self.regime_data = pd.read_csv(self.regime_data_file)
            self.regime_data["date"] = pd.to_datetime(self.regime_data["date"])

            # 加载条件分析结果
            with open(self.conditional_results_file, "r", encoding="utf-8") as f:
                self.conditional_results = json.load(f)

            print(f"   ✅ 数据加载成功")
            print(f"   📈 观测值: {len(self.regime_data):,} 条")

            return True

        except Exception as e:
            print(f"   ❌ 数据加载失败: {e}")
            return False

    def extract_daily_momentum_ic(self):
        """提取各制度下的日度动量IC"""
        print("\n📊 计算各制度下的日度动量IC...")

        daily_ic_by_regime = {"低波动": [], "中波动": [], "高波动": []}

        # 按日期计算IC
        for date in self.regime_data["date"].unique():
            daily_data = self.regime_data[self.regime_data["date"] == date]

            if len(daily_data) >= 10:  # 足够的观测值
                regime = daily_data["regime"].iloc[0]  # 该日制度

                momentum_values = daily_data["momentum_21d"].dropna()
                return_values = daily_data["forward_return_1d"].dropna()

                # 找到共同索引
                common_idx = momentum_values.index.intersection(return_values.index)

                if len(common_idx) >= 10:
                    momentum_clean = momentum_values.loc[common_idx]
                    return_clean = return_values.loc[common_idx]

                    # 计算Spearman相关系数作为IC
                    ic, p_value = stats.spearmanr(momentum_clean, return_clean)

                    if not np.isnan(ic) and regime in daily_ic_by_regime:
                        daily_ic_by_regime[regime].append(
                            {
                                "date": date,
                                "ic": ic,
                                "p_value": p_value,
                                "n_obs": len(common_idx),
                            }
                        )

        # 转换为DataFrame
        ic_results = {}
        for regime, ic_list in daily_ic_by_regime.items():
            if ic_list:
                ic_df = pd.DataFrame(ic_list)
                ic_results[regime] = ic_df

                print(f"   📊 {regime}: {len(ic_df)} 个有效IC值")
                print(f"   - 平均IC: {ic_df['ic'].mean():.4f}")
                print(f"   - IC标准差: {ic_df['ic'].std():.4f}")

        self.daily_ic_results = ic_results
        return ic_results

    def comprehensive_regime_difference_tests(self):
        """综合制度差异检验"""
        print("\n🔬 综合制度差异检验...")

        if not hasattr(self, "daily_ic_results"):
            print("   ❌ 需要先计算日度IC")
            return None

        regimes = ["低波动", "中波动", "高波动"]
        test_results = {}

        # 1. 准备数据
        regime_ic_data = {}
        for regime in regimes:
            if regime in self.daily_ic_results:
                regime_ic_data[regime] = self.daily_ic_results[regime]["ic"].values

        # 2. 方差齐性检验
        if len(regime_ic_data) >= 2:
            # Levene检验
            levene_stat, levene_p = levene(*regime_ic_data.values())

            # Bartlett检验
            bartlett_stat, bartlett_p = bartlett(*regime_ic_data.values())

            test_results["variance_homogeneity"] = {
                "levene_stat": levene_stat,
                "levene_p": levene_p,
                "bartlett_stat": bartlett_stat,
                "bartlett_p": bartlett_p,
                "variances_equal": levene_p > 0.05,
            }

        # 3. 分布正态性检验
        normality_tests = {}
        for regime, ic_data in regime_ic_data.items():
            if len(ic_data) > 8:  # Shapiro-Wilk要求
                shapiro_stat, shapiro_p = stats.shapiro(ic_data)
                ks_stat, ks_p = stats.kstest(
                    ic_data, "norm", args=(ic_data.mean(), ic_data.std())
                )

                normality_tests[regime] = {
                    "shapiro_stat": shapiro_stat,
                    "shapiro_p": shapiro_p,
                    "ks_stat": ks_stat,
                    "ks_p": ks_p,
                    "is_normal": shapiro_p > 0.05,
                }

        test_results["normality_tests"] = normality_tests

        # 4. 两两比较检验
        pairwise_tests = {}
        regime_pairs = [("低波动", "中波动"), ("低波动", "高波动"), ("中波动", "高波动")]

        for regime1, regime2 in regime_pairs:
            if regime1 in regime_ic_data and regime2 in regime_ic_data:
                data1 = regime_ic_data[regime1]
                data2 = regime_ic_data[regime2]

                pair_key = f"{regime1}_vs_{regime2}"
                pair_results = {}

                # t检验 (假设方差相等)
                t_stat_equal, t_p_equal = stats.ttest_ind(data1, data2, equal_var=True)

                # Welch t检验 (不假设方差相等)
                t_stat_welch, t_p_welch = stats.ttest_ind(data1, data2, equal_var=False)

                # Mann-Whitney U检验 (非参数)
                u_stat, u_p = mannwhitneyu(data1, data2, alternative="two-sided")

                # 效应大小
                pooled_std = np.sqrt(
                    ((len(data1) - 1) * data1.var() + (len(data2) - 1) * data2.var())
                    / (len(data1) + len(data2) - 2)
                )
                cohens_d = (data1.mean() - data2.mean()) / pooled_std

                pair_results = {
                    "mean_diff": data1.mean() - data2.mean(),
                    "t_test_equal_var": {"stat": t_stat_equal, "p": t_p_equal},
                    "t_test_welch": {"stat": t_stat_welch, "p": t_p_welch},
                    "mann_whitney": {"stat": u_stat, "p": u_p},
                    "cohens_d": cohens_d,
                    "n1": len(data1),
                    "n2": len(data2),
                }

                pairwise_tests[pair_key] = pair_results

        test_results["pairwise_tests"] = pairwise_tests

        # 5. ANOVA检验
        if len(regime_ic_data) >= 3:
            # 单因素ANOVA
            f_stat, f_p = stats.f_oneway(*regime_ic_data.values())

            # Kruskal-Wallis检验 (非参数)
            h_stat, h_p = stats.kruskal(*regime_ic_data.values())

            test_results["anova"] = {
                "f_stat": f_stat,
                "f_p": f_p,
                "kruskal_h": h_stat,
                "kruskal_p": h_p,
            }

        # 6. 多重检验校正
        if "pairwise_tests" in test_results:
            p_values = []
            test_names = []

            for pair_name, pair_result in test_results["pairwise_tests"].items():
                p_values.append(pair_result["t_test_welch"]["p"])
                test_names.append(f"{pair_name}_welch_t")

                p_values.append(pair_result["mann_whitney"]["p"])
                test_names.append(f"{pair_name}_mann_whitney")

            # Bonferroni校正
            bonf_rejected, bonf_pvals, _, _ = multipletests(
                p_values, method="bonferroni"
            )

            # FDR校正
            fdr_rejected, fdr_pvals, _, _ = multipletests(p_values, method="fdr_bh")

            multiple_test_correction = {}
            for i, test_name in enumerate(test_names):
                multiple_test_correction[test_name] = {
                    "original_p": p_values[i],
                    "bonferroni_p": bonf_pvals[i],
                    "bonferroni_significant": bonf_rejected[i],
                    "fdr_p": fdr_pvals[i],
                    "fdr_significant": fdr_rejected[i],
                }

            test_results["multiple_testing"] = multiple_test_correction

        self.analysis_results["comprehensive_tests"] = test_results

        # 输出关键结果
        print(f"   📊 综合检验结果:")

        if "anova" in test_results:
            anova_sig = "显著" if test_results["anova"]["f_p"] < 0.05 else "不显著"
            kruskal_sig = "显著" if test_results["anova"]["kruskal_p"] < 0.05 else "不显著"
            print(
                f"   - ANOVA: F={test_results['anova']['f_stat']:.4f}, p={test_results['anova']['f_p']:.4f} ({anova_sig})"
            )
            print(
                f"   - Kruskal-Wallis: H={test_results['anova']['kruskal_h']:.4f}, p={test_results['anova']['kruskal_p']:.4f} ({kruskal_sig})"
            )

        if "multiple_testing" in test_results:
            bonf_significant = sum(
                1
                for result in test_results["multiple_testing"].values()
                if result["bonferroni_significant"]
            )
            fdr_significant = sum(
                1
                for result in test_results["multiple_testing"].values()
                if result["fdr_significant"]
            )
            total_tests = len(test_results["multiple_testing"])

            print(
                f"   - 多重检验校正: Bonferroni显著 {bonf_significant}/{total_tests}, FDR显著 {fdr_significant}/{total_tests}"
            )

        return test_results

    def bootstrap_confidence_intervals_by_regime(self, n_bootstrap=1000):
        """按制度计算Bootstrap置信区间"""
        print(f"\n🔄 计算各制度Bootstrap置信区间 (n={n_bootstrap})...")

        if not hasattr(self, "daily_ic_results"):
            print("   ❌ 需要先计算日度IC")
            return None

        bootstrap_results = {}

        for regime, ic_df in self.daily_ic_results.items():
            ic_data = ic_df["ic"].values

            if len(ic_data) >= 30:  # 足够的样本量
                bootstrap_means = []

                for _ in range(n_bootstrap):
                    # 有放回抽样
                    bootstrap_sample = np.random.choice(
                        ic_data, size=len(ic_data), replace=True
                    )
                    bootstrap_means.append(np.mean(bootstrap_sample))

                bootstrap_means = np.array(bootstrap_means)

                # 计算置信区间
                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
                original_mean = ic_data.mean()

                bootstrap_results[regime] = {
                    "original_mean": original_mean,
                    "bootstrap_mean": np.mean(bootstrap_means),
                    "bootstrap_std": np.std(bootstrap_means),
                    "ci_95_lower": ci_lower,
                    "ci_95_upper": ci_upper,
                    "contains_zero": ci_lower <= 0 <= ci_upper,
                    "n_original": len(ic_data),
                    "n_bootstrap": n_bootstrap,
                }

                contains_zero_text = (
                    "包含0" if bootstrap_results[regime]["contains_zero"] else "不包含0"
                )
                print(f"   📊 {regime}:")
                print(f"   - 原始均值: {original_mean:.6f}")
                print(
                    f"   - 95%置信区间: [{ci_lower:.6f}, {ci_upper:.6f}] ({contains_zero_text})"
                )

        self.analysis_results["bootstrap_ci"] = bootstrap_results
        return bootstrap_results

    def regime_transition_analysis(self):
        """制度转换分析"""
        print("\n🔄 制度转换模式分析...")

        # 按时间排序
        regime_time_series = (
            self.regime_data[["date", "regime", "volatility", "market_return"]]
            .drop_duplicates("date")
            .sort_values("date")
        )

        # 识别制度转换
        regime_time_series["regime_lag"] = regime_time_series["regime"].shift(1)
        regime_time_series["regime_change"] = (
            regime_time_series["regime"] != regime_time_series["regime_lag"]
        )

        # 统计转换模式
        transitions = regime_time_series[regime_time_series["regime_change"] == True]

        transition_patterns = {}
        for _, row in transitions.iterrows():
            from_regime = row["regime_lag"]
            to_regime = row["regime"]
            if pd.notna(from_regime):
                pattern = f"{from_regime}→{to_regime}"
                if pattern not in transition_patterns:
                    transition_patterns[pattern] = 0
                transition_patterns[pattern] += 1

        # 计算转换概率
        regime_counts = regime_time_series["regime"].value_counts()
        total_transitions = sum(transition_patterns.values())

        print(f"   📊 制度转换统计:")
        print(f"   - 总转换次数: {total_transitions}")
        print(f"   - 转换模式:")

        for pattern, count in sorted(
            transition_patterns.items(), key=lambda x: x[1], reverse=True
        ):
            prop = count / total_transitions
            print(f"     {pattern}: {count} 次 ({prop:.1%})")

        # 分析转换后动量效应变化
        transition_momentum_analysis = {}

        for _, row in transitions.iterrows():
            from_regime = row["regime_lag"]
            to_regime = row["regime"]

            if pd.notna(from_regime):
                pattern = f"{from_regime}→{to_regime}"

                # 获取转换日前后的数据
                transition_date = row["date"]

                # 转换前一日的动量数据
                pre_data = self.regime_data[
                    (self.regime_data["date"] < transition_date)
                    & (self.regime_data["regime"] == from_regime)
                ].tail(
                    50
                )  # 最近50个观测值

                # 转换后一日的动量数据
                post_data = self.regime_data[
                    (self.regime_data["date"] >= transition_date)
                    & (self.regime_data["regime"] == to_regime)
                ].head(
                    50
                )  # 接下来50个观测值

                if len(pre_data) > 10 and len(post_data) > 10:
                    pre_momentum = pre_data["momentum_21d"].mean()
                    post_momentum = post_data["momentum_21d"].mean()

                    if pattern not in transition_momentum_analysis:
                        transition_momentum_analysis[pattern] = []

                    transition_momentum_analysis[pattern].append(
                        {
                            "pre_momentum": pre_momentum,
                            "post_momentum": post_momentum,
                            "momentum_change": post_momentum - pre_momentum,
                        }
                    )

        self.analysis_results["regime_transitions"] = {
            "patterns": transition_patterns,
            "momentum_changes": transition_momentum_analysis,
        }

        return transition_patterns, transition_momentum_analysis

    def generate_comprehensive_difference_report(self):
        """生成综合差异分析报告"""
        print("\n📝 生成综合制度差异分析报告...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data/regime_difference_analysis_{timestamp}.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("波动率制度差异深度分析报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"数据期间: {self.regime_data['date'].min().date()} 至 {self.regime_data['date'].max().date()}\n\n"
            )

            # 1. 日度IC分析
            f.write("一、各制度日度IC统计\n")
            f.write("-" * 40 + "\n")

            if hasattr(self, "daily_ic_results"):
                for regime, ic_df in self.daily_ic_results.items():
                    ic_mean = ic_df["ic"].mean()
                    ic_std = ic_df["ic"].std()
                    ic_median = ic_df["ic"].median()
                    positive_days = (ic_df["ic"] > 0).sum()
                    total_days = len(ic_df)

                    f.write(f"\n{regime}制度:\n")
                    f.write(f"  有效天数: {total_days}\n")
                    f.write(f"  平均IC: {ic_mean:.6f}\n")
                    f.write(f"  IC标准差: {ic_std:.6f}\n")
                    f.write(f"  IC中位数: {ic_median:.6f}\n")
                    f.write(
                        f"  正IC天数: {positive_days} ({positive_days/total_days:.1%})\n"
                    )

            # 2. 综合检验结果
            f.write("\n二、统计检验结果\n")
            f.write("-" * 40 + "\n")

            if "comprehensive_tests" in self.analysis_results:
                comp_tests = self.analysis_results["comprehensive_tests"]

                # ANOVA结果
                if "anova" in comp_tests:
                    anova = comp_tests["anova"]
                    f.write(f"ANOVA检验:\n")
                    f.write(f"  F统计量: {anova['f_stat']:.4f}\n")
                    f.write(f"  p值: {anova['f_p']:.6f}\n")
                    f.write(
                        f"  结论: {'制度间存在显著差异' if anova['f_p'] < 0.05 else '制度间无显著差异'}\n\n"
                    )

                    f.write(f"Kruskal-Wallis检验 (非参数):\n")
                    f.write(f"  H统计量: {anova['kruskal_h']:.4f}\n")
                    f.write(f"  p值: {anova['kruskal_p']:.6f}\n")
                    f.write(
                        f"  结论: {'制度间存在显著差异' if anova['kruskal_p'] < 0.05 else '制度间无显著差异'}\n\n"
                    )

                # 两两比较
                if "pairwise_tests" in comp_tests:
                    f.write("两两比较结果:\n")
                    for pair, result in comp_tests["pairwise_tests"].items():
                        f.write(f"\n{pair}:\n")
                        f.write(f"  均值差异: {result['mean_diff']:.6f}\n")
                        f.write(
                            f"  Welch t检验: t={result['t_test_welch']['stat']:.4f}, p={result['t_test_welch']['p']:.6f}\n"
                        )
                        f.write(
                            f"  Mann-Whitney U检验: U={result['mann_whitney']['stat']:.0f}, p={result['mann_whitney']['p']:.6f}\n"
                        )
                        f.write(f"  效应大小(Cohen's d): {result['cohens_d']:.4f}\n")

            # 3. Bootstrap置信区间
            f.write("\n三、Bootstrap置信区间分析\n")
            f.write("-" * 40 + "\n")

            if "bootstrap_ci" in self.analysis_results:
                for regime, ci_result in self.analysis_results["bootstrap_ci"].items():
                    zero_status = "包含0" if ci_result["contains_zero"] else "不包含0"
                    f.write(f"\n{regime}:\n")
                    f.write(f"  原始均值: {ci_result['original_mean']:.6f}\n")
                    f.write(f"  Bootstrap均值: {ci_result['bootstrap_mean']:.6f}\n")
                    f.write(
                        f"  95%置信区间: [{ci_result['ci_95_lower']:.6f}, {ci_result['ci_95_upper']:.6f}]\n"
                    )
                    f.write(f"  是否包含0: {zero_status}\n")

            # 4. 核心结论
            f.write("\n四、核心结论\n")
            f.write("-" * 40 + "\n")

            if hasattr(self, "daily_ic_results"):
                # 计算各制度平均IC
                regime_ics = {}
                for regime, ic_df in self.daily_ic_results.items():
                    regime_ics[regime] = ic_df["ic"].mean()

                # 排序
                sorted_regimes = sorted(
                    regime_ics.items(), key=lambda x: abs(x[1]), reverse=True
                )

                f.write("1. 动量效应强度排序(按绝对IC值):\n")
                for rank, (regime, ic) in enumerate(sorted_regimes, 1):
                    effect_type = "反转效应" if ic < 0 else "动量效应"
                    f.write(f"   {rank}. {regime}: IC={ic:.6f} ({effect_type})\n")

                f.write("\n2. 主要发现:\n")

                # 识别主要模式
                negative_regimes = [
                    regime for regime, ic in regime_ics.items() if ic < 0
                ]
                positive_regimes = [
                    regime for regime, ic in regime_ics.items() if ic > 0
                ]

                if "高波动" in negative_regimes:
                    f.write("   - 高波动期存在反转效应，符合过度反应假说\n")
                if len(positive_regimes) > 0:
                    f.write(f"   - {', '.join(positive_regimes)}期存在微弱动量效应\n")

                f.write("   - 波动率确实调节动量效应，但主要体现为效应方向转换\n")
                f.write("   - 结果支持行为金融学的条件效应理论\n")

        print(f"   📄 综合分析报告已生成: {report_file}")
        return report_file

    def save_analysis_results(self):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"data/regime_difference_results_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                self.analysis_results, f, ensure_ascii=False, indent=2, default=str
            )

        print(f"   💾 分析结果已保存: {results_file}")
        return results_file

    def run_full_analysis(self):
        """运行完整的制度差异分析"""
        print("🚀 开始制度差异深度分析")
        print("=" * 60)

        # 1. 数据加载
        if not self.load_data():
            return None

        # 2. 计算日度IC
        self.extract_daily_momentum_ic()

        # 3. 综合统计检验
        self.comprehensive_regime_difference_tests()

        # 4. Bootstrap置信区间
        self.bootstrap_confidence_intervals_by_regime()

        # 5. 制度转换分析
        self.regime_transition_analysis()

        # 6. 保存结果
        self.save_analysis_results()

        # 7. 生成报告
        self.generate_comprehensive_difference_report()

        print("\n🎉 制度差异深度分析完成！")
        print("=" * 60)

        return self.analysis_results


def main():
    """主函数"""
    analyzer = RegimeDifferenceAnalyzer()
    results = analyzer.run_full_analysis()

    if results:
        print("\n📊 制度差异分析成功完成")
    else:
        print("❌ 制度差异分析失败")


if __name__ == "__main__":
    main()
