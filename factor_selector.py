#!/usr/bin/env python3
"""
因子筛选器

功能:
- 从因子数据（推荐: data/simple_factor_data.csv）中识别 *_std 因子
- 计算每日Spearman IC → ic_mean, t, p, 胜率
- 首轮筛选: |ic_mean| >= 阈值 且 p < 0.05 （默认阈值: 0.01）
- 二轮去相关: 依据全样本因子值相关系数 |rho| < 0.8，保留 |ic| 较高者
- 输出: 选择的因子列表 和 汇总报告
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def detect_factors(df: pd.DataFrame) -> list[str]:
    exclude = {
        "date",
        "stock_code",
        "close",
        "daily_return",
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_10d",
        "regime",
        "volatility",
        "market_return",
    }
    return [c for c in df.columns if c.endswith("_std") and c not in exclude]


def daily_ic(
    df: pd.DataFrame, factor: str, return_col: str = "forward_return_1d"
) -> pd.Series:
    ics = []
    dates = []
    for dt, g in df.groupby("date"):
        x = g[factor]
        y = g[return_col]
        mask = x.notna() & y.notna()
        if mask.sum() >= 10:
            ic, _ = stats.spearmanr(x[mask], y[mask])
            if not np.isnan(ic):
                ics.append(ic)
                dates.append(dt)
    return pd.Series(ics, index=dates)


def select_factors(
    df: pd.DataFrame,
    candidate_factors: list[str],
    ic_thresh: float = 0.01,
    corr_thresh: float = 0.8,
) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for f in candidate_factors:
        ics = daily_ic(df, f)
        if len(ics) < 60:
            continue
        ic_mean = ics.mean()
        ic_std = ics.std()
        t_stat, p_val = stats.ttest_1samp(ics, 0)
        win_rate = (ics > 0).mean()
        rows.append(
            {
                "factor": f,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "t": t_stat,
                "p": p_val,
                "win_rate": win_rate,
                "n_days": len(ics),
            }
        )
    summary = pd.DataFrame(rows).sort_values("ic_mean", key=np.abs, ascending=False)
    # 首轮筛选
    pool = summary[
        (summary["p"] < 0.05) & (summary["ic_mean"].abs() >= ic_thresh)
    ].copy()
    selected = []
    if not pool.empty:
        # 准备相关矩阵（全样本因子值）
        sub = df[["date", "stock_code"] + pool["factor"].tolist()].dropna()
        corr = sub[pool["factor"].tolist()].corr().abs()
        for f in pool["factor"]:
            if not selected:
                selected.append(f)
                continue
            # 与已选因子相关性检测
            high_corr = False
            for s in selected:
                if corr.loc[f, s] >= corr_thresh:
                    high_corr = True
                    break
            if not high_corr:
                selected.append(f)
    return selected, summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="因子筛选器")
    parser.add_argument("--data-file", type=str, default="data/simple_factor_data.csv")
    parser.add_argument("--ic-thresh", type=float, default=0.01)
    parser.add_argument("--corr-thresh", type=float, default=0.8)
    args = parser.parse_args()

    df = pd.read_csv(args.data_file)
    df["date"] = pd.to_datetime(df["date"])
    factors = detect_factors(df)
    print(f"候选因子数量: {len(factors)}")
    selected, summary = select_factors(
        df, factors, ic_thresh=args.ic_thresh, corr_thresh=args.corr_thresh
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data")
    out_summary = out_dir / f"factor_selection_summary_{ts}.csv"
    out_list = out_dir / f"selected_factors_{ts}.txt"
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    with open(out_list, "w", encoding="utf-8") as f:
        for s in selected:
            f.write(s + "\n")
    print(f"已选因子数: {len(selected)} → {selected}")
    print(f"💾 汇总: {out_summary}\n💾 列表: {out_list}")


if __name__ == "__main__":
    main()
