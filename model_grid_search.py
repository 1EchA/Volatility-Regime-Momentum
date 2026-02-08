#!/usr/bin/env python3
"""
岭回归网格搜索（交叉截面预测）

组合维度：
- use_regime ∈ {True, False}
- train_window_days ∈ {504, 756}
- alpha ∈ {0.5, 1.0, 2.0, 5.0}
- 可选：对因子做方向对齐（依据 OOS 起点之前的平均日度IC，负IC因子整体乘以-1）

输出：
- data/model_grid_results_YYYYMMDD_HHMMSS.csv
- 控制台打印Top结果
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from predictive_model import (
    load_data,
    detect_factor_columns,
    walk_forward_predict,
    evaluate_predictions,
)
from scipy import stats


def align_factor_signs(
    df: pd.DataFrame, factor_cols: list[str], start_oos: str
) -> list[str]:
    pre = df[df["date"] < pd.to_datetime(start_oos)]
    aligned = []
    for f in factor_cols:
        ics = []
        for dt, g in pre.groupby("date"):
            x = g[f]
            y = (
                g["forward_return_1d"]
                if "forward_return_1d" in g.columns
                else g["daily_return"].shift(-1)
            )
            mask = x.notna() & y.notna()
            if mask.sum() >= 10:
                ic, _ = stats.spearmanr(x[mask], y[mask])
                if not np.isnan(ic):
                    ics.append(ic)
        if len(ics) >= 60 and np.nanmean(ics) < 0:
            # 负向，整体翻转方向
            df[f] = -df[f]
            aligned.append(f)
    return aligned


def main():
    import argparse

    parser = argparse.ArgumentParser(description="岭回归网格搜索（交叉截面预测）")
    parser.add_argument("--data-file", type=str, default="data/simple_factor_data.csv")
    parser.add_argument("--factors-file", type=str, default=None)
    parser.add_argument("--start-oos", type=str, default="2022-01-01")
    parser.add_argument("--align-signs", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data_file, use_regime=True)
    if args.factors_file and Path(args.factors_file).exists():
        with open(args.factors_file, "r", encoding="utf-8") as f:
            factor_cols = [line.strip() for line in f if line.strip()]
    else:
        factor_cols = detect_factor_columns(df)

    if args.align_signs:
        flipped = align_factor_signs(df, factor_cols, args.start_oos)
        print(f"方向对齐：翻转 {len(flipped)} 个因子 → {flipped}")

    configs = []
    use_regimes = [True, False]
    windows = [504, 756]
    alphas = [0.5, 1.0, 2.0, 5.0]

    for ur in use_regimes:
        for w in windows:
            for a in alphas:
                print(f">> 训练: regime={ur}, window={w}, alpha={a}")
                pred = walk_forward_predict(
                    df,
                    factor_cols,
                    start_oos=args.start_oos,
                    train_window_days=w,
                    alpha=a,
                    use_regime=ur and ("regime" in df.columns),
                )
                if pred.empty:
                    continue
                metrics = evaluate_predictions(pred)
                cfg = {
                    "use_regime": ur,
                    "train_window": w,
                    "alpha": a,
                }
                cfg.update(metrics)
                configs.append(cfg)

    res = pd.DataFrame(configs)
    if not res.empty:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("data") / f"model_grid_results_{ts}.csv"
        res.to_csv(out, index=False, encoding="utf-8-sig")
        print("\nTop by IC_IR:")
        print(res.sort_values("ic_ir", ascending=False).head(5))
        print("\nTop by LS_IR:")
        print(res.sort_values("ls_ir", ascending=False).head(5))
        print(f"💾 保存: {out}")
    else:
        print("❌ 没有有效的组合结果")


if __name__ == "__main__":
    main()
