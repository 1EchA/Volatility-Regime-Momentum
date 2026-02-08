#!/usr/bin/env python3
"""
分制度岭回归网格搜索

组合维度：
- train_window_days ∈ {504, 756, 1008}
- alpha ∈ {0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0}
- 使用最新制度因子映射 JSON（由 regime_factor_selector.py 生成）

输出：
- data/regime_model_grid_results_YYYYMMDD_HHMMSS.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from predictive_model import (
    load_data,
    walk_forward_predict_regime_specific,
    evaluate_predictions,
)


def find_latest_mapping() -> str | None:
    """Locate latest regime→factors mapping JSON.
    Supports both historical name patterns and the pipeline-produced files.
    """
    data = Path("data")
    patterns = [
        "selected_factors_by_regime_*.json",
        "pipeline_selected_factors_*.json",
    ]
    cands: list[Path] = []
    for pat in patterns:
        cands.extend(data.glob(pat))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="分制度岭回归网格搜索")
    parser.add_argument("--data-file", type=str, default="data/simple_factor_data.csv")
    parser.add_argument("--mapping", type=str, default=None)
    parser.add_argument("--start-oos", type=str, default="2022-01-01")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--bottom-n", type=int, default=30)
    parser.add_argument("--cost-bps", type=float, default=0.0005)
    parser.add_argument("--industry-neutral", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data_file, use_regime=True)
    if args.industry_neutral:
        if "industry" not in df.columns:
            print("⚠️ 数据缺少 industry 列，无法行业中性化，忽略该选项")
            args.industry_neutral = False
        else:
            factor_cols = [c for c in df.columns if c.endswith("_std")]
            df[factor_cols] = df.groupby(["date", "industry"])[factor_cols].transform(
                lambda x: x - x.mean()
            )
            if "forward_return_1d" in df.columns:
                df["forward_return_1d"] = df.groupby(["date", "industry"])[
                    "forward_return_1d"
                ].transform(lambda x: x - x.mean())

    mapping_file = args.mapping or find_latest_mapping()
    if not mapping_file:
        print("❌ 未找到制度因子映射 JSON")
        return
    with open(mapping_file, "r", encoding="utf-8") as f:
        regime_mapping = json.load(f)

    windows = [504, 756, 1008]
    alphas = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    rows = []
    for w in windows:
        for a in alphas:
            print(f">> 训练: window={w}, alpha={a}")
            pred = walk_forward_predict_regime_specific(
                df,
                regime_mapping,
                start_oos=args.start_oos,
                train_window_days=w,
                alpha=a,
            )
            if pred.empty:
                continue
            metrics = evaluate_predictions(
                pred, top_n=args.top_n, bottom_n=args.bottom_n, cost_bps=args.cost_bps
            )
            rows.append({"train_window": w, "alpha": a, **metrics})

    res = pd.DataFrame(rows)
    if res.empty:
        print("❌ 无有效结果")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("data") / f"regime_model_grid_results_{ts}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print("\nTop by LS_IR:")
    print(res.sort_values("ls_ir", ascending=False).head(5))
    print("\nTop by IC_IR:")
    print(res.sort_values("ic_ir", ascending=False).head(5))
    print(f"💾 保存: {out}")


if __name__ == "__main__":
    main()
