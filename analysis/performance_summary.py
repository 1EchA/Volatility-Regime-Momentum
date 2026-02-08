#!/usr/bin/env python3
"""汇总预测结果的性能指标"""

from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(path: Path, top_n: int, bottom_n: int, label: str) -> dict:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")
    groups = df.groupby("date")
    ic_series = groups.apply(lambda g: g["y_pred"].corr(g["y_true"], method="spearman"))

    rows = []
    prev_long = set()
    prev_short = set()
    for date, g in groups:
        g = g.sort_values("y_pred", ascending=False)
        lg = g.head(top_n)
        sg = g.tail(bottom_n)
        if lg.empty or sg.empty:
            continue
        long_ret = lg["y_true"].mean()
        short_ret = sg["y_true"].mean()
        ls = long_ret - short_ret
        rows.append({"date": date, "long": long_ret, "short": short_ret, "ls": ls})
        prev_long = set(lg["stock_code"])
        prev_short = set(sg["stock_code"])

    res = pd.DataFrame(rows).sort_values("date")
    res["cum_ls"] = res["ls"].cumsum()
    drawdown = res["cum_ls"] - res["cum_ls"].cummax()

    metrics = {
        "label": label,
        "file": path.name,
        "top_n": top_n,
        "bottom_n": bottom_n,
        "n_days": len(res),
        "ic_mean": ic_series.mean(),
        "ic_ir": ic_series.mean() / (ic_series.std() + 1e-12),
        "ic_win_rate": (ic_series > 0).mean(),
        "long_mean": res["long"].mean(),
        "short_mean": res["short"].mean(),
        "ls_mean": res["ls"].mean(),
        "ls_ann": res["ls"].mean() * 252,
        "ls_ir": (res["ls"].mean() / (res["ls"].std(ddof=1) + 1e-12)) * np.sqrt(252),
        "ls_win_rate": (res["ls"] > 0).mean(),
        "max_drawdown": drawdown.min(),
    }
    return metrics


def main():
    entries = [
        ("data/predictions_20250918_153334.csv", 30, 30, "zscore_top30_cost5bp"),
        ("data/predictions_20250922_092207.csv", 20, 20, "zscore_top20_cost3bp"),
        ("data/predictions_20250918_155237.csv", 30, 30, "rank_top30_cost5bp"),
        ("data/predictions_20250918_180604.csv", 20, 20, "rank_top20_cost3bp"),
        ("data/predictions_20250922_093929.csv", 25, 25, "zscore_top25_cost4bp"),
    ]

    results = []
    for report_path, top_n, bottom_n, label in entries:
        pred_path = Path(
            report_path.replace("report", "predictions").replace(".txt", ".csv")
        )
        if not pred_path.exists():
            continue
        metrics = evaluate(pred_path, top_n, bottom_n, label)
        results.append(metrics)

    if not results:
        print("❌ 未找到任何预测结果文件")
        return

    df = pd.DataFrame(results)
    out = Path("data/performance_summary.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df)
    print(f"💾 汇总保存至 {out}")


if __name__ == "__main__":
    main()
