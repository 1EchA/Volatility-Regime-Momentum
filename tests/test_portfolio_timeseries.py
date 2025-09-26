import pandas as pd
from datetime import datetime, timedelta

from analysis.performance_reporter import compute_portfolio_timeseries, compute_summary_metrics


def _toy_predictions():
    # 两个日期、4只股票的迷你数据集
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(2):
        dt = base + timedelta(days=i)
        for s, pred, true in [
            ("000001", 0.9 - i*0.1, 0.02 - i*0.01),
            ("000002", 0.8 - i*0.1, 0.01 - i*0.01),
            ("000003", 0.2 + i*0.1, -0.01 + i*0.005),
            ("000004", 0.1 + i*0.1, -0.02 + i*0.005),
        ]:
            rows.append({"date": dt, "stock_code": s, "y_pred": pred, "y_true": true})
    return pd.DataFrame(rows)


def test_compute_portfolio_timeseries_minimal():
    df = _toy_predictions()
    ts = compute_portfolio_timeseries(df, top_n=2, bottom_n=2, cost_bps=0.0005)
    # 应至少包含两个交易日
    assert len(ts) >= 2
    # 核心列存在
    for col in ["date", "long", "short", "ls_net", "cum_ls_net", "drawdown", "turnover"]:
        assert col in ts.columns

    # 计算摘要指标不报错
    # 这里用简单的IC序列（按日期分组的秩相关在性能报告函数里计算），
    # 测试时用 y_true 与自身的相关性代替，确保接口可用
    ic_series = df.groupby('date')['y_true'].mean()  # 仅为占位
    metrics = compute_summary_metrics(ts, ic_series)
    assert "ls_ann" in metrics and "ls_ir" in metrics

