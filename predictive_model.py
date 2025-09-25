#!/usr/bin/env python3
"""
交叉截面收益率预测模型（岭回归，支持分制度）

功能:
- 读取因子数据（推荐: data/simple_factor_data.csv 或最新制度数据）
- 自动识别标准化因子列(*_std)
- 使用滚动训练窗口（按天）拟合岭回归预测 next-day return（forward_return_1d）
- 可选按波动率制度分组建模（正常/高/极高）
- 样本外评估: 日度IC、IR、命中率、分组长短组合收益

输出:
- 预测明细: data/predictions_YYYYMMDD_HHMMSS.csv
- 报告: data/predictive_model_report_YYYYMMDD_HHMMSS.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from sklearn.linear_model import Ridge
except Exception:
    Ridge = None


def find_latest_regime_file() -> str | None:
    """查找最近的制度数据文件。
    优先 data/ 根目录；若无则回退搜索 data/archive/。
    """
    data_dir = Path('data')
    cands = list(data_dir.glob('volatility_regime_data_*.csv'))
    arch = data_dir / 'archive'
    if not cands and arch.exists():
        cands = list(arch.glob('volatility_regime_data_*.csv'))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])


def load_data(data_file: str | None = None, use_regime: bool = True) -> pd.DataFrame:
    if use_regime:
        rf = find_latest_regime_file()
        if rf:
            df = pd.read_csv(rf)
        elif data_file:
            df = pd.read_csv(data_file)
        else:
            raise FileNotFoundError('No regime or data file found')
    else:
        if not data_file:
            data_file = 'data/simple_factor_data.csv'
        df = pd.read_csv(data_file)
    # 类型处理
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'stock_code' not in df.columns and 'code' in df.columns:
        df['stock_code'] = df['code']
    # 按日期/股票排序
    df = df.sort_values(['date', 'stock_code']).reset_index(drop=True)
    return df


def detect_factor_columns(df: pd.DataFrame) -> list[str]:
    exclude = {'date', 'stock_code', 'close', 'daily_return',
               'forward_return_1d', 'forward_return_5d', 'forward_return_10d',
               'regime', 'volatility', 'market_return'}
    cols = [c for c in df.columns if c.endswith('_std') and c not in exclude]
    # 防止空
    if not cols:
        # 退化为简单集合
        fallback = [c for c in df.columns if c.startswith(('momentum_', 'volatility_', 'volume_ratio', 'turnover_rate', 'illiq'))]
        cols = [c for c in fallback if c not in exclude]
    return cols


def ridge_fit_predict(X_train, y_train, X_test, alpha=1.0):
    # 优先使用 sklearn
    if Ridge is not None:
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
        model.fit(X_train, y_train)
        return model.predict(X_test), model
    # 退化实现: 闭式解 (X'X + alpha I)^-1 X'y
    Xtr = np.asarray(X_train)
    ytr = np.asarray(y_train)
    n_features = Xtr.shape[1]
    XtX = Xtr.T @ Xtr
    regI = alpha * np.eye(n_features)
    beta = np.linalg.pinv(XtX + regI) @ (Xtr.T @ ytr)
    return np.asarray(X_test) @ beta, {'coef_': beta}


def walk_forward_predict(df: pd.DataFrame,
                         factor_cols: list[str],
                         start_oos: str = '2022-01-01',
                         train_window_days: int = 756,
                         alpha: float = 1.0,
                         use_regime: bool = True) -> pd.DataFrame:
    dates = sorted(df['date'].unique())
    start_dt = pd.to_datetime(start_oos)
    preds = []
    for t in dates:
        if t < start_dt:
            continue
        # 训练集窗口
        train_start = t - pd.Timedelta(days=train_window_days)
        train_mask = (df['date'] >= train_start) & (df['date'] < t)
        test_mask = (df['date'] == t)
        if use_regime and 'regime' in df.columns:
            reg = df.loc[test_mask, 'regime']
            if reg.empty:
                continue
            rname = reg.iloc[0]
            train_mask = train_mask & (df['regime'] == rname)
        train_df = df.loc[train_mask, ['date', 'stock_code', 'forward_return_1d'] + factor_cols].dropna()
        test_df = df.loc[test_mask, ['date', 'stock_code', 'forward_return_1d'] + factor_cols].dropna(subset=factor_cols)
        if len(train_df) < 1000 or test_df.empty:
            continue
        X_train = train_df[factor_cols].values
        y_train = train_df['forward_return_1d'].values
        X_test = test_df[factor_cols].values
        y_test = test_df['forward_return_1d'].values
        y_pred, _ = ridge_fit_predict(X_train, y_train, X_test, alpha=alpha)
        out = test_df[['date', 'stock_code']].copy()
        out['y_true'] = y_test
        out['y_pred'] = y_pred
        preds.append(out)
    if not preds:
        return pd.DataFrame()
    pred_df = pd.concat(preds, ignore_index=True)
    return pred_df

def walk_forward_predict_regime_specific(df: pd.DataFrame,
                                         regime_factors: dict,
                                         start_oos: str = '2022-01-01',
                                         train_window_days: int = 756,
                                         alpha: float = 1.0) -> pd.DataFrame:
    dates = sorted(df['date'].unique())
    start_dt = pd.to_datetime(start_oos)
    preds = []
    for t in dates:
        if t < start_dt:
            continue
        if 'regime' not in df.columns:
            break
        reg_today = df.loc[df['date'] == t, 'regime']
        if reg_today.empty:
            continue
        rname = reg_today.iloc[0]
        fcols = regime_factors.get(rname, [])
        if not fcols:
            continue
        train_start = t - pd.Timedelta(days=train_window_days)
        train_mask = (df['date'] >= train_start) & (df['date'] < t) & (df['regime'] == rname)
        test_mask = (df['date'] == t)
        train_df = df.loc[train_mask, ['date', 'stock_code', 'forward_return_1d'] + fcols].dropna()
        test_df = df.loc[test_mask, ['date', 'stock_code', 'forward_return_1d'] + fcols].dropna(subset=fcols)
        if len(train_df) < 800 or test_df.empty:
            continue
        X_train = train_df[fcols].values
        y_train = train_df['forward_return_1d'].values
        X_test = test_df[fcols].values
        y_test = test_df['forward_return_1d'].values
        y_pred, _ = ridge_fit_predict(X_train, y_train, X_test, alpha=alpha)
        out = test_df[['date', 'stock_code']].copy()
        out['y_true'] = y_test
        out['y_pred'] = y_pred
        preds.append(out)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def evaluate_predictions(pred_df: pd.DataFrame, top_n: int = 30, bottom_n: int = 30, cost_bps: float = 0.0) -> dict:
    # 日度IC
    ics = pred_df.groupby('date').apply(lambda g: g['y_pred'].corr(g['y_true'], method='spearman'))
    ic_series = ics.dropna()
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    win_rate = (ic_series > 0).mean()

    # 组合收益（等权，未计成本）
    long_ret = []
    short_ret = []
    ls_ret = []
    prev_long = set()
    prev_short = set()
    for dt, g in pred_df.groupby('date'):
        g = g.sort_values('y_pred', ascending=False)
        lg = g.head(top_n)
        sg = g.tail(bottom_n)
        if len(lg) < max(1, top_n//2) or len(sg) < max(1, bottom_n//2):
            continue
        long_r = lg['y_true'].mean()
        short_r = sg['y_true'].mean()
        long_ret.append(long_r)
        short_ret.append(short_r)
        ls = long_r - short_r
        if cost_bps > 0:
            curr_long = set(lg['stock_code'])
            curr_short = set(sg['stock_code'])
            overlap_long = len(prev_long & curr_long) / max(1, len(curr_long))
            overlap_short = len(prev_short & curr_short) / max(1, len(curr_short))
            turnover = (1 - overlap_long) + (1 - overlap_short)
            ls -= cost_bps * turnover
            prev_long, prev_short = curr_long, curr_short
        ls_ret.append(ls)

    long_ret = pd.Series(long_ret)
    short_ret = pd.Series(short_ret)
    ls_ret = pd.Series(ls_ret)

    metrics = {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'ic_win_rate': win_rate,
        'n_days': len(ic_series),
        'long_mean': long_ret.mean() if len(long_ret) else np.nan,
        'short_mean': short_ret.mean() if len(short_ret) else np.nan,
        'ls_mean': ls_ret.mean() if len(ls_ret) else np.nan,
        'ls_ann': ls_ret.mean() * 252 if len(ls_ret) else np.nan,
        'ls_ir': (ls_ret.mean() / (ls_ret.std() + 1e-12)) * np.sqrt(252) if len(ls_ret) else np.nan,
    }
    return metrics


def save_outputs(pred_df: pd.DataFrame, metrics: dict):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    pred_file = Path('data') / f'predictions_{ts}.csv'
    rep_file = Path('data') / f'predictive_model_report_{ts}.txt'
    pred_df.to_csv(pred_file, index=False, encoding='utf-8-sig')

    with open(rep_file, 'w', encoding='utf-8') as f:
        f.write('交叉截面收益率预测模型报告\n')
        f.write('='*50 + '\n\n')
        f.write(f"样本外天数: {metrics.get('n_days')}\n")
        f.write(f"IC均值: {metrics.get('ic_mean'):.6f}\n")
        f.write(f"IC_IR: {metrics.get('ic_ir'):.4f}\n")
        f.write(f"IC胜率: {metrics.get('ic_win_rate'):.2%}\n")
        f.write(f"多头日均收益: {metrics.get('long_mean'):.6f}\n")
        f.write(f"空头日均收益: {metrics.get('short_mean'):.6f}\n")
        f.write(f"多空日均: {metrics.get('ls_mean'):.6f}\n")
        f.write(f"多空年化(简单×252): {metrics.get('ls_ann'):.2%}\n")
        f.write(f"多空IR(年化): {metrics.get('ls_ir'):.3f}\n")
    return str(pred_file), str(rep_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='交叉截面预测模型（岭回归）')
    parser.add_argument('--data-file', type=str, default='data/simple_factor_data.csv')
    parser.add_argument('--start-oos', type=str, default='2022-01-01')
    parser.add_argument('--train-window-days', type=int, default=756)
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--no-regime', action='store_true')
    parser.add_argument('--factors-file', type=str, default=None,
                        help='指定因子列表文件（每行一个因子名）')
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--bottom-n', type=int, default=30)
    parser.add_argument('--cost-bps', type=float, default=0.0005)
    parser.add_argument('--regime-factors-file', type=str, default=None,
                        help='制度→因子列表的JSON文件（与--no-regime=False配合）')
    parser.add_argument('--industry-neutral', action='store_true',
                        help='是否按行业进行中性化（需数据包含industry列）')
    args = parser.parse_args()

    use_regime = not args.no_regime
    df = load_data(args.data_file, use_regime=use_regime)
    if args.factors_file and Path(args.factors_file).exists():
        with open(args.factors_file, 'r', encoding='utf-8') as f:
            factor_cols = [line.strip() for line in f if line.strip()]
    else:
        factor_cols = detect_factor_columns(df)

    print('📋 数据概览:')
    print(f"  观测: {len(df):,}, 股票: {df['stock_code'].nunique()}, 日期: {df['date'].nunique()}")
    print(f"  因子: {len(factor_cols)} → {factor_cols}")
    print(f"  制度建模: {'启用' if use_regime and 'regime' in df.columns else '禁用'}")

    if args.industry_neutral:
        if 'industry' not in df.columns:
            print('⚠️ 数据缺少 industry 列，无法行业中性化，忽略该选项')
            args.industry_neutral = False
        else:
            group_keys = ['date', 'industry']
            df[factor_cols] = df.groupby(group_keys)[factor_cols].transform(lambda x: x - x.mean())
            if 'forward_return_1d' in df.columns:
                df['forward_return_1d'] = df.groupby(group_keys)['forward_return_1d'].transform(lambda x: x - x.mean())

    regime_mapping = None
    pred_df = pd.DataFrame()
    if use_regime and args.regime_factors_file and Path(args.regime_factors_file).exists():
        import json
        with open(args.regime_factors_file, 'r', encoding='utf-8') as f:
            regime_mapping = json.load(f)
        pred_df = walk_forward_predict_regime_specific(
            df, regime_mapping,
            start_oos=args.start_oos,
            train_window_days=args.train_window_days,
            alpha=args.alpha
        )
    else:
        pred_df = walk_forward_predict(
            df, factor_cols,
            start_oos=args.start_oos,
            train_window_days=args.train_window_days,
            alpha=args.alpha,
            use_regime=use_regime and ('regime' in df.columns)
        )
    if pred_df.empty:
        print('❌ 无预测结果（样本不足）')
        return
    metrics = evaluate_predictions(pred_df, top_n=args.top_n, bottom_n=args.bottom_n, cost_bps=args.cost_bps)
    print('\n📊 样本外评估:')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    pred_file, rep_file = save_outputs(pred_df, metrics)
    print(f"\n💾 已保存: {pred_file}\n📝 报告: {rep_file}")


if __name__ == '__main__':
    main()
