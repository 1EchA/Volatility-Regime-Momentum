#!/usr/bin/env python3
"""
低换手执行策略集合（独立于训练与打分），用于在给定预测明细上
构建不同的调仓机制与由此产生的收益、换手与成本序列。

输入假设（pred_df）：
- 列包含: `date`, `stock_code`, `y_true`, `y_pred`
- `date` 为可排序日期，`stock_code` 为股票代码字符串

输出：
- 按日时间序列DataFrame，含列：
  `date`, `long`, `short`, `ls_gross`, `turnover`, `cost_bps`, `ls_net`,
  `cum_ls_net`, `cum_ls_gross`, `drawdown`

注意：
- 成本以十进制计价（5bp=0.0005），与 analysis/performance_reporter 对齐。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    local = df.copy()
    local['date'] = pd.to_datetime(local['date'])
    local['stock_code'] = local['stock_code'].astype(str).str.zfill(6)
    return local.sort_values(['date', 'stock_code']).reset_index(drop=True)


def _finalise_rows(rows: list[dict]) -> pd.DataFrame:
    ts = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    if ts.empty:
        return ts
    ts['cum_ls_net'] = ts['ls_net'].cumsum()
    ts['cum_ls_gross'] = ts['ls_gross'].cumsum()
    peak = ts['cum_ls_net'].cummax()
    ts['drawdown'] = ts['cum_ls_net'] - peak
    return ts


def _compute_overlap(prev_set: set, curr_set: set) -> float:
    if not curr_set:
        return 0.0
    return len(prev_set & curr_set) / max(1, len(curr_set))


def baseline_daily(pred_df: pd.DataFrame,
                   top_n: int,
                   bottom_n: int,
                   cost_bps: float,
                   return_details: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, list]:
    """基线：每日按分数重排，等权多空，成本=重叠率定义的换手×cost。"""
    df = _prepare(pred_df)
    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    daily_sets: list = []
    for date, g in df.groupby('date'):
        g_sorted = g.sort_values('y_pred', ascending=False)
        longs = g_sorted.head(top_n)
        shorts = g_sorted.tail(bottom_n)
        if longs.empty or shorts.empty:
            continue
        long_ret = float(longs['y_true'].mean())
        short_ret = float(shorts['y_true'].mean())
        ls_gross = long_ret - short_ret
        curr_long = set(longs['stock_code'])
        curr_short = set(shorts['stock_code'])
        overlap_long = _compute_overlap(prev_long, curr_long)
        overlap_short = _compute_overlap(prev_short, curr_short)
        long_turn = (1 - overlap_long)
        short_turn = (1 - overlap_short)
        turnover = long_turn + short_turn
        added_long = len(curr_long - prev_long)
        removed_long = len(prev_long - curr_long)
        added_short = len(curr_short - prev_short)
        removed_short = len(prev_short - curr_short)
        ls_net = ls_gross - cost_bps * turnover
        rows.append({'date': date, 'long': long_ret, 'short': short_ret,
                     'ls_gross': ls_gross, 'turnover': turnover,
                     'cost_bps': cost_bps, 'ls_net': ls_net,
                     'overlap_long': overlap_long, 'overlap_short': overlap_short,
                     'long_turnover': long_turn, 'short_turnover': short_turn,
                     'added_long': added_long, 'removed_long': removed_long,
                     'added_short': added_short, 'removed_short': removed_short})
        daily_sets.append({'date': date, 'long': curr_long.copy(), 'short': curr_short.copy()})
        prev_long, prev_short = curr_long, curr_short
    ts = _finalise_rows(rows)
    return (ts, daily_sets) if return_details else ts


def low_freq_rebalance(pred_df: pd.DataFrame,
                       top_n: int,
                       bottom_n: int,
                       cost_bps: float,
                       k: int = 5,
                       return_details: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, list]:
    """低频再平衡：每 k 天才更新持仓，非再平衡日沿用前一日持仓。"""
    df = _prepare(pred_df)
    dates = sorted(df['date'].unique())
    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    curr_long: set[str] = set()
    curr_short: set[str] = set()
    daily_sets: list = []
    for idx, date in enumerate(dates):
        g = df[df['date'] == date].sort_values('y_pred', ascending=False)
        if idx % max(1, k) == 0 or not curr_long or not curr_short:
            longs = g.head(top_n)
            shorts = g.tail(bottom_n)
            curr_long = set(longs['stock_code'])
            curr_short = set(shorts['stock_code'])
        # 收益按当前持仓计算
        g_map = g.set_index('stock_code')['y_true']
        long_ret = float(pd.Series(list(curr_long)).map(g_map).dropna().mean()) if curr_long else np.nan
        short_ret = float(pd.Series(list(curr_short)).map(g_map).dropna().mean()) if curr_short else np.nan
        if np.isnan(long_ret) or np.isnan(short_ret):
            continue
        ls_gross = long_ret - short_ret
        # 只有再平衡日计换手
        if idx % max(1, k) == 0:
            overlap_long = _compute_overlap(prev_long, curr_long)
            overlap_short = _compute_overlap(prev_short, curr_short)
            long_turn = (1 - overlap_long)
            short_turn = (1 - overlap_short)
            turnover = long_turn + short_turn
            added_long = len(curr_long - prev_long)
            removed_long = len(prev_long - curr_long)
            added_short = len(curr_short - prev_short)
            removed_short = len(prev_short - curr_short)
        else:
            overlap_long = 1.0
            overlap_short = 1.0
            long_turn = 0.0
            short_turn = 0.0
            turnover = 0.0
            added_long = removed_long = added_short = removed_short = 0
        ls_net = ls_gross - cost_bps * turnover
        rows.append({'date': date, 'long': long_ret, 'short': short_ret,
                     'ls_gross': ls_gross, 'turnover': turnover,
                     'cost_bps': cost_bps, 'ls_net': ls_net,
                     'overlap_long': overlap_long, 'overlap_short': overlap_short,
                     'long_turnover': long_turn, 'short_turnover': short_turn,
                     'added_long': added_long, 'removed_long': removed_long,
                     'added_short': added_short, 'removed_short': removed_short})
        daily_sets.append({'date': date, 'long': curr_long.copy(), 'short': curr_short.copy()})
        if idx % max(1, k) == 0:
            prev_long, prev_short = curr_long.copy(), curr_short.copy()
    ts = _finalise_rows(rows)
    return (ts, daily_sets) if return_details else ts


def hysteresis_bands(pred_df: pd.DataFrame,
                     top_n: int,
                     bottom_n: int,
                     cost_bps: float,
                     delta: int = 10,
                     return_details: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, list]:
    """进出场滞后带：入选阈值 TopN，剔除阈值 Top(N+delta)。
    空头对称（BottomN 与 BottomN+delta）。"""
    df = _prepare(pred_df)
    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    daily_sets: list = []
    for date, g in df.groupby('date'):
        g_desc = g.sort_values('y_pred', ascending=False)
        g_asc = g_desc.iloc[::-1]
        # Long leg
        top_target = set(g_desc.head(top_n)['stock_code'])
        keep_zone = set(g_desc.head(top_n + max(0, delta))['stock_code'])
        curr_long = set([s for s in prev_long if s in keep_zone])
        if len(curr_long) < top_n:
            # 补足：用当日高分未持有股票
            candidates = [s for s in g_desc['stock_code'] if s not in curr_long]
            curr_long.update(candidates[: max(0, top_n - len(curr_long))])
        # Short leg（从分数低到高排序）
        bottom_target = set(g_asc.head(bottom_n)['stock_code'])
        keep_zone_s = set(g_asc.head(bottom_n + max(0, delta))['stock_code'])
        curr_short = set([s for s in prev_short if s in keep_zone_s])
        if len(curr_short) < bottom_n:
            candidates_s = [s for s in g_asc['stock_code'] if s not in curr_short]
            curr_short.update(candidates_s[: max(0, bottom_n - len(curr_short))])

        # 收益与成本
        long_ret = float(g[g['stock_code'].isin(curr_long)]['y_true'].mean()) if curr_long else np.nan
        short_ret = float(g[g['stock_code'].isin(curr_short)]['y_true'].mean()) if curr_short else np.nan
        if np.isnan(long_ret) or np.isnan(short_ret):
            continue
        ls_gross = long_ret - short_ret
        overlap_long = _compute_overlap(prev_long, curr_long)
        overlap_short = _compute_overlap(prev_short, curr_short)
        long_turn = (1 - overlap_long)
        short_turn = (1 - overlap_short)
        turnover = long_turn + short_turn
        added_long = len(curr_long - prev_long)
        removed_long = len(prev_long - curr_long)
        added_short = len(curr_short - prev_short)
        removed_short = len(prev_short - curr_short)
        ls_net = ls_gross - cost_bps * turnover
        rows.append({'date': date, 'long': long_ret, 'short': short_ret,
                     'ls_gross': ls_gross, 'turnover': turnover,
                     'cost_bps': cost_bps, 'ls_net': ls_net,
                     'overlap_long': overlap_long, 'overlap_short': overlap_short,
                     'long_turnover': long_turn, 'short_turnover': short_turn,
                     'added_long': added_long, 'removed_long': removed_long,
                     'added_short': added_short, 'removed_short': removed_short})
        daily_sets.append({'date': date, 'long': curr_long.copy(), 'short': curr_short.copy()})
        prev_long, prev_short = curr_long, curr_short
    ts = _finalise_rows(rows)
    return (ts, daily_sets) if return_details else ts


def ema_smoothed(pred_df: pd.DataFrame,
                 top_n: int,
                 bottom_n: int,
                 cost_bps: float,
                 ema_span: int = 5,
                 return_details: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, list]:
    """对 y_pred 进行按股票的 EMA 平滑后再做基线日更组合。
    返回：(时序数据, 附加了 y_score 列的预测明细)，便于外部计算 IC。"""
    df = _prepare(pred_df)
    df['y_score'] = (
        df.sort_values('date')
          .groupby('stock_code')['y_pred']
          .transform(lambda s: s.ewm(span=max(1, int(ema_span)), adjust=False).mean())
    )
    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    daily_sets: list = []
    for date, g in df.groupby('date'):
        g_sorted = g.sort_values('y_score', ascending=False)
        longs = g_sorted.head(top_n)
        shorts = g_sorted.tail(bottom_n)
        if longs.empty or shorts.empty:
            continue
        long_ret = float(longs['y_true'].mean())
        short_ret = float(shorts['y_true'].mean())
        ls_gross = long_ret - short_ret
        curr_long = set(longs['stock_code'])
        curr_short = set(shorts['stock_code'])
        overlap_long = _compute_overlap(prev_long, curr_long)
        overlap_short = _compute_overlap(prev_short, curr_short)
        long_turn = (1 - overlap_long)
        short_turn = (1 - overlap_short)
        turnover = long_turn + short_turn
        added_long = len(curr_long - prev_long)
        removed_long = len(prev_long - curr_long)
        added_short = len(curr_short - prev_short)
        removed_short = len(prev_short - curr_short)
        ls_net = ls_gross - cost_bps * turnover
        rows.append({'date': date, 'long': long_ret, 'short': short_ret,
                     'ls_gross': ls_gross, 'turnover': turnover,
                     'cost_bps': cost_bps, 'ls_net': ls_net,
                     'overlap_long': overlap_long, 'overlap_short': overlap_short,
                     'long_turnover': long_turn, 'short_turnover': short_turn,
                     'added_long': added_long, 'removed_long': removed_long,
                     'added_short': added_short, 'removed_short': removed_short})
        prev_long, prev_short = curr_long, curr_short
    ts = _finalise_rows(rows)
    if return_details:
        return ts, df, daily_sets
    return ts, df


def swap_cap_limited(pred_df: pd.DataFrame,
                     top_n: int,
                     bottom_n: int,
                     cost_bps: float,
                     swap_cap_ratio: float = 0.2,
                     return_details: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, list]:
    """换手上限：每日计算目标组合，但限制每侧最多替换比例。"""
    df = _prepare(pred_df)
    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    daily_sets: list = []
    for date, g in df.groupby('date'):
        g_desc = g.sort_values('y_pred', ascending=False)
        g_asc = g_desc.iloc[::-1]
        target_long = list(g_desc.head(top_n)['stock_code'])
        target_short = list(g_asc.head(bottom_n)['stock_code'])

        # Long side
        curr_long = set(prev_long)
        to_add = [s for s in target_long if s not in curr_long]
        to_drop = [s for s in curr_long if s not in set(target_long)]
        # 按当前得分排序：待加按高分优先，待删按低分优先
        score_map = g.set_index('stock_code')['y_pred']
        to_add_sorted = sorted(to_add, key=lambda s: score_map.get(s, -np.inf), reverse=True)
        to_drop_sorted = sorted(to_drop, key=lambda s: score_map.get(s, np.inf))
        cap = int(np.ceil(max(0.0, min(1.0, swap_cap_ratio)) * top_n))
        # 执行有限互换
        drop_exec = to_drop_sorted[:cap]
        add_exec = to_add_sorted[:cap]
        curr_long.difference_update(drop_exec)
        need_add = max(0, top_n - len(curr_long))
        curr_long.update(add_exec[:need_add])
        # 若仍不足，补齐（严格保证持仓数量）
        if len(curr_long) < top_n:
            for s in target_long:
                if s not in curr_long:
                    curr_long.add(s)
                if len(curr_long) >= top_n:
                    break

        # Short side（对称处理，排序按低分优先进入）
        curr_short = set(prev_short)
        to_add_s = [s for s in target_short if s not in curr_short]
        to_drop_s = [s for s in curr_short if s not in set(target_short)]
        to_add_s_sorted = sorted(to_add_s, key=lambda s: score_map.get(s, np.inf))  # 低分优先
        to_drop_s_sorted = sorted(to_drop_s, key=lambda s: score_map.get(s, -np.inf), reverse=True)
        cap_s = int(np.ceil(max(0.0, min(1.0, swap_cap_ratio)) * bottom_n))
        drop_exec_s = to_drop_s_sorted[:cap_s]
        add_exec_s = to_add_s_sorted[:cap_s]
        curr_short.difference_update(drop_exec_s)
        need_add_s = max(0, bottom_n - len(curr_short))
        # 低分优先补齐
        curr_short.update(add_exec_s[:need_add_s])
        if len(curr_short) < bottom_n:
            for s in target_short:
                if s not in curr_short:
                    curr_short.add(s)
                if len(curr_short) >= bottom_n:
                    break

        # 收益与成本
        long_ret = float(g[g['stock_code'].isin(curr_long)]['y_true'].mean()) if curr_long else np.nan
        short_ret = float(g[g['stock_code'].isin(curr_short)]['y_true'].mean()) if curr_short else np.nan
        if np.isnan(long_ret) or np.isnan(short_ret):
            continue
        ls_gross = long_ret - short_ret
        overlap_long = _compute_overlap(prev_long, curr_long)
        overlap_short = _compute_overlap(prev_short, curr_short)
        long_turn = (1 - overlap_long)
        short_turn = (1 - overlap_short)
        turnover = long_turn + short_turn
        added_long = len(curr_long - prev_long)
        removed_long = len(prev_long - curr_long)
        added_short = len(curr_short - prev_short)
        removed_short = len(prev_short - curr_short)
        ls_net = ls_gross - cost_bps * turnover
        rows.append({'date': date, 'long': long_ret, 'short': short_ret,
                     'ls_gross': ls_gross, 'turnover': turnover,
                     'cost_bps': cost_bps, 'ls_net': ls_net,
                     'overlap_long': overlap_long, 'overlap_short': overlap_short,
                     'long_turnover': long_turn, 'short_turnover': short_turn,
                     'added_long': added_long, 'removed_long': removed_long,
                     'added_short': added_short, 'removed_short': removed_short})
        daily_sets.append({'date': date, 'long': curr_long.copy(), 'short': curr_short.copy()})
        prev_long, prev_short = curr_long, curr_short
    ts = _finalise_rows(rows)
    return (ts, daily_sets) if return_details else ts


def ema_hysteresis_combo(pred_df: pd.DataFrame,
                         top_n: int,
                         bottom_n: int,
                         cost_bps: float,
                         ema_span: int = 4,
                         delta: int = 15,
                         return_details: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, list]:
    """EMA+滞后带组合策略：先对信号做EMA平滑，再应用滞后带策略。
    返回：(时序数据, 附加了 y_score 列的预测明细)，便于外部计算 IC。"""
    df = _prepare(pred_df)
    # 第一步：EMA平滑信号
    df['y_score'] = (
        df.sort_values('date')
          .groupby('stock_code')['y_pred']
          .transform(lambda s: s.ewm(span=max(1, int(ema_span)), adjust=False).mean())
    )
    
    # 第二步：基于平滑信号应用滞后带策略
    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    
    daily_sets: list = []
    for date, g in df.groupby('date'):
        g_desc = g.sort_values('y_score', ascending=False)  # 使用平滑后的分数
        g_asc = g_desc.iloc[::-1]
        
        # Long leg with hysteresis
        top_target = set(g_desc.head(top_n)['stock_code'])
        keep_zone = set(g_desc.head(top_n + max(0, delta))['stock_code'])
        curr_long = set([s for s in prev_long if s in keep_zone])
        if len(curr_long) < top_n:
            # 补足：用当日高分未持有股票
            candidates = [s for s in g_desc['stock_code'] if s not in curr_long]
            curr_long.update(candidates[: max(0, top_n - len(curr_long))])
            
        # Short leg with hysteresis（从分数低到高排序）
        bottom_target = set(g_asc.head(bottom_n)['stock_code'])
        keep_zone_s = set(g_asc.head(bottom_n + max(0, delta))['stock_code'])
        curr_short = set([s for s in prev_short if s in keep_zone_s])
        if len(curr_short) < bottom_n:
            candidates_s = [s for s in g_asc['stock_code'] if s not in curr_short]
            curr_short.update(candidates_s[: max(0, bottom_n - len(curr_short))])

        # 收益与成本计算
        long_ret = float(g[g['stock_code'].isin(curr_long)]['y_true'].mean()) if curr_long else np.nan
        short_ret = float(g[g['stock_code'].isin(curr_short)]['y_true'].mean()) if curr_short else np.nan
        if np.isnan(long_ret) or np.isnan(short_ret):
            continue
            
        ls_gross = long_ret - short_ret
        overlap_long = _compute_overlap(prev_long, curr_long)
        overlap_short = _compute_overlap(prev_short, curr_short)
        long_turn = (1 - overlap_long)
        short_turn = (1 - overlap_short)
        turnover = long_turn + short_turn
        ls_net = ls_gross - cost_bps * turnover
        
        added_long = len(curr_long - prev_long)
        removed_long = len(prev_long - curr_long)
        added_short = len(curr_short - prev_short)
        removed_short = len(prev_short - curr_short)
        rows.append({
            'date': date, 'long': long_ret, 'short': short_ret,
            'ls_gross': ls_gross, 'turnover': turnover,
            'cost_bps': cost_bps, 'ls_net': ls_net,
            'overlap_long': overlap_long, 'overlap_short': overlap_short,
            'long_turnover': long_turn, 'short_turnover': short_turn,
            'added_long': added_long, 'removed_long': removed_long,
            'added_short': added_short, 'removed_short': removed_short
        })
        daily_sets.append({'date': date, 'long': curr_long.copy(), 'short': curr_short.copy()})
        prev_long, prev_short = curr_long, curr_short
    ts = _finalise_rows(rows)
    if return_details:
        return ts, df, daily_sets
    return ts, df


def compute_ic_series_with_score(pred_df: pd.DataFrame, score_col: str = 'y_pred') -> pd.Series:
    """按指定分数列计算日度IC（Spearman）。避开 pandas 分组警告。"""
    df = _prepare(pred_df)
    # 仅在需要的两列上分组，避免 FutureWarning
    ic = (
        df.groupby('date')[[score_col, 'y_true']]
          .apply(lambda g: g[score_col].corr(g['y_true'], method='spearman'))
    )
    return ic.dropna()
