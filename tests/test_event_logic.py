import pandas as pd


def detect_events(df: pd.DataFrame) -> pd.DataFrame:
    """复制应用中的事件识别逻辑，返回包含四类事件布尔列的数据框。"""
    evt = df[['date', 'in_long', 'in_short']].copy()
    evt[['in_long', 'in_short']] = evt[['in_long', 'in_short']].fillna(False).astype(bool)
    prev = evt[['in_long', 'in_short']].shift(1).fillna(False)
    evt['long_open'] = (~prev['in_long']) & (evt['in_long'])
    evt['short_open'] = (~prev['in_short']) & (evt['in_short'])
    evt['long_close'] = (prev['in_long']) & (~evt['in_long'])
    evt['short_close'] = (prev['in_short']) & (~evt['in_short'])
    return evt


def test_event_open_close_counts():
    dates = pd.date_range('2024-01-01', periods=6, freq='D')
    # in_long 序列: F F T T F F  → open at t2, close at t4
    # in_short序列: F T T F F F  → open at t1, close at t3
    df = pd.DataFrame({
        'date': dates,
        'in_long': [False, False, True, True, False, False],
        'in_short': [False, True, True, False, False, False],
    })
    evt = detect_events(df)
    assert evt['long_open'].sum() == 1
    assert evt['short_open'].sum() == 1
    assert evt['long_close'].sum() == 1
    assert evt['short_close'].sum() == 1

