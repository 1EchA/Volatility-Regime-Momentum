import glob
import os
import pytest


def latest(pattern: str):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def test_predictions_exist_or_skip():
    f = latest("data/predictions_*.csv")
    if not f:
        pytest.skip("no predictions_* found (run pipeline first)")
    assert os.path.getsize(f) > 0


def test_execution_exist_or_skip():
    ts = latest("data/pipeline_execution_*_timeseries.csv")
    met = latest("data/pipeline_execution_*_metrics.json")
    if not (ts and met):
        pytest.skip("no pipeline_execution_* artifacts found (run pipeline with execution layer)")
    assert os.path.getsize(ts) > 0
    assert os.path.getsize(met) > 0

