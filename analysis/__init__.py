"""Analysis modules for performance reporting and execution strategies."""

from .execution_strategies import (
    baseline_daily,
    ema_hysteresis_combo,
    hysteresis_bands,
    low_freq_rebalance,
    swap_cap_limited,
)
from .performance_reporter import (
    compute_ic_series,
    compute_portfolio_timeseries,
    compute_regime_contributions,
    compute_summary_metrics,
)

__all__ = [
    "compute_ic_series",
    "compute_portfolio_timeseries",
    "compute_summary_metrics",
    "compute_regime_contributions",
    "baseline_daily",
    "hysteresis_bands",
    "ema_hysteresis_combo",
    "low_freq_rebalance",
    "swap_cap_limited",
]
