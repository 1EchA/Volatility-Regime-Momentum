"""Analysis modules for performance reporting and execution strategies."""

from .performance_reporter import (
    compute_ic_series,
    compute_portfolio_timeseries,
    compute_summary_metrics,
    compute_regime_contributions,
)
from .execution_strategies import (
    baseline_daily,
    hysteresis_bands,
    ema_hysteresis_combo,
    low_freq_rebalance,
    swap_cap_limited,
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
