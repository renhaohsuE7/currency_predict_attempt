"""Backtesting framework for walk-forward model validation."""

from .metrics import FinancialMetrics, FinancialMetricsResult
from .result import BacktestResult, FoldResult
from .runner import BacktestRunner
from .splitter import FoldSpec, WalkForwardSplitter

__all__ = [
    "BacktestRunner",
    "BacktestResult",
    "FinancialMetrics",
    "FinancialMetricsResult",
    "FoldResult",
    "FoldSpec",
    "WalkForwardSplitter",
]
