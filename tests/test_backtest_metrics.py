"""Tests for FinancialMetrics."""

import numpy as np
import pytest

from currency_predictor.backtesting.metrics import FinancialMetrics, FinancialMetricsResult


class TestComputeReturns:
    """Test return calculation."""

    def test_basic_returns(self):
        prices = np.array([100, 110, 105, 115])
        returns = FinancialMetrics.compute_returns(prices)
        expected = np.array([0.1, -0.04545454545, 0.095238095])
        np.testing.assert_allclose(returns, expected, rtol=1e-5)

    def test_single_price(self):
        returns = FinancialMetrics.compute_returns(np.array([100]))
        assert len(returns) == 0

    def test_empty_prices(self):
        returns = FinancialMetrics.compute_returns(np.array([]))
        assert len(returns) == 0


class TestSharpeRatio:
    """Test Sharpe ratio calculation."""

    def test_positive_returns(self):
        """Positive returns should give positive Sharpe."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        sharpe = FinancialMetrics.sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe > 0

    def test_zero_returns(self):
        """All-zero returns (zero std) should return 0."""
        returns = np.array([0.0, 0.0, 0.0, 0.0])
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_empty_returns(self):
        sharpe = FinancialMetrics.sharpe_ratio(np.array([]))
        assert sharpe == 0.0

    def test_with_risk_free_rate(self):
        """Higher risk-free rate should reduce Sharpe."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        s1 = FinancialMetrics.sharpe_ratio(returns, risk_free_rate=0.0)
        s2 = FinancialMetrics.sharpe_ratio(returns, risk_free_rate=0.05)
        assert s2 < s1

    def test_not_annualized(self):
        """Non-annualized Sharpe should be smaller magnitude."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        s_ann = FinancialMetrics.sharpe_ratio(returns, annualize=True)
        s_raw = FinancialMetrics.sharpe_ratio(returns, annualize=False)
        assert abs(s_ann) > abs(s_raw)


class TestMaxDrawdown:
    """Test max drawdown calculation."""

    def test_no_drawdown(self):
        """Monotonically increasing equity has 0 drawdown."""
        equity = np.array([100, 110, 120, 130, 140])
        dd, dur = FinancialMetrics.max_drawdown(equity)
        assert dd == 0.0
        assert dur == 0

    def test_known_drawdown(self):
        """Hand-calculated drawdown example."""
        # Peak at 120, trough at 90 → drawdown = 30/120 = 0.25
        equity = np.array([100, 120, 90, 110, 130])
        dd, dur = FinancialMetrics.max_drawdown(equity)
        assert abs(dd - 0.25) < 1e-10
        assert dur >= 1

    def test_full_recovery(self):
        equity = np.array([100, 80, 90, 100, 110])
        dd, dur = FinancialMetrics.max_drawdown(equity)
        assert dd == pytest.approx(0.2)

    def test_single_point(self):
        dd, dur = FinancialMetrics.max_drawdown(np.array([100]))
        assert dd == 0.0
        assert dur == 0


class TestWinRate:
    """Test win rate calculation."""

    def test_perfect_prediction(self):
        """100% correct direction."""
        actual = np.array([0.01, 0.02, -0.01, 0.03])
        predicted = np.array([0.005, 0.01, -0.02, 0.01])
        wr = FinancialMetrics.win_rate(actual, predicted)
        assert wr == 1.0

    def test_zero_win_rate(self):
        """All wrong direction."""
        actual = np.array([0.01, 0.02, 0.01])
        predicted = np.array([-0.01, -0.02, -0.01])
        wr = FinancialMetrics.win_rate(actual, predicted)
        assert wr == 0.0

    def test_partial_win_rate(self):
        """2 out of 4 correct."""
        actual = np.array([0.01, -0.01, 0.01, -0.01])
        predicted = np.array([-0.01, -0.01, -0.01, -0.01])
        wr = FinancialMetrics.win_rate(actual, predicted)
        assert wr == 0.5

    def test_empty_returns(self):
        wr = FinancialMetrics.win_rate(np.array([]), np.array([]))
        assert wr == 0.0


class TestProfitFactor:
    """Test profit factor calculation."""

    def test_all_wins(self):
        """All winning trades → infinite profit factor."""
        actual = np.array([0.01, 0.02, 0.03])
        directions = np.array([1.0, 1.0, 1.0])
        pf = FinancialMetrics.profit_factor(actual, directions)
        assert pf == float("inf")

    def test_known_value(self):
        """Hand-calculated profit factor."""
        actual = np.array([0.02, -0.01, 0.03, -0.005])
        directions = np.array([1.0, 1.0, 1.0, 1.0])  # always long
        pf = FinancialMetrics.profit_factor(actual, directions)
        # profit: 0.02 + 0.03 = 0.05, loss: 0.01 + 0.005 = 0.015
        assert pf == pytest.approx(0.05 / 0.015, rel=1e-5)

    def test_no_trades(self):
        """All flat → no gains, no losses."""
        actual = np.array([0.01, 0.02])
        directions = np.array([0.0, 0.0])  # flat
        pf = FinancialMetrics.profit_factor(actual, directions)
        assert pf == 0.0


class TestEquityCurve:
    """Test equity curve simulation."""

    def test_always_long_bull(self):
        """Always long in rising market → equity grows."""
        prices = np.array([100, 110, 121, 133.1])
        directions = np.array([1.0, 1.0, 1.0])
        equity = FinancialMetrics.compute_equity_curve(prices, directions, 10000)
        assert equity[0] == 10000
        assert equity[-1] > 10000

    def test_always_flat(self):
        """Always flat → equity stays constant."""
        prices = np.array([100, 110, 90, 120])
        directions = np.array([0.0, 0.0, 0.0])
        equity = FinancialMetrics.compute_equity_curve(prices, directions, 10000)
        np.testing.assert_allclose(equity, [10000, 10000, 10000, 10000])

    def test_single_price(self):
        equity = FinancialMetrics.compute_equity_curve(
            np.array([100]), np.array([]), 10000
        )
        assert len(equity) == 1
        assert equity[0] == 10000

    def test_equity_matches_buy_hold(self):
        """Always-long equity should match buy-and-hold."""
        prices = np.array([100, 105, 110, 108, 115])
        directions = np.array([1.0, 1.0, 1.0, 1.0])
        equity = FinancialMetrics.compute_equity_curve(prices, directions, 10000)
        # Buy-and-hold: 10000 * (115/100) = 11500
        assert equity[-1] == pytest.approx(11500, rel=1e-5)


class TestComputeAll:
    """Test compute_all integration."""

    def test_returns_correct_type(self):
        actual = np.array([100, 105, 110, 108, 115, 112, 118])
        predicted = np.array([100, 104, 109, 110, 114, 113, 117])
        result = FinancialMetrics.compute_all(actual, predicted)
        assert isinstance(result, FinancialMetricsResult)

    def test_all_fields_present(self):
        actual = np.array([100, 105, 110, 108, 115, 112, 118])
        predicted = np.array([100, 104, 109, 110, 114, 113, 117])
        result = FinancialMetrics.compute_all(actual, predicted)

        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "annualized_return")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "max_drawdown_duration")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "profit_factor")
        assert hasattr(result, "cumulative_return")
        assert hasattr(result, "volatility")
        assert hasattr(result, "calmar_ratio")

    def test_max_drawdown_non_negative(self):
        actual = np.array([100, 95, 90, 85, 80])
        predicted = np.array([100, 96, 91, 86, 81])
        result = FinancialMetrics.compute_all(actual, predicted)
        assert result.max_drawdown >= 0

    def test_win_rate_bounded(self):
        actual = np.array([100, 105, 103, 108, 106, 110])
        predicted = np.array([100, 104, 104, 107, 107, 109])
        result = FinancialMetrics.compute_all(actual, predicted)
        assert 0.0 <= result.win_rate <= 1.0
