"""Financial metrics for backtesting evaluation."""

from dataclasses import dataclass

import numpy as np


TRADING_DAYS_PER_YEAR = 252


@dataclass
class FinancialMetricsResult:
    """Container for financial backtest metrics."""

    sharpe_ratio: float
    annualized_return: float
    max_drawdown: float
    max_drawdown_duration: int  # in trading days
    win_rate: float
    profit_factor: float
    cumulative_return: float
    volatility: float  # annualized
    calmar_ratio: float


class FinancialMetrics:
    """Compute financial metrics from actual vs predicted price series.

    All methods are static — no state. Each can be tested independently.
    """

    @staticmethod
    def compute_returns(prices: np.ndarray) -> np.ndarray:
        """Compute simple returns from price series.

        Returns:
            Array of length len(prices) - 1.
        """
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 2:
            return np.array([], dtype=float)
        return np.diff(prices) / prices[:-1]

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualize: bool = True,
    ) -> float:
        """Compute Sharpe ratio.

        Args:
            returns: Array of period returns.
            risk_free_rate: Annualized risk-free rate.
            annualize: Whether to annualize the ratio.

        Returns:
            Sharpe ratio. Returns 0.0 if std is zero or returns is empty.
        """
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            return 0.0

        rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        excess = returns - rf_daily
        std = np.std(excess, ddof=1) if len(excess) > 1 else 0.0

        if std == 0.0:
            return 0.0

        ratio = float(np.mean(excess) / std)
        if annualize:
            ratio *= np.sqrt(TRADING_DAYS_PER_YEAR)
        return ratio

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> tuple[float, int]:
        """Compute maximum drawdown and its duration.

        Args:
            equity_curve: Cumulative equity values over time.

        Returns:
            (max_drawdown_fraction, max_drawdown_duration_days)
            drawdown is expressed as a positive fraction (e.g. 0.15 = 15% decline).
        """
        equity = np.asarray(equity_curve, dtype=float)
        if len(equity) < 2:
            return 0.0, 0

        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max

        max_dd = float(np.max(drawdowns))

        # Compute duration of the deepest drawdown
        max_duration = 0
        current_duration = 0
        for dd in drawdowns:
            if dd > 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, max_duration

    @staticmethod
    def win_rate(
        actual_returns: np.ndarray,
        predicted_returns: np.ndarray,
    ) -> float:
        """Percentage of correctly predicted return directions.

        Args:
            actual_returns: Actual period returns.
            predicted_returns: Predicted period returns.

        Returns:
            Win rate as a fraction (0.0 to 1.0).
        """
        actual = np.asarray(actual_returns, dtype=float)
        predicted = np.asarray(predicted_returns, dtype=float)

        if len(actual) == 0:
            return 0.0

        actual_dir = np.sign(actual)
        pred_dir = np.sign(predicted)
        return float(np.mean(actual_dir == pred_dir))

    @staticmethod
    def profit_factor(
        actual_returns: np.ndarray,
        predicted_directions: np.ndarray,
    ) -> float:
        """Gross profit / gross loss from following predicted signals.

        Strategy: go long when predicted direction > 0, flat otherwise.

        Args:
            actual_returns: Actual period returns.
            predicted_directions: Predicted directions (sign of predicted returns).

        Returns:
            Profit factor. Returns inf if no losses, 0.0 if no gains.
        """
        actual = np.asarray(actual_returns, dtype=float)
        directions = np.sign(np.asarray(predicted_directions, dtype=float))

        if len(actual) == 0:
            return 0.0

        # Strategy returns: earn actual return when long (direction > 0)
        strategy_returns = actual * (directions > 0).astype(float)

        gross_profit = float(np.sum(strategy_returns[strategy_returns > 0]))
        gross_loss = float(np.abs(np.sum(strategy_returns[strategy_returns < 0])))

        if gross_loss == 0.0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def compute_equity_curve(
        actual_prices: np.ndarray,
        predicted_directions: np.ndarray,
        initial_capital: float = 10000.0,
    ) -> np.ndarray:
        """Simulate equity curve based on prediction signals.

        Simple strategy: go long when predicted direction is up, flat otherwise.
        The first value in the returned array is initial_capital.

        Args:
            actual_prices: Actual price series.
            predicted_directions: Predicted directions for each period.
            initial_capital: Starting capital.

        Returns:
            Equity curve array of length len(actual_prices).
        """
        prices = np.asarray(actual_prices, dtype=float)
        directions = np.sign(np.asarray(predicted_directions, dtype=float))

        if len(prices) < 2:
            return np.array([initial_capital], dtype=float)

        returns = np.diff(prices) / prices[:-1]
        # Only participate when predicted direction is positive
        strategy_returns = returns * (directions[: len(returns)] > 0).astype(float)

        equity = np.empty(len(prices), dtype=float)
        equity[0] = initial_capital
        for i, r in enumerate(strategy_returns):
            equity[i + 1] = equity[i] * (1 + r)

        return equity

    @classmethod
    def compute_all(
        cls,
        actual_prices: np.ndarray,
        predicted_prices: np.ndarray,
        risk_free_rate: float = 0.0,
        initial_capital: float = 10000.0,
    ) -> FinancialMetricsResult:
        """Compute all financial metrics at once.

        Args:
            actual_prices: Actual price series.
            predicted_prices: Predicted price series (same length).
            risk_free_rate: Annualized risk-free rate.
            initial_capital: Starting capital for equity curve.

        Returns:
            FinancialMetricsResult with all computed metrics.
        """
        actual = np.asarray(actual_prices, dtype=float)
        predicted = np.asarray(predicted_prices, dtype=float)

        actual_returns = cls.compute_returns(actual)
        predicted_returns = cls.compute_returns(predicted)

        # Equity curve from strategy
        equity = cls.compute_equity_curve(actual, predicted_returns, initial_capital)

        # Sharpe
        strategy_returns = actual_returns * (np.sign(predicted_returns[: len(actual_returns)]) > 0).astype(float)
        sharpe = cls.sharpe_ratio(strategy_returns, risk_free_rate)

        # Drawdown
        dd, dd_dur = cls.max_drawdown(equity)

        # Win rate
        wr = cls.win_rate(actual_returns, predicted_returns[: len(actual_returns)])

        # Profit factor
        pf = cls.profit_factor(actual_returns, predicted_returns[: len(actual_returns)])

        # Cumulative return
        cum_return = float((equity[-1] / equity[0]) - 1) if len(equity) > 0 else 0.0

        # Annualized return
        n_days = len(actual_returns)
        if n_days > 0 and cum_return > -1:
            ann_return = float((1 + cum_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1)
        else:
            ann_return = 0.0

        # Volatility (annualized)
        vol = float(np.std(strategy_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(strategy_returns) > 1 else 0.0

        # Calmar ratio
        calmar = ann_return / dd if dd > 0 else 0.0

        return FinancialMetricsResult(
            sharpe_ratio=sharpe,
            annualized_return=ann_return,
            max_drawdown=dd,
            max_drawdown_duration=dd_dur,
            win_rate=wr,
            profit_factor=pf,
            cumulative_return=cum_return,
            volatility=vol,
            calmar_ratio=calmar,
        )
