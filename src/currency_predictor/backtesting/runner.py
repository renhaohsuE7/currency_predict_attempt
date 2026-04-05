"""BacktestRunner — orchestrates walk-forward backtesting."""

import logging
import time
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .metrics import FinancialMetrics
from .result import BacktestResult, FoldResult
from .splitter import WalkForwardSplitter

from ..data.collectors import YahooFinanceCollector
from ..data_processor import DataProcessor
from ..models.factory import ModelFactory
from ..prediction.comparer import ModelComparer
from ..utils.asset_type import classify_symbol, AssetType

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Orchestrates walk-forward backtesting across models and symbols.

    For each (symbol, model) pair:
    1. Load & preprocess all available data
    2. Generate walk-forward folds via WalkForwardSplitter
    3. For each fold: create fresh model, train, predict, evaluate
    4. Compute prediction metrics + financial metrics
    5. Aggregate results across folds
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "results",
    ):
        self.config = config
        self.output_dir = output_dir
        self.data_processor = DataProcessor()
        self.collector = YahooFinanceCollector()

    def run(
        self,
        symbols: List[str],
        model_names: Optional[List[str]] = None,
        strategy: str = "rolling",
        initial_train_days: int = 252,
        test_step_days: int = 30,
        test_window_days: int = 30,
        data_period: str = "3y",
    ) -> Dict[str, Dict[str, BacktestResult]]:
        """Run walk-forward backtest.

        Args:
            symbols: List of ticker symbols.
            model_names: Model names (defaults to config model).
            strategy: "rolling" or "expanding".
            initial_train_days: Initial training window size.
            test_step_days: Step forward per fold.
            test_window_days: Test window size.
            data_period: Data collection period.

        Returns:
            Nested dict: {symbol: {model_name: BacktestResult}}
        """
        if model_names is None:
            model_names = [self.config.get("model_name", "patchtst_sklearn")]

        results: Dict[str, Dict[str, BacktestResult]] = {}

        for symbol in symbols:
            logger.info(f"Backtesting symbol: {symbol}")
            results[symbol] = {}

            # Collect and process data once per symbol
            processed = self._prepare_full_data(symbol, data_period)
            if processed is None or len(processed) == 0:
                logger.error(f"No data for {symbol}, skipping")
                continue

            # Build splitter
            splitter = WalkForwardSplitter(
                strategy=strategy,  # type: ignore[arg-type]
                initial_train_size=initial_train_days,
                test_size=test_window_days,
                step_size=test_step_days,
            )

            try:
                folds = splitter.split(len(processed))
            except ValueError as e:
                logger.error(f"Insufficient data for {symbol}: {e}")
                continue

            logger.info(
                f"  {symbol}: {len(processed)} rows, {len(folds)} folds "
                f"({strategy}, train={initial_train_days}, test={test_window_days}, step={test_step_days})"
            )

            for model_name in model_names:
                logger.info(f"  Model: {model_name}")
                try:
                    bt_result = self._run_single_backtest(
                        symbol=symbol,
                        model_name=model_name,
                        processed_data=processed,
                        folds=folds,
                        strategy=strategy,
                    )
                    results[symbol][model_name] = bt_result
                except Exception as e:
                    logger.error(f"  Backtest failed for {symbol}/{model_name}: {e}")

        return results

    def _prepare_full_data(
        self,
        symbol: str,
        period: str,
    ) -> Optional[pd.DataFrame]:
        """Load and preprocess full dataset for a symbol."""
        try:
            raw = self.collector.get_currency_data(symbol, period=period, interval="1d")
            if raw is None or raw.empty:
                return None

            cleaned = self.data_processor.clean_data(raw)
            with_indicators = self.data_processor.create_technical_indicators(cleaned)

            # CAPM features for stocks
            asset_type = classify_symbol(symbol)
            capm_config = self.config.get("capm", {})
            if asset_type == AssetType.STOCK and capm_config.get("enabled", False):
                market_index = capm_config.get("market_index", "^GSPC")
                market_raw = self.collector.get_currency_data(
                    market_index, period=period, interval="1d"
                )
                if market_raw is not None and not market_raw.empty:
                    with_indicators = self.data_processor.create_capm_features(
                        with_indicators,
                        market_raw,
                        risk_free_rate=0.04,
                        rolling_window=capm_config.get("rolling_window", 252),
                    )

            processed = self.data_processor.create_lagged_features(
                with_indicators, lags=[1, 2, 3]
            )
            return processed

        except Exception as e:
            logger.error(f"Data preparation failed for {symbol}: {e}")
            return None

    def _run_single_backtest(
        self,
        symbol: str,
        model_name: str,
        processed_data: pd.DataFrame,
        folds: List,
        strategy: str,
        target_column: str = "Close",
    ) -> BacktestResult:
        """Run backtest for a single (symbol, model) pair."""
        model_params = self.config.get("model_params", {})
        fold_results: List[FoldResult] = []
        all_actual: List[np.ndarray] = []
        all_predicted: List[np.ndarray] = []
        all_equity: List[np.ndarray] = []
        all_dates: List[pd.Timestamp] = []
        total_time = 0.0

        for fold in folds:
            fr = self._run_fold(
                model_name=model_name,
                model_params=model_params,
                processed_data=processed_data,
                fold=fold,
                target_column=target_column,
            )
            fold_results.append(fr)
            all_actual.append(fr.actual_prices)
            all_predicted.append(fr.predicted_prices)
            all_equity.append(fr.equity_curve)
            total_time += fr.training_time

            # Collect test dates
            test_slice = processed_data.iloc[fold.test_start : fold.test_end]
            all_dates.extend(test_slice.index.tolist())

        # Aggregate metrics
        pred_keys = fold_results[0].prediction_metrics.keys()
        avg_pred = {k: float(np.mean([f.prediction_metrics[k] for f in fold_results])) for k in pred_keys}
        std_pred = {k: float(np.std([f.prediction_metrics[k] for f in fold_results])) for k in pred_keys}

        fin_keys = fold_results[0].financial_metrics.keys()
        avg_fin = {k: float(np.mean([f.financial_metrics[k] for f in fold_results])) for k in fin_keys}

        # Overall equity curve: chain fold equity curves
        overall_equity = self._chain_equity_curves(fold_results)

        return BacktestResult(
            symbol=symbol,
            model_name=model_name,
            strategy=strategy,
            n_folds=len(fold_results),
            fold_results=fold_results,
            avg_prediction_metrics=avg_pred,
            avg_financial_metrics=avg_fin,
            std_prediction_metrics=std_pred,
            overall_equity_curve=overall_equity,
            overall_actual_prices=np.concatenate(all_actual),
            overall_predicted_prices=np.concatenate(all_predicted),
            overall_dates=all_dates,
            total_training_time=total_time,
            config_snapshot={
                "model_name": model_name,
                "model_params": model_params,
                "strategy": strategy,
            },
        )

    def _run_fold(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        processed_data: pd.DataFrame,
        fold: Any,
        target_column: str,
    ) -> FoldResult:
        """Execute a single fold: create fresh model, train, predict, evaluate."""
        # Split data
        train_data = processed_data.iloc[fold.train_start : fold.train_end]
        test_data = processed_data.iloc[fold.test_start : fold.test_end]

        feature_cols = [c for c in processed_data.columns if c != target_column]
        X_train = train_data[feature_cols]
        y_train = train_data[target_column]
        X_test = test_data[feature_cols]
        y_test = test_data[target_column]

        # Create fresh model (no data leakage)
        model = ModelFactory.create_model(model_name, **model_params)

        # Train
        start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start

        # Predict — model.predict returns pred_len values from end of input
        predictions = model.predict(X_test, horizon=len(y_test))

        # Align lengths (model may return fewer predictions)
        actual = y_test.values
        pred = np.asarray(predictions, dtype=float)
        min_len = min(len(actual), len(pred))
        actual = actual[:min_len]
        pred = pred[:min_len]

        # Prediction metrics
        pred_metrics = ModelComparer.compute_unified_metrics(actual, pred)

        # Financial metrics
        fin_result = FinancialMetrics.compute_all(actual, pred)
        fin_metrics = {
            "sharpe_ratio": fin_result.sharpe_ratio,
            "max_drawdown": fin_result.max_drawdown,
            "win_rate": fin_result.win_rate,
            "profit_factor": fin_result.profit_factor,
            "cumulative_return": fin_result.cumulative_return,
        }

        # Equity curve
        pred_returns = FinancialMetrics.compute_returns(pred)
        equity = FinancialMetrics.compute_equity_curve(actual, pred_returns)

        return FoldResult(
            fold_index=fold.fold_index,
            train_start_date=pd.Timestamp(train_data.index[0]),
            train_end_date=pd.Timestamp(train_data.index[-1]),
            test_start_date=pd.Timestamp(test_data.index[0]),
            test_end_date=pd.Timestamp(test_data.index[-1]),
            train_size=len(X_train),
            test_size=min_len,
            prediction_metrics=pred_metrics,
            financial_metrics=fin_metrics,
            actual_prices=actual,
            predicted_prices=pred,
            equity_curve=equity,
            training_time=training_time,
        )

    @staticmethod
    def _chain_equity_curves(fold_results: List[FoldResult]) -> np.ndarray:
        """Chain fold equity curves so each fold starts where previous ended."""
        if not fold_results:
            return np.array([10000.0])

        curves: List[np.ndarray] = []
        carry = 10000.0

        for fr in fold_results:
            eq = fr.equity_curve.copy()
            if len(eq) == 0:
                continue
            # Scale so the fold starts at carry value
            scale = carry / eq[0] if eq[0] != 0 else 1.0
            scaled = eq * scale
            curves.append(scaled)
            carry = scaled[-1]

        if not curves:
            return np.array([10000.0])

        return np.concatenate(curves)
