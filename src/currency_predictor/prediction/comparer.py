"""
多模型比較器

提供多個模型的訓練、預測與統一比較功能
"""

import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from .predictor import CurrencyPredictor, _clean_symbol
from ..models.factory import ModelFactory

logger = logging.getLogger(__name__)

# CLI 短名稱 → ModelFactory 完整名稱
MODEL_SHORTNAMES: Dict[str, str] = {
    'sklearn': 'patchtst_sklearn',
    'huggingface': 'patchtst_huggingface',
    'hf': 'patchtst_huggingface',
    'transformer': 'patchtst_huggingface',
    'lightning': 'patchtst_lightning',
}


def resolve_model_name(name: str) -> str:
    """將 CLI 短名稱解析為 ModelFactory 完整名稱。

    如果 name 已是完整名稱（存在於 ModelFactory），直接回傳；
    否則查詢 MODEL_SHORTNAMES。
    """
    available = ModelFactory.get_available_models()
    if name in available:
        return name
    resolved = MODEL_SHORTNAMES.get(name.lower())
    if resolved is not None:
        return resolved
    raise ValueError(
        f"未知的模型名稱: '{name}'。"
        f"可用名稱: {list(available.keys()) + list(MODEL_SHORTNAMES.keys())}"
    )


class ModelComparer:
    """多模型比較器

    為每個指定的模型建立獨立的 CurrencyPredictor，
    執行 collect → train → predict → evaluate 流程，
    最後彙整比較結果。
    """

    def __init__(
        self,
        model_names: List[str],
        config: Dict[str, Any],
        output_dir: str = "results",
    ):
        """
        Args:
            model_names: 模型名稱列表（支援短名稱如 'sklearn'）
            config: 應用設定字典（同 PredictionPipeline）
            output_dir: 輸出目錄
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 解析並去重
        self.model_names = list(dict.fromkeys(
            resolve_model_name(n) for n in model_names
        ))

        # 為每個模型建立獨立的 CurrencyPredictor
        self.predictors: Dict[str, CurrencyPredictor] = self._init_predictors()

        logger.info(f"ModelComparer 已初始化，模型: {self.model_names}")

    def _init_predictors(self) -> Dict[str, CurrencyPredictor]:
        """為每個模型建立 CurrencyPredictor 實例"""
        predictors = {}
        for name in self.model_names:
            try:
                predictors[name] = CurrencyPredictor(
                    model_name=name,
                    model_params=self.config.get('model_params', {}),
                    data_storage_path=self.config.get('data_storage_path', 'data'),
                    capm_config=self.config.get('capm', {}),
                )
            except Exception as e:
                logger.error(f"建立 {name} 預測器失敗: {e}")
        return predictors

    # ------------------------------------------------------------------
    # 靜態工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def direction_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
        """計算方向準確率（預測漲跌方向是否正確）。

        Args:
            actual: 實際值序列（至少 2 個值）
            predicted: 預測值序列（長度須與 actual 相同）

        Returns:
            0.0 ~ 1.0 的方向準確率；長度不足時回傳 0.0
        """
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        if len(actual) < 2 or len(predicted) < 2:
            return 0.0
        actual_dir = np.sign(np.diff(actual))
        pred_dir = np.sign(np.diff(predicted))
        min_len = min(len(actual_dir), len(pred_dir))
        if min_len == 0:
            return 0.0
        return float(np.mean(actual_dir[:min_len] == pred_dir[:min_len]))

    @staticmethod
    def compute_unified_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> Dict[str, float]:
        """計算統一的比較指標。

        Returns:
            包含 rmse, mae, mape, direction_accuracy 的字典
        """
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]

        errors = actual - predicted
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        # MAPE: 避免除以零
        nonzero_mask = actual != 0
        if nonzero_mask.any():
            mape = float(np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100)
        else:
            mape = float('inf')

        dir_acc = ModelComparer.direction_accuracy(actual, predicted)

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': dir_acc,
        }

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def compare(
        self,
        symbols: List[str],
        prediction_horizon: int = 7,
        force_update: bool = False,
    ) -> Dict[str, Any]:
        """執行多模型比較。

        對每個 symbol，各模型依序執行 collect → train → predict，
        再以統一指標比較。

        Args:
            symbols: 要預測的符號列表
            prediction_horizon: 預測天數
            force_update: 是否強制重新下載資料

        Returns:
            結構化的比較結果字典
        """
        data_config = self.config.get('data_collection', {})
        training_config = self.config.get('model_training', {})

        symbols_results: Dict[str, Any] = {}
        all_model_rmse: Dict[str, List[float]] = {n: [] for n in self.model_names}

        for symbol in symbols:
            logger.info(f"=== 比較 {symbol} ===")
            symbol_result: Dict[str, Any] = {'models': {}}

            # 收集資料（只需一次，但每個 predictor 都有自己的 storage）
            for name, predictor in self.predictors.items():
                predictor.collect_and_store_data(
                    [symbol],
                    period=data_config.get('period', '1y'),
                    interval=data_config.get('interval', '1d'),
                    force_update=force_update,
                )

            # 準備 actual（用第一個 predictor 取得 test 區間的真值）
            first_predictor = next(iter(self.predictors.values()))
            try:
                _, _, X_test, y_test = first_predictor.prepare_training_data(
                    symbol,
                    period=training_config.get('period', '1y'),
                    target_column=training_config.get('target_column', 'Close'),
                )
                actual_series = y_test
                symbol_result['actual'] = actual_series
            except Exception as e:
                logger.error(f"準備 {symbol} 測試資料失敗: {e}")
                symbols_results[symbol] = {'error': str(e)}
                continue

            # 各模型 train + predict
            for name, predictor in self.predictors.items():
                logger.info(f"  模型: {name}")
                model_result: Dict[str, Any] = {}

                try:
                    # 訓練
                    t0 = time.time()
                    train_result = predictor.train_model(
                        symbol=symbol,
                        period=training_config.get('period', '1y'),
                        target_column=training_config.get('target_column', 'Close'),
                        feature_columns=training_config.get('feature_columns'),
                        **training_config.get('train_params', {}),
                    )
                    training_time = time.time() - t0

                    model_result['training_time'] = training_time
                    model_result['training_completed'] = train_result.get('training_completed', False)

                    if not train_result.get('training_completed', False):
                        model_result['error'] = train_result.get('error', '訓練失敗')
                        symbol_result['models'][name] = model_result
                        continue

                    # 儲存模型
                    clean_sym = _clean_symbol(symbol)
                    model_path = self.output_dir / "models" / f"{clean_sym}_{name}.joblib"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    predictor.save_model(str(model_path))
                    model_result['model_path'] = str(model_path)

                    # 預測
                    pred_result = predictor.predict(
                        symbol=symbol,
                        horizon=prediction_horizon,
                        period=training_config.get('period', '1y'),
                    )

                    if pred_result.get('error'):
                        model_result['error'] = pred_result['error']
                        symbol_result['models'][name] = model_result
                        continue

                    predictions = np.asarray(pred_result['predictions'])
                    model_result['predictions'] = predictions
                    model_result['prediction_dates'] = pred_result.get('prediction_dates')
                    model_result['last_known_date'] = pred_result.get('last_known_date')
                    model_result['last_known_value'] = pred_result.get('last_known_value')

                    # 如果有 test metrics from training，記錄下來
                    model_result['train_metrics'] = train_result.get('train_metrics', {})
                    model_result['test_metrics'] = train_result.get('test_metrics', {})

                    # 統一指標（使用 test 資料）
                    # 由於 predict 產生的是未來預測，test_metrics 已在 train_model 中計算
                    # 我們使用 test_metrics 中的 rmse 來排名
                    test_rmse = train_result.get('test_metrics', {}).get('rmse')
                    if test_rmse is not None:
                        all_model_rmse[name].append(test_rmse)

                except Exception as e:
                    logger.error(f"  {name} 比較失敗: {e}")
                    model_result['error'] = str(e)

                symbol_result['models'][name] = model_result

            # 找出此 symbol 的最佳模型
            best_model = None
            best_rmse = float('inf')
            for mname, mresult in symbol_result['models'].items():
                rmse = mresult.get('test_metrics', {}).get('rmse')
                if rmse is not None and rmse < best_rmse:
                    best_rmse = rmse
                    best_model = mname
            symbol_result['best_model'] = best_model

            symbols_results[symbol] = symbol_result

        # 整體排名（依平均 RMSE）
        overall_ranking = []
        for name in self.model_names:
            rmse_list = all_model_rmse.get(name, [])
            if rmse_list:
                avg_rmse = float(np.mean(rmse_list))
                overall_ranking.append((name, avg_rmse))
        overall_ranking.sort(key=lambda x: x[1])

        return {
            'symbols_results': symbols_results,
            'overall_ranking': overall_ranking,
            'model_names': self.model_names,
            'prediction_horizon': prediction_horizon,
        }

    def train_only(
        self,
        symbols: List[str],
        force_update: bool = False,
    ) -> Dict[str, Any]:
        """只執行訓練（跳過預測），儲存所有模型

        Args:
            symbols: 符號列表
            force_update: 是否強制重新下載資料

        Returns:
            訓練結果字典
        """
        data_config = self.config.get('data_collection', {})
        training_config = self.config.get('model_training', {})

        symbols_results: Dict[str, Any] = {}

        for symbol in symbols:
            logger.info(f"=== Train-only: {symbol} ===")
            symbol_result: Dict[str, Any] = {'models': {}}

            # 收集資料
            for name, predictor in self.predictors.items():
                predictor.collect_and_store_data(
                    [symbol],
                    period=data_config.get('period', '1y'),
                    interval=data_config.get('interval', '1d'),
                    force_update=force_update,
                )

            # 訓練各模型
            for name, predictor in self.predictors.items():
                logger.info(f"  訓練模型: {name}")
                model_result: Dict[str, Any] = {}

                try:
                    t0 = time.time()
                    train_result = predictor.train_model(
                        symbol=symbol,
                        period=training_config.get('period', '1y'),
                        target_column=training_config.get('target_column', 'Close'),
                        feature_columns=training_config.get('feature_columns'),
                        **training_config.get('train_params', {}),
                    )
                    model_result['training_time'] = time.time() - t0
                    model_result['training_completed'] = train_result.get('training_completed', False)
                    model_result['train_metrics'] = train_result.get('train_metrics', {})
                    model_result['test_metrics'] = train_result.get('test_metrics', {})

                    if train_result.get('training_completed', False):
                        clean_sym = _clean_symbol(symbol)
                        model_path = self.output_dir / "models" / f"{clean_sym}_{name}.joblib"
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        predictor.save_model(str(model_path))
                        model_result['model_path'] = str(model_path)

                except Exception as e:
                    logger.error(f"  {name} 訓練失敗: {e}")
                    model_result['error'] = str(e)

                symbol_result['models'][name] = model_result

            symbols_results[symbol] = symbol_result

        return {
            'symbols_results': symbols_results,
            'model_names': self.model_names,
            'operation_type': 'train_only',
        }

    def predict_only(
        self,
        symbols: List[str],
        prediction_horizon: int = 7,
        model_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """載入既有模型並只執行預測

        Args:
            symbols: 符號列表
            prediction_horizon: 預測天數
            model_dir: 模型所在目錄（內含 models/ 子目錄）

        Returns:
            比較結果字典（與 compare() 相同格式）
        """
        training_config = self.config.get('model_training', {})
        model_base = Path(model_dir) / "models" if model_dir else self.output_dir / "models"

        symbols_results: Dict[str, Any] = {}
        all_model_rmse: Dict[str, List[float]] = {n: [] for n in self.model_names}

        for symbol in symbols:
            logger.info(f"=== Predict-only: {symbol} ===")
            symbol_result: Dict[str, Any] = {'models': {}}
            clean_sym = _clean_symbol(symbol)

            # 準備 actual 資料
            first_predictor = next(iter(self.predictors.values()))
            try:
                _, _, X_test, y_test = first_predictor.prepare_training_data(
                    symbol,
                    period=training_config.get('period', '1y'),
                    target_column=training_config.get('target_column', 'Close'),
                )
                symbol_result['actual'] = y_test
            except Exception as e:
                logger.error(f"準備 {symbol} 測試資料失敗: {e}")
                symbols_results[symbol] = {'error': str(e)}
                continue

            for name, predictor in self.predictors.items():
                logger.info(f"  載入模型: {name}")
                model_result: Dict[str, Any] = {}

                try:
                    model_path = model_base / f"{clean_sym}_{name}.joblib"
                    if not model_path.exists():
                        model_result['error'] = f"模型不存在: {model_path}"
                        symbol_result['models'][name] = model_result
                        continue

                    if not predictor.load_model(str(model_path)):
                        model_result['error'] = f"載入模型失敗: {model_path}"
                        symbol_result['models'][name] = model_result
                        continue

                    # 預測
                    pred_result = predictor.predict(
                        symbol=symbol,
                        horizon=prediction_horizon,
                        period=training_config.get('period', '1y'),
                    )

                    if pred_result.get('error'):
                        model_result['error'] = pred_result['error']
                    else:
                        model_result['predictions'] = np.asarray(pred_result['predictions'])
                        model_result['prediction_dates'] = pred_result.get('prediction_dates')
                        model_result['last_known_date'] = pred_result.get('last_known_date')
                        model_result['last_known_value'] = pred_result.get('last_known_value')

                except Exception as e:
                    logger.error(f"  {name} 預測失敗: {e}")
                    model_result['error'] = str(e)

                symbol_result['models'][name] = model_result

            symbols_results[symbol] = symbol_result

        return {
            'symbols_results': symbols_results,
            'model_names': self.model_names,
            'prediction_horizon': prediction_horizon,
            'operation_type': 'predict_only',
        }
