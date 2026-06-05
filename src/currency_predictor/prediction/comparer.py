"""
多模型比較器

提供多個模型的訓練、預測與統一比較功能
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from .predictor import CurrencyPredictor, _clean_symbol
from ..models.factory import ModelFactory

logger = logging.getLogger(__name__)

# CLI 短名稱 → ModelFactory 完整名稱
MODEL_SHORTNAMES: Dict[str, str] = {
    'naive': 'naive',
    'baseline': 'naive',
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

        Delegates to ``prediction.metrics.mda`` for the actual computation.

        Args:
            actual: 實際值序列（至少 2 個值）
            predicted: 預測值序列（長度須與 actual 相同）

        Returns:
            0.0 ~ 1.0 的方向準確率；長度不足時回傳 0.0
        """
        from .metrics import mda
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        if len(actual) < 2 or len(predicted) < 2:
            return 0.0
        return mda(actual, predicted)

    @staticmethod
    def compute_unified_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """計算統一的比較指標。

        Args:
            actual: 實際值序列
            predicted: 預測值序列
            y_train: 訓練集目標值（用於計算 MASE 的 naive 基準）

        Returns:
            包含 rmse, mae, mape, direction_accuracy（及 mase 如有 y_train）的字典
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

        metrics: Dict[str, float] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': dir_acc,
            'mda': dir_acc,
        }

        # MASE (requires training data for naive scaling factor)
        if y_train is not None:
            y_train_arr = np.asarray(y_train, dtype=float)
            if len(y_train_arr) > 1:
                from .metrics import mase as compute_mase
                metrics['mase'] = compute_mase(actual, predicted, y_train_arr)

        return metrics

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

        # 計算 effective test_days（至少 seq_len + pred_len，讓模型能在 test set 上 evaluate）
        configured_test_days = training_config.get('test_days')
        seq_len = self.config.get('model_params', {}).get('seq_len', 64)
        min_test = seq_len + prediction_horizon
        effective_test_days = configured_test_days or max(2 * prediction_horizon, min_test)

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
                    test_days=effective_test_days,
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
                        test_days=effective_test_days,
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

    def train(
        self,
        symbols: List[str],
        force_update: bool = False,
    ) -> Dict[str, Any]:
        """訓練所有模型（跳過預測），儲存模型檔案

        Args:
            symbols: 符號列表
            force_update: 是否強制重新下載資料

        Returns:
            訓練結果字典
        """
        data_config = self.config.get('data_collection', {})
        training_config = self.config.get('model_training', {})

        # train-only：用 config 或基於 seq_len 的預設
        seq_len = self.config.get('model_params', {}).get('seq_len', 64)
        pred_len = self.config.get('model_params', {}).get('pred_len', 15)
        min_test = seq_len + pred_len
        effective_test_days = training_config.get('test_days') or max(min_test, 30)

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
                        test_days=effective_test_days,
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
            'operation_type': 'train',
        }

    # Backward-compatible alias
    train_only = train

    def predict(
        self,
        symbols: List[str],
        prediction_horizon: int = 7,
        model_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """載入既有模型，預測並評估

        Args:
            symbols: 符號列表
            prediction_horizon: 預測天數
            model_dir: 模型所在目錄（內含 models/ 子目錄）

        Returns:
            比較結果字典（與 compare() 相同格式，含 metrics 和 ranking）
        """
        training_config = self.config.get('model_training', {})
        model_base = Path(model_dir) / "models" if model_dir else self.output_dir / "models"

        # 計算 effective test_days（至少 seq_len + pred_len）
        configured_test_days = training_config.get('test_days')
        seq_len = self.config.get('model_params', {}).get('seq_len', 64)
        min_test = seq_len + prediction_horizon
        effective_test_days = configured_test_days or max(2 * prediction_horizon, min_test)

        symbols_results: Dict[str, Any] = {}
        all_model_rmse: Dict[str, List[float]] = {n: [] for n in self.model_names}

        for symbol in symbols:
            logger.info(f"=== Predict: {symbol} ===")
            symbol_result: Dict[str, Any] = {'models': {}}
            clean_sym = _clean_symbol(symbol)

            # 準備 actual 資料（含 y_train 給 MASE）
            first_predictor = next(iter(self.predictors.values()))
            try:
                _, y_train, X_test, y_test = first_predictor.prepare_training_data(
                    symbol,
                    period=training_config.get('period', '1y'),
                    target_column=training_config.get('target_column', 'Close'),
                    test_days=effective_test_days,
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

                        # Evaluate on test set
                        test_metrics = predictor.model.evaluate_single_shot(
                            X_test, y_test, y_train=y_train
                        )
                        model_result['test_metrics'] = dict(test_metrics)

                        test_rmse = test_metrics.get('rmse')
                        if test_rmse is not None:
                            all_model_rmse[name].append(test_rmse)

                except Exception as e:
                    logger.error(f"  {name} 預測失敗: {e}")
                    model_result['error'] = str(e)

                symbol_result['models'][name] = model_result

            # Best model for this symbol
            best_model = None
            best_rmse = float('inf')
            for mname, mresult in symbol_result['models'].items():
                rmse = mresult.get('test_metrics', {}).get('rmse')
                if rmse is not None and rmse < best_rmse:
                    best_rmse = rmse
                    best_model = mname
            symbol_result['best_model'] = best_model

            symbols_results[symbol] = symbol_result

        # Overall ranking by avg RMSE
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
            'operation_type': 'predict',
        }

    # Backward-compatible alias
    predict_only = predict

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def save_predictions_csv(
        self, comparison_results: Dict[str, Any], output_dir: str | Path
    ) -> None:
        """儲存各 symbol 的預測值 CSV（wide format）"""
        output_dir = Path(output_dir)

        for symbol, sym_data in comparison_results.get('symbols_results', {}).items():
            actual = sym_data.get('actual')
            models = sym_data.get('models', {})

            # 收集有 predictions + prediction_dates 的模型
            model_preds: Dict[str, tuple] = {}
            for model_name, model_data in models.items():
                if 'error' in model_data:
                    continue
                preds = model_data.get('predictions')
                dates = model_data.get('prediction_dates')
                if preds is not None and dates is not None:
                    model_preds[model_name] = (dates, np.asarray(preds))

            if not model_preds:
                continue

            # 用第一個模型的 dates 作為共用 index
            first_model = next(iter(model_preds))
            dates = model_preds[first_model][0]
            n = len(dates)

            df_data: Dict[str, Any] = {'Date': dates}

            if actual is not None:
                actual_arr = np.asarray(actual)
                df_data['Actual'] = actual_arr[-n:] if len(actual_arr) >= n else actual_arr

            for model_name, (_, preds) in model_preds.items():
                df_data[model_name] = preds[:n]

            try:
                df = pd.DataFrame(df_data)
                clean_sym = _clean_symbol(symbol)
                csv_path = output_dir / f"predictions_{clean_sym}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Predictions CSV saved: {csv_path}")
            except Exception as e:
                logger.error(f"Failed to save predictions CSV for {symbol}: {e}")

    def save_metrics_csv(
        self, comparison_results: Dict[str, Any], output_dir: str | Path
    ) -> None:
        """儲存所有模型的評估指標 CSV（long format）"""
        output_dir = Path(output_dir)
        metric_keys = ['mse', 'mae', 'rmse', 'mape', 'mase', 'mda']
        rows: list[Dict[str, Any]] = []

        for symbol, sym_data in comparison_results.get('symbols_results', {}).items():
            for model_name, model_data in sym_data.get('models', {}).items():
                if model_data.get('error'):
                    continue
                for dataset, key in [('train', 'train_metrics'), ('test', 'test_metrics')]:
                    metrics = model_data.get(key, {})
                    if not metrics:
                        continue
                    row: Dict[str, Any] = {
                        'Symbol': symbol,
                        'Model': model_name,
                        'Dataset': dataset,
                    }
                    for mk in metric_keys:
                        row[mk.upper()] = metrics.get(mk)
                    rows.append(row)

        if not rows:
            return

        try:
            df = pd.DataFrame(rows)
            csv_path = output_dir / "metrics.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Metrics CSV saved: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics CSV: {e}")

    def save_forecast_json(
        self, comparison_results: Dict[str, Any], output_dir: str | Path
    ) -> None:
        """儲存 out-of-sample forecast JSON（供日後驗證用）。

        每個 symbol 產生一個 forecast_{SYMBOL}.json，包含各模型的
        prediction_dates、predictions、last_known_date 等資訊。
        """
        output_dir = Path(output_dir)

        for symbol, sym_data in comparison_results.get('symbols_results', {}).items():
            models_data = sym_data.get('models', {})
            forecast: Dict[str, Any] = {
                'symbol': symbol,
                'forecast_generated_at': datetime.now().isoformat(timespec='seconds'),
                'models': {},
            }

            for model_name, model_data in models_data.items():
                if 'error' in model_data:
                    continue
                preds = model_data.get('predictions')
                dates = model_data.get('prediction_dates')
                if preds is None or dates is None:
                    continue

                last_known_date = model_data.get('last_known_date')
                date_strs = []
                for d in dates:
                    if hasattr(d, 'strftime'):
                        date_strs.append(d.strftime('%Y-%m-%d'))
                    else:
                        date_strs.append(str(d))

                last_known_str = ''
                if last_known_date is not None:
                    if hasattr(last_known_date, 'strftime'):
                        last_known_str = last_known_date.strftime('%Y-%m-%d')
                    else:
                        last_known_str = str(last_known_date)

                forecast['models'][model_name] = {
                    'last_known_date': last_known_str,
                    'last_known_value': float(model_data.get('last_known_value', 0)),
                    'prediction_dates': date_strs,
                    'predictions': [float(p) for p in np.asarray(preds).flatten()],
                }

            if not forecast['models']:
                continue

            # prediction_horizon = max number of prediction dates across models
            max_horizon = max(
                len(m['prediction_dates']) for m in forecast['models'].values()
            )
            forecast['prediction_horizon'] = max_horizon

            try:
                clean_sym = _clean_symbol(symbol)
                json_path = output_dir / f"forecast_{clean_sym}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(forecast, f, indent=2, ensure_ascii=False)
                logger.info(f"Forecast JSON saved: {json_path}")
            except Exception as e:
                logger.error(f"Failed to save forecast JSON for {symbol}: {e}")
