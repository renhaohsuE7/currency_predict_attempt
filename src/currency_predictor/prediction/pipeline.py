"""
預測流程管道

提供完整的端到端預測流程自動化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

from .predictor import CurrencyPredictor
from ..utils import setup_logging

logger = logging.getLogger(__name__)


def convert_numpy_to_native(obj: Any) -> Any:
    """
    遞迴轉換物件中的 numpy 類型為 Python 原生類型

    Args:
        obj: 任意物件

    Returns:
        轉換後的物件
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_native(item) for item in obj]
    return obj


class PredictionPipeline:
    """
    預測流程管道
    
    自動化完整的預測工作流程：
    1. 資料收集
    2. 資料處理
    3. 模型訓練
    4. 預測執行
    5. 結果儲存
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "results"
    ):
        """
        初始化預測管道
        
        Args:
            config: 配置字典
            output_dir: 輸出目錄
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設置日誌
        setup_logging(self.config.get('log_level', 'INFO'))
        
        # 初始化預測器
        self.predictor = CurrencyPredictor(
            model_name=config.get('model_name', 'PatchTST'),
            model_params=config.get('model_params', {}),
            data_storage_path=config.get('data_storage_path', 'data'),
            capm_config=config.get('capm', {}),
            target_transform=config.get('model_training', {}).get('target_transform', 'price'),
        )
        
        # 流程狀態
        self.pipeline_status = {
            'data_collection': False,
            'model_training': False,
            'prediction': False,
            'results_saved': False
        }
        
        logger.info("預測流程管道已初始化")
    
    def run_full_pipeline(
        self,
        symbols: List[str],
        prediction_horizon: int = 7,
        save_results: bool = True,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        執行完整預測流程
        
        Args:
            symbols: 貨幣對符號列表
            prediction_horizon: 預測時間範圍
            save_results: 是否儲存結果
            force_retrain: 是否強制重新訓練
            
        Returns:
            完整結果字典
        """
        pipeline_results = {
            'symbols': symbols,
            'prediction_horizon': prediction_horizon,
            'start_time': datetime.now().isoformat(),
            'results': {},
            'errors': [],
            'pipeline_status': {}
        }
        
        try:
            logger.info(f"開始執行完整預測流程，貨幣對: {symbols}")
            
            # 1. 資料收集階段
            logger.info("=== 階段 1: 資料收集 ===")
            data_results = self._collect_data_phase(symbols, force_update=force_retrain)
            pipeline_results['data_collection'] = data_results
            self.pipeline_status['data_collection'] = all(data_results.values())
            
            if not self.pipeline_status['data_collection']:
                failed_symbols = [s for s, success in data_results.items() if not success]
                logger.warning(f"部分符號資料收集失敗: {failed_symbols}")
            
            # 2. 模型訓練階段
            logger.info("=== 階段 2: 模型訓練 ===")
            training_results = self._training_phase(symbols, force_retrain)
            pipeline_results['training'] = training_results
            
            successful_training = [
                result for result in training_results 
                if result.get('training_completed', False)
            ]
            self.pipeline_status['model_training'] = len(successful_training) > 0
            
            # 3. 預測執行階段
            logger.info("=== 階段 3: 預測執行 ===")
            prediction_results = self._prediction_phase(symbols, prediction_horizon)
            pipeline_results['predictions'] = prediction_results
            
            successful_predictions = [
                result for result in prediction_results 
                if not result.get('error')
            ]
            self.pipeline_status['prediction'] = len(successful_predictions) > 0
            
            # 4. 結果儲存階段
            if save_results:
                logger.info("=== 階段 4: 結果儲存 ===")
                save_results_status = self._save_results_phase(pipeline_results)
                self.pipeline_status['results_saved'] = save_results_status
            
            # 更新最終狀態
            pipeline_results['end_time'] = datetime.now().isoformat()
            pipeline_results['pipeline_status'] = self.pipeline_status
            pipeline_results['success'] = all(self.pipeline_status.values())
            
            logger.info("完整預測流程執行完成")
            self._log_pipeline_summary(pipeline_results)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"預測流程執行失敗: {str(e)}")
            pipeline_results['error'] = str(e)
            pipeline_results['success'] = False
            return pipeline_results
    
    def _collect_data_phase(
        self, symbols: List[str], force_update: bool = False
    ) -> Dict[str, bool]:
        """資料收集階段"""
        data_config = self.config.get('data_collection', {})

        return self.predictor.collect_and_store_data(
            symbols=symbols,
            period=data_config.get('period', '1y'),
            interval=data_config.get('interval', '1d'),
            force_update=force_update or data_config.get('force_update', False),
        )
    
    def _training_phase(
        self, 
        symbols: List[str], 
        force_retrain: bool = False
    ) -> List[Dict[str, Any]]:
        """模型訓練階段"""
        training_config = self.config.get('model_training', {})
        prediction_horizon = self.config.get('prediction_horizon', 7)
        seq_len = self.config.get('model_params', {}).get('seq_len', 64)
        configured_test_days = training_config.get('test_days')
        min_test = seq_len + prediction_horizon
        effective_test_days = configured_test_days or max(2 * prediction_horizon, min_test)
        training_results = []

        for symbol in symbols:
            try:
                # 檢查是否已有訓練好的模型
                model_path = self.output_dir / f"models/{symbol}_{self.config['model_name']}.joblib"

                if model_path.exists() and not force_retrain:
                    logger.info(f"{symbol} 模型已存在，載入現有模型")
                    if self.predictor.load_model(str(model_path)):
                        training_results.append({
                            'symbol': symbol,
                            'model_name': self.config['model_name'],
                            'training_completed': True,
                            'loaded_existing': True
                        })
                        continue

                # 訓練新模型
                logger.info(f"開始訓練 {symbol} 模型")
                result = self.predictor.train_model(
                    symbol=symbol,
                    period=training_config.get('period', '1y'),
                    target_column=training_config.get('target_column', 'Close'),
                    feature_columns=training_config.get('feature_columns'),
                    test_days=effective_test_days,
                    **training_config.get('train_params', {})
                )
                
                # 儲存模型
                if result.get('training_completed', False):
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    # fail-loud:save_model 失敗時不可謊報 model_saved=True
                    saved = self.predictor.save_model(str(model_path))
                    result['model_saved'] = bool(saved)
                    if saved:
                        result['model_path'] = str(model_path)
                    else:
                        result['model_error'] = f"模型儲存失敗: {model_path}"
                        logger.error(f"模型儲存失敗: {model_path}")
                
                training_results.append(result)
                
            except Exception as e:
                logger.error(f"訓練 {symbol} 模型時發生錯誤: {str(e)}")
                training_results.append({
                    'symbol': symbol,
                    'error': str(e),
                    'training_completed': False
                })
        
        return training_results
    
    def _prediction_phase(
        self, 
        symbols: List[str], 
        prediction_horizon: int
    ) -> List[Dict[str, Any]]:
        """預測執行階段"""
        prediction_config = self.config.get('prediction', {})
        prediction_results = []
        
        for symbol in symbols:
            try:
                logger.info(f"開始預測 {symbol}")
                
                result = self.predictor.predict(
                    symbol=symbol,
                    horizon=prediction_horizon,
                    period=prediction_config.get('period', '1y'),
                    return_uncertainty=prediction_config.get('return_uncertainty', True)
                )
                
                prediction_results.append(result)
                
            except Exception as e:
                logger.error(f"預測 {symbol} 時發生錯誤: {str(e)}")
                prediction_results.append({
                    'symbol': symbol,
                    'error': str(e),
                    'success': False
                })
        
        return prediction_results
    
    def _save_results_phase(self, pipeline_results: Dict[str, Any]) -> bool:
        """結果儲存階段"""
        try:
            # 轉換 numpy 類型為 Python 原生類型
            serializable_results = convert_numpy_to_native(pipeline_results)

            # 儲存完整結果（run 目錄已含時間戳，檔名不需要）
            results_file = self.output_dir / "pipeline_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2, default=str)

            # 儲存預測結果為CSV
            self._save_predictions_csv(pipeline_results.get('predictions', []))

            # 儲存 forecast JSON（供日後驗證）
            self._save_forecast_json(pipeline_results.get('predictions', []))

            # 產生報告
            self._generate_report(pipeline_results)

            logger.info(f"結果已儲存至: {results_file}")
            return True

        except Exception as e:
            logger.error(f"儲存結果失敗: {str(e)}")
            return False

    def _save_predictions_csv(self, predictions: List[Dict[str, Any]]):
        """儲存預測結果為CSV格式"""
        for prediction in predictions:
            if prediction.get('error'):
                continue

            try:
                symbol = prediction['symbol']
                pred_df = pd.DataFrame({
                    'Date': prediction['prediction_dates'],
                    'Predicted_Close': prediction['predictions']
                })

                if 'uncertainty' in prediction:
                    pred_df['Uncertainty'] = prediction['uncertainty']

                csv_file = self.output_dir / f"{symbol}_predictions.csv"
                pred_df.to_csv(csv_file, index=False)

            except Exception as e:
                logger.error(f"儲存 {symbol} 預測CSV失敗: {str(e)}")

    def _save_forecast_json(self, predictions: List[Dict[str, Any]]):
        """儲存 out-of-sample forecast JSON（供日後驗證用）。"""
        for prediction in predictions:
            if prediction.get('error'):
                continue

            try:
                symbol = prediction['symbol']
                pred_dates = prediction.get('prediction_dates', [])
                preds = prediction.get('predictions', [])
                if not pred_dates or preds is None:
                    continue

                date_strs = []
                for d in pred_dates:
                    if hasattr(d, 'strftime'):
                        date_strs.append(d.strftime('%Y-%m-%d'))
                    else:
                        date_strs.append(str(d))

                last_known_date = prediction.get('last_known_date')
                last_known_str = ''
                if last_known_date is not None:
                    if hasattr(last_known_date, 'strftime'):
                        last_known_str = last_known_date.strftime('%Y-%m-%d')
                    else:
                        last_known_str = str(last_known_date)

                model_name = self.config.get('model_name', 'patchtst_sklearn')
                forecast = {
                    'symbol': symbol,
                    'forecast_generated_at': datetime.now().isoformat(timespec='seconds'),
                    'prediction_horizon': len(date_strs),
                    'models': {
                        model_name: {
                            'last_known_date': last_known_str,
                            'last_known_value': float(prediction.get('last_known_value', 0)),
                            'prediction_dates': date_strs,
                            'predictions': [float(p) for p in np.asarray(preds).flatten()],
                        }
                    },
                }

                clean_sym = symbol.replace('=', '').replace('/', '_')
                json_path = self.output_dir / f"forecast_{clean_sym}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(forecast, f, indent=2, ensure_ascii=False)
                logger.info(f"Forecast JSON saved: {json_path}")

            except Exception as e:
                logger.error(f"儲存 {symbol} forecast JSON 失敗: {str(e)}")

    def _generate_report(self, pipeline_results: Dict[str, Any]):
        """產生預測報告"""
        try:
            report_lines = []
            report_lines.append("# 貨幣預測流程報告")
            start_time = pipeline_results.get('start_time', 'N/A')
            end_time = pipeline_results.get('end_time', datetime.now().isoformat())
            report_lines.append(f"執行時間: {start_time} - {end_time}")
            report_lines.append("")
            
            # 資料收集摘要
            report_lines.append("## 資料收集結果")
            data_results = pipeline_results.get('data_collection', {})
            for symbol, success in data_results.items():
                status = "[OK] 成功" if success else "[FAIL] 失敗"
                report_lines.append(f"- {symbol}: {status}")
            report_lines.append("")
            
            # 訓練摘要
            report_lines.append("## 模型訓練結果")
            training_results = pipeline_results.get('training', [])
            for result in training_results:
                symbol = result['symbol']
                if result.get('training_completed', False):
                    metrics = result.get('test_metrics', {})
                    rmse = metrics.get('rmse', 0)
                    report_lines.append(f"- {symbol}: [OK] 訓練成功 (RMSE: {rmse:.6f})")
                else:
                    report_lines.append(f"- {symbol}: [FAIL] 訓練失敗")
            report_lines.append("")
            
            # 預測摘要
            report_lines.append("## 預測結果")
            predictions = pipeline_results.get('predictions', [])
            for prediction in predictions:
                symbol = prediction['symbol']
                if not prediction.get('error'):
                    last_value = prediction.get('last_known_value', 0)
                    predictions_array = prediction.get('predictions', [])
                    first_pred = predictions_array[0] if predictions_array is not None and len(predictions_array) > 0 else 0
                    change = ((first_pred - last_value) / last_value * 100) if last_value != 0 else 0
                    report_lines.append(f"- {symbol}: [OK] 預測完成 (預期變化: {change:+.2f}%)")
                else:
                    report_lines.append(f"- {symbol}: [FAIL] 預測失敗")
            
            # 儲存報告
            report_file = self.output_dir / "prediction_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"預測報告已產生: {report_file}")
            
        except Exception as e:
            logger.error(f"產生報告失敗: {str(e)}")
    
    def _log_pipeline_summary(self, results: Dict[str, Any]):
        """記錄流程摘要"""
        logger.info("=== 預測流程摘要 ===")
        logger.info(f"處理貨幣對數量: {len(results['symbols'])}")
        logger.info(f"資料收集: {'[OK]' if results['pipeline_status']['data_collection'] else '[FAIL]'}")
        logger.info(f"模型訓練: {'[OK]' if results['pipeline_status']['model_training'] else '[FAIL]'}")
        logger.info(f"預測執行: {'[OK]' if results['pipeline_status']['prediction'] else '[FAIL]'}")
        logger.info(f"結果儲存: {'[OK]' if results['pipeline_status'].get('results_saved', False) else '[FAIL]'}")
        logger.info(f"整體成功: {'[OK]' if results['success'] else '[FAIL]'}")
    
    def run_train(
        self,
        symbols: List[str],
        force_retrain: bool = True,
    ) -> Dict[str, Any]:
        """資料收集 + 模型訓練（跳過預測）

        Args:
            symbols: 貨幣對符號列表
            force_retrain: 是否強制重新訓練

        Returns:
            訓練結果字典
        """
        results: Dict[str, Any] = {
            'symbols': symbols,
            'start_time': datetime.now().isoformat(),
            'operation_type': 'train_only',
            'pipeline_status': {
                'data_collection': False,
                'model_training': False,
                'prediction': True,      # 不執行，視為通過
                'results_saved': False,
            }
        }

        try:
            # 1. 資料收集
            logger.info("=== Train-only: 資料收集 ===")
            data_results = self._collect_data_phase(symbols, force_update=force_retrain)
            results['data_collection'] = data_results
            results['pipeline_status']['data_collection'] = all(data_results.values())

            # 2. 訓練
            logger.info("=== Train-only: 模型訓練 ===")
            training_results = self._training_phase(symbols, force_retrain)
            results['training'] = training_results
            results['pipeline_status']['model_training'] = any(
                r.get('training_completed') for r in training_results
            )

            # 儲存結果 JSON
            serializable = convert_numpy_to_native(results)
            results_file = self.output_dir / "pipeline_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
            results['pipeline_status']['results_saved'] = True

            results['end_time'] = datetime.now().isoformat()
            results['success'] = all(results['pipeline_status'].values())

            logger.info(f"Train-only 完成，結果儲存至: {results_file}")
            return results

        except Exception as e:
            logger.error(f"Train-only 執行失敗: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

    def run_predict(
        self,
        symbols: List[str],
        prediction_horizon: int = 7,
        model_paths: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """載入既有模型並預測（跳過訓練）

        Args:
            symbols: 貨幣對符號列表
            prediction_horizon: 預測時間範圍
            model_paths: 各 symbol 的模型路徑 {symbol: path}

        Returns:
            預測結果字典
        """
        results: Dict[str, Any] = {
            'symbols': symbols,
            'prediction_horizon': prediction_horizon,
            'start_time': datetime.now().isoformat(),
            'operation_type': 'predict_only',
            'pipeline_status': {
                'data_collection': True,   # 不執行，視為通過
                'model_training': True,    # 不執行，視為通過
                'prediction': False,
                'results_saved': False,
            }
        }

        try:
            # 1. 載入模型
            logger.info("=== Predict-only: 載入模型 ===")
            loaded = {}
            for symbol in symbols:
                path = (model_paths or {}).get(symbol)
                if path:
                    success = self.predictor.load_model(str(path))
                    loaded[symbol] = success
                    if success:
                        logger.info(f"已載入 {symbol} 模型: {path}")
                    else:
                        logger.error(f"無法載入 {symbol} 模型: {path}")
                else:
                    # 檢查 output_dir 內是否有模型
                    default_path = (
                        self.output_dir
                        / f"models/{symbol}_{self.config['model_name']}.joblib"
                    )
                    if default_path.exists():
                        loaded[symbol] = self.predictor.load_model(str(default_path))
                    else:
                        logger.error(f"找不到 {symbol} 的模型檔案")
                        loaded[symbol] = False

            results['model_loading'] = loaded

            # 2. 預測
            logger.info("=== Predict-only: 預測執行 ===")
            prediction_results = self._prediction_phase(symbols, prediction_horizon)
            results['predictions'] = prediction_results
            results['pipeline_status']['prediction'] = any(
                not p.get('error') for p in prediction_results
            )

            # 3. 儲存
            self._save_predictions_csv(prediction_results)
            self._generate_report(results)

            serializable = convert_numpy_to_native(results)
            results_file = self.output_dir / "pipeline_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
            results['pipeline_status']['results_saved'] = True

            results['end_time'] = datetime.now().isoformat()
            results['success'] = all(results['pipeline_status'].values())

            logger.info(f"Predict-only 完成，結果儲存至: {results_file}")
            return results

        except Exception as e:
            logger.error(f"Predict-only 執行失敗: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

    # Backward-compatible aliases
    run_train_only = run_train
    run_predict_only = run_predict

    def run_batch_prediction(
        self,
        symbols: List[str],
        model_paths: Dict[str, str],
        prediction_horizon: int = 7
    ) -> List[Dict[str, Any]]:
        """
        批次預測（使用已訓練的模型）
        
        Args:
            symbols: 貨幣對符號列表
            model_paths: 各貨幣對的模型路徑
            prediction_horizon: 預測時間範圍
            
        Returns:
            預測結果列表
        """
        results = []
        
        for symbol in symbols:
            try:
                # 載入對應的模型
                if symbol in model_paths:
                    success = self.predictor.load_model(model_paths[symbol])
                    if not success:
                        results.append({
                            'symbol': symbol,
                            'error': f'無法載入模型: {model_paths[symbol]}',
                            'success': False
                        })
                        continue
                
                # 進行預測
                result = self.predictor.predict(
                    symbol=symbol,
                    horizon=prediction_horizon
                )
                results.append(result)
                
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    @classmethod
    def from_config_file(cls, config_path: str, **kwargs):
        """從配置檔案創建流程"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls(config, **kwargs)