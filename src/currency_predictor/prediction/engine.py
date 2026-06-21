"""
預測引擎

用已訓練的模型對指定符號做未來預測。模型由 ModelTrainer 持有(本引擎持參照,
故 load_model 後自動使用新模型);資料前處理委派 DataManager。
"""

import logging
from datetime import timedelta
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PredictionEngine:
    """產生預測結果(含日期、最後已知值,選用不確定性)。"""

    def __init__(self, trainer, data_manager):
        self._trainer = trainer
        self._data = data_manager

    @property
    def model(self):
        return self._trainer.model

    def predict(
        self,
        symbol: str,
        horizon: int = 7,
        period: str = "1y",
        return_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        """進行預測,回傳結果 dict(失敗回 {'symbol', 'error', 'success': False})。"""
        try:
            if not self.model.is_fitted:
                raise ValueError("模型尚未訓練，請先調用 train_model()")

            processed_data = self._data.process(symbol, period)

            if return_uncertainty and hasattr(self.model, 'predict_with_uncertainty'):
                prediction_result = self.model.predict_with_uncertainty(processed_data, horizon)
            else:
                predictions = self.model.predict(processed_data, horizon)
                prediction_result = {'predictions': predictions}

            last_date = processed_data.index[-1]
            prediction_dates = [
                last_date + timedelta(days=i + 1)
                for i in range(len(prediction_result['predictions']))
            ]

            result = {
                'symbol': symbol,
                'prediction_dates': prediction_dates,
                'last_known_date': last_date,
                'last_known_value': processed_data['Close'].iloc[-1],
                **prediction_result,
            }

            logger.info(f"成功預測 {symbol}，預測 {len(prediction_dates)} 個時間點")
            return result

        except Exception as e:
            logger.error(f"預測 {symbol} 時發生錯誤: {str(e)}")
            return {'symbol': symbol, 'error': str(e), 'success': False}
