"""
金融預測執行器

提供貨幣匯率及股票預測的核心功能
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from datetime import timedelta
from pathlib import Path

from ..data.collectors import YahooFinanceCollector
from ..data.storage import DataStorage
from ..models.factory import ModelFactory
from ..models.patchtst.config import TrainingConfig
from ..data_processor import DataProcessor
from ..utils.asset_type import AssetType, classify_symbol

logger = logging.getLogger(__name__)


def _clean_symbol(symbol: str) -> str:
    """清理符號名稱，用於檔案儲存路徑。

    貨幣對（結尾 =X）去除 '=X'，股票 ticker 保持不變。
    """
    if YahooFinanceCollector.is_currency_pair(symbol):
        return symbol.replace("=X", "")
    return symbol


class CurrencyPredictor:
    """
    金融預測執行器

    整合資料收集、處理和模型預測功能，支援貨幣對及股票 ticker
    """

    def __init__(
        self,
        model_name: str = "patchtst_sklearn",  # 改為使用工廠模型名稱
        model_params: Optional[Dict[str, Any]] = None,
        data_storage_path: str = "data",
        capm_config: Optional[Dict[str, Any]] = None,
        target_transform: str = "price",
    ):
        """
        初始化預測器

        Args:
            model_name: 模型名稱 ('patchtst_sklearn' 或 'patchtst_transformer')
            model_params: 模型參數
            data_storage_path: 資料儲存路徑
            capm_config: CAPM 配置 (enabled, market_index, risk_free_rate_symbol, rolling_window)
            target_transform: 目標轉換 "price"（直接預測價格）或 "log_return"
                （預測對數報酬，predict() 輸出時還原為價格）
        """
        self.model_name = model_name
        self.model_params = model_params or {}
        self.capm_config = capm_config or {}
        self.target_transform = target_transform
        # 訓練時記住目標欄位，供 log_return 還原使用
        self._target_column = "Close"

        # 初始化組件
        self.data_collector = YahooFinanceCollector()
        self.data_storage = DataStorage(base_dir=data_storage_path)
        self.data_processor = DataProcessor()

        # 初始化模型
        self.model = self._create_model()

        logger.info(f"貨幣預測器已初始化，使用模型: {model_name}")

    def _create_model(self):
        """創建指定的模型"""
        try:
            return ModelFactory.create_model(self.model_name, **self.model_params)
        except Exception as e:
            logger.error(f"創建模型失敗: {str(e)}")
            # 回退到 sklearn 版本
            logger.info("回退到 sklearn 版本的 PatchTST")
            return ModelFactory.create_model("patchtst_sklearn", **self.model_params)

    def _validate_cached_data(
        self,
        symbol: str,
        existing_data: "pd.DataFrame",
        n_samples: int = 3,
    ) -> bool:
        """抽樣驗證快取資料的正確性。

        從已有資料隨機抽取 n_samples 個日期，從 Yahoo Finance
        重新下載驗證，若 Close 值差異超過容忍範圍則視為資料失效。

        Args:
            symbol: 金融符號
            existing_data: 快取的 DataFrame
            n_samples: 抽樣數量

        Returns:
            True 表示資料有效，False 表示需要重新下載
        """
        if len(existing_data) < 10:
            return True  # 資料太少，不做抽樣驗證

        if "Close" not in existing_data.columns:
            return True

        rng = np.random.default_rng(seed=42)
        indices = rng.choice(
            len(existing_data), size=min(n_samples, len(existing_data)), replace=False
        )
        sample_dates = existing_data.index[indices]

        for date in sample_dates:
            try:
                spot_data = self.data_collector.get_spot_check_data(symbol, date)
                if spot_data is None or spot_data.empty:
                    continue  # 網路問題，跳過此 sample

                # 找到最近的匹配日期
                date_normalized = pd.Timestamp(date).normalize()
                if spot_data.index.tz is not None:
                    spot_data.index = spot_data.index.tz_localize(None)

                matching = spot_data[spot_data.index.normalize() == date_normalized]
                if matching.empty:
                    continue

                cached_close = float(existing_data.loc[date, "Close"])
                spot_close = float(matching.iloc[0]["Close"])

                # 允許 0.01 的絕對誤差
                if abs(cached_close - spot_close) > 0.01:
                    logger.warning(
                        f"資料驗證失敗: {symbol} {date} "
                        f"cached={cached_close:.4f} vs fresh={spot_close:.4f}"
                    )
                    return False

            except Exception as e:
                logger.debug(f"Spot check skipped for {date}: {e}")
                continue

        logger.info(
            f"{symbol} 快取資料抽樣驗證通過 ({min(n_samples, len(existing_data))} samples)"
        )
        return True

    def collect_and_store_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        force_update: bool = False,
    ) -> Dict[str, bool]:
        """
        收集並儲存貨幣資料

        Args:
            symbols: 貨幣對符號列表
            period: 資料期間
            interval: 資料間隔
            force_update: 是否強制更新資料

        Returns:
            各貨幣對的收集結果
        """
        results = {}

        for symbol in symbols:
            try:
                logger.info(f"開始收集 {symbol} 資料")

                # 檢查是否已有資料且不需要強制更新
                if not force_update:
                    existing_data = self.data_storage.load_raw_data(
                        _clean_symbol(symbol), period
                    )
                    if existing_data is not None:
                        # 抽樣驗證快取資料的正確性
                        if self._validate_cached_data(symbol, existing_data):
                            logger.info(f"{symbol} 資料已存在且驗證通過，跳過收集")
                            results[symbol] = True
                            continue
                        else:
                            logger.warning(f"{symbol} 快取資料驗證失敗，重新下載")

                # 收集資料
                data = self.data_collector.get_currency_data(symbol, period, interval)

                if data is not None and not data.empty:
                    # 儲存資料
                    clean_symbol = _clean_symbol(symbol)
                    success = self.data_storage.save_raw_data(
                        data, clean_symbol, period
                    )
                    results[symbol] = success

                    if success:
                        logger.info(f"[OK] {symbol} 資料收集並儲存成功")
                    else:
                        logger.error(f"[FAIL] {symbol} 資料儲存失敗")
                else:
                    logger.error(f"[FAIL] 無法收集 {symbol} 資料")
                    results[symbol] = False

            except Exception as e:
                logger.error(f"收集 {symbol} 資料時發生錯誤: {str(e)}")
                results[symbol] = False

        return results

    def prepare_training_data(
        self,
        symbol: str,
        period: str = "1y",
        target_column: str = "Close",
        feature_columns: Optional[List[str]] = None,
        test_days: Optional[int] = None,
    ) -> tuple:
        """
        準備訓練資料

        Args:
            symbol: 貨幣對符號
            period: 資料期間
            target_column: 目標欄位
            feature_columns: 特徵欄位列表
            test_days: Test set 固定天數（None = 30）

        Returns:
            (X_train, y_train, X_test, y_test)
        """
        # 載入原始資料
        clean_symbol = _clean_symbol(symbol)
        raw_data = self.data_storage.load_raw_data(clean_symbol, period)

        if raw_data is None or raw_data.empty:
            raise ValueError(f"找不到 {symbol} 的資料")

        logger.info(f"載入 {symbol} 資料，共 {len(raw_data)} 筆")

        # 資料清理和處理
        cleaned_data = self.data_processor.clean_data(raw_data)

        # 特徵工程 - 創建技術指標
        data_with_indicators = self.data_processor.create_technical_indicators(
            cleaned_data
        )

        # CAPM features (stocks only, when enabled)
        asset_type = classify_symbol(symbol)
        if asset_type == AssetType.STOCK and self.capm_config.get("enabled", False):
            market_data = self._get_market_index_data(period=period)
            rf_rate = self._get_risk_free_rate()
            rolling_w = self.capm_config.get("rolling_window", 252)
            data_with_indicators = self.data_processor.create_capm_features(
                data_with_indicators, market_data, rf_rate, rolling_w
            )

        # 創建滯後特徵 (使用較短的滯後期)
        processed_data = self.data_processor.create_lagged_features(
            data_with_indicators, lags=[1, 2, 3]
        )

        # 準備特徵和目標
        if feature_columns is None:
            # 使用所有欄位except目標欄位作為特徵
            feature_columns = [
                col for col in processed_data.columns if col != target_column
            ]

        self._target_column = target_column
        X = processed_data[feature_columns]
        if self.target_transform == "log_return":
            # 目標改為對數報酬；首列為 NaN，連同 X 一併丟棄
            y = self.data_processor.to_log_returns(processed_data[target_column])
            valid = y.notna()
            X = X[valid]
            y = y[valid]
            processed_data = processed_data[valid]
        else:
            y = processed_data[target_column]

        # 時間分割（固定天數）
        if test_days is None:
            test_days = 30
        # 安全上限：不超過資料總量的 50%
        max_test = len(processed_data) // 2
        effective_test_days = min(test_days, max_test)

        # 事前驗證：窗口模型需要 test 視窗 >= seq_len + pred_len 才能評估
        seq_len = self.model_params.get(
            "seq_len", self.model_params.get("context_length", 64)
        )
        pred_len = self.model_params.get(
            "pred_len", self.model_params.get("prediction_length", 15)
        )
        required = seq_len + pred_len
        if effective_test_days < required:
            raise ValueError(
                f"測試視窗太小無法評估：effective_test_days={effective_test_days} "
                f"< seq_len + pred_len = {seq_len} + {pred_len} = {required}。"
                f"請將 test_days 設為 >= {required}"
                + (
                    f"（注意資料僅 {len(processed_data)} 筆，上限 {max_test}，"
                    "可能需更長歷史）"
                    if effective_test_days == max_test
                    else ""
                )
            )

        split_idx = len(processed_data) - effective_test_days

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        logger.info(f"訓練資料: {len(X_train)} 筆，測試資料: {len(X_test)} 筆")

        return X_train, y_train, X_test, y_test

    def train_model(
        self,
        symbol: str,
        period: str = "1y",
        target_column: str = "Close",
        feature_columns: Optional[List[str]] = None,
        test_days: Optional[int] = None,
        **train_kwargs,
    ) -> Dict[str, Any]:
        """
        訓練模型

        Args:
            symbol: 貨幣對符號
            period: 資料期間
            target_column: 目標欄位
            feature_columns: 特徵欄位
            test_days: Test set 固定天數
            **train_kwargs: 訓練參數

        Returns:
            訓練結果字典
        """
        try:
            # 準備訓練資料
            X_train, y_train, X_test, y_test = self.prepare_training_data(
                symbol,
                period,
                target_column,
                feature_columns,
                test_days=test_days,
            )

            # 訓練模型
            logger.info(f"開始訓練 {self.model_name} 模型")

            # 將 validation_split 轉為 TrainingConfig，委託模型處理分割邏輯
            if "validation_split" in train_kwargs:
                val_split = train_kwargs.pop("validation_split")
                tc = TrainingConfig(validation_split=val_split)
                self.model.fit(X_train, y_train, training_config=tc, **train_kwargs)
            else:
                self.model.fit(X_train, y_train, **train_kwargs)

            # 評估模型
            train_metrics = self._evaluate_model(
                X_train, y_train, "訓練", y_train=y_train
            )
            test_metrics = self._evaluate_model(X_test, y_test, "測試", y_train=y_train)

            training_results = {
                "symbol": symbol,
                "model_name": self.model_name,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "training_completed": True,
            }

            logger.info("模型訓練完成")
            return training_results

        except Exception as e:
            logger.error(f"模型訓練失敗: {str(e)}")
            return {
                "symbol": symbol,
                "model_name": self.model_name,
                "error": str(e),
                "training_completed": False,
            }

    def _evaluate_model(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        dataset_name: str,
        y_train: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """評估模型性能。

        優先使用 rolling origin evaluation（更穩健），
        test set 不夠大時 fallback 到 single-shot。
        """
        try:
            seq_len = self.model_params.get(
                "seq_len", self.model_params.get("context_length", 64)
            )
            pred_len = self.model_params.get(
                "pred_len", self.model_params.get("prediction_length", 15)
            )

            if len(X) >= seq_len + pred_len:
                rolling = self.model.evaluate_rolling(
                    X,
                    y_true,
                    seq_len,
                    pred_len,
                    y_train=y_train,
                )
                metrics = dict(rolling["aggregate"])
                metrics["per_horizon"] = rolling.get("per_horizon", {})
                n_origins = int(metrics.get("n_origins", 0))
                logger.info(
                    f"{dataset_name}集評估 (rolling, {n_origins} origins): "
                    f"RMSE={metrics.get('rmse', 0):.6f}"
                )
            else:
                metrics = self.model.evaluate_single_shot(
                    X,
                    y_true,
                    y_train=y_train,
                )
                logger.info(
                    f"{dataset_name}集評估 (single-shot): "
                    f"MSE={metrics.get('mse', 0):.6f}"
                )

            return dict(metrics)
        except Exception as e:
            logger.error(f"模型評估失敗: {str(e)}")
            return {}

    def predict(
        self,
        symbol: str,
        horizon: int = 7,
        period: str = "1y",
        return_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        """
        進行預測

        Args:
            symbol: 貨幣對符號
            horizon: 預測時間範圍
            period: 用於預測的歷史資料期間
            return_uncertainty: 是否返回不確定性

        Returns:
            預測結果字典
        """
        try:
            if not self.model.is_fitted:
                raise ValueError("模型尚未訓練，請先調用 train_model()")

            # 載入最新資料
            clean_symbol = _clean_symbol(symbol)
            raw_data = self.data_storage.load_raw_data(clean_symbol, period)

            if raw_data is None or raw_data.empty:
                raise ValueError(f"找不到 {symbol} 的資料")

            # 資料處理
            cleaned_data = self.data_processor.clean_data(raw_data)
            data_with_indicators = self.data_processor.create_technical_indicators(
                cleaned_data
            )

            # CAPM features (stocks only, when enabled)
            asset_type = classify_symbol(symbol)
            if asset_type == AssetType.STOCK and self.capm_config.get("enabled", False):
                market_data = self._get_market_index_data(period=period)
                rf_rate = self._get_risk_free_rate()
                rolling_w = self.capm_config.get("rolling_window", 252)
                data_with_indicators = self.data_processor.create_capm_features(
                    data_with_indicators, market_data, rf_rate, rolling_w
                )

            processed_data = self.data_processor.create_lagged_features(
                data_with_indicators, lags=[1, 2, 3]
            )

            # 進行預測
            if return_uncertainty and hasattr(self.model, "predict_with_uncertainty"):
                prediction_result = self.model.predict_with_uncertainty(
                    processed_data, horizon
                )
            else:
                predictions = self.model.predict(processed_data, horizon)
                prediction_result = {"predictions": predictions}

            # log_return 模式：模型輸出為對數報酬，還原為價格
            if self.target_transform == "log_return":
                last_price = float(processed_data[self._target_column].iloc[-1])
                for key in ("predictions", "upper_bound", "lower_bound"):
                    if key in prediction_result:
                        prediction_result[key] = self.data_processor.from_log_returns(
                            last_price, prediction_result[key]
                        )
                # std 在報酬空間無法線性對應價格，移除以免誤導
                prediction_result.pop("std", None)

            # 生成預測日期
            last_date = processed_data.index[-1]
            prediction_dates = [
                last_date + timedelta(days=i + 1)
                for i in range(len(prediction_result["predictions"]))
            ]

            result = {
                "symbol": symbol,
                "prediction_dates": prediction_dates,
                "last_known_date": last_date,
                "last_known_value": processed_data["Close"].iloc[-1],
                **prediction_result,
            }

            logger.info(f"成功預測 {symbol}，預測 {len(prediction_dates)} 個時間點")
            return result

        except Exception as e:
            logger.error(f"預測 {symbol} 時發生錯誤: {str(e)}")
            return {"symbol": symbol, "error": str(e), "success": False}

    def save_model(self, filepath: str) -> bool:
        """
        儲存模型

        Args:
            filepath: 儲存路徑

        Returns:
            是否儲存成功
        """
        try:
            # 確保目錄存在
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # 儲存模型
            success: bool = self.model.save_model(filepath)

            if success:
                logger.info(f"模型已儲存至: {filepath}")

            return success

        except Exception as e:
            logger.error(f"儲存模型失敗: {str(e)}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        載入模型

        Args:
            filepath: 模型檔案路徑

        Returns:
            是否載入成功
        """
        try:
            success: bool = self.model.load_model(filepath)

            if success:
                logger.info(f"模型已從 {filepath} 載入")

            return success

        except Exception as e:
            logger.error(f"載入模型失敗: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        取得模型資訊

        Returns:
            模型資訊字典
        """
        base_info = {
            "predictor_model": self.model_name,
            "model_params": self.model_params,
            "data_storage_path": str(self.data_storage.base_dir),
        }

        if hasattr(self.model, "get_model_info"):
            model_info = self.model.get_model_info()
            base_info.update(model_info)

        return base_info

    # ------------------------------------------------------------------
    # CAPM helpers
    # ------------------------------------------------------------------

    def _get_market_index_data(self, period: str = "2y") -> pd.DataFrame:
        """Fetch market index data (e.g. S&P 500) for CAPM calculations.

        Args:
            period: Data period; should match the asset's training/prediction
                period so CAPM features cover the full history (otherwise the
                early portion of a long history has all-NaN market features).
        """
        market_symbol = self.capm_config.get("market_index", "^GSPC")
        logger.info(f"Fetching market index data: {market_symbol} (period={period})")
        data = self.data_collector.get_currency_data(market_symbol, period=period)
        if data is None or data.empty:
            raise ValueError(f"無法取得市場指數 {market_symbol} 的資料")
        return data

    def _get_risk_free_rate(self) -> float:
        """Get current risk-free rate from Treasury data."""
        rf_symbol = self.capm_config.get("risk_free_rate_symbol", "^IRX")
        try:
            rf_data = self.data_collector.get_currency_data(rf_symbol, period="1mo")
            if rf_data is not None and not rf_data.empty:
                rate = float(rf_data["Close"].iloc[-1]) / 100  # Convert percentage
                logger.info(f"Risk-free rate from {rf_symbol}: {rate:.4f}")
                return rate
        except Exception as e:
            logger.warning(f"無法取得無風險利率 ({rf_symbol}): {e}")
        # Default fallback
        logger.info("Using default risk-free rate: 0.04")
        return 0.04
