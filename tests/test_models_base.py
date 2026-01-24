"""
測試基礎模型模組

測試 BaseModel 和相關基礎類別
"""

import pytest
import pandas as pd
import numpy as np

from src.currency_predictor.models.base import (
    BaseModel,
    TimeSeriesModel,
    SklearnBasedModel,
    TransformerBasedModel,
    ModelType
)


class ConcreteModel(BaseModel):
    """用於測試的具體模型實現"""

    def fit(self, X, y, validation_data=None):
        self.is_fitted = True
        return self

    def predict(self, X, horizon=1):
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")
        return np.zeros(len(X))

    def predict_with_uncertainty(self, X, horizon=1, confidence_level=0.95):
        predictions = self.predict(X, horizon)
        return {
            'predictions': predictions,
            'std': np.ones_like(predictions) * 0.1,
            'lower_bound': predictions - 0.2,
            'upper_bound': predictions + 0.2
        }


class TestModelType:
    """測試 ModelType 枚舉"""

    def test_model_types_exist(self):
        """測試模型類型存在"""
        assert ModelType.SKLEARN_BASED
        assert ModelType.TRANSFORMER_BASED
        assert ModelType.DEEP_LEARNING

    def test_model_type_values(self):
        """測試模型類型值"""
        assert ModelType.SKLEARN_BASED.value == "sklearn_based"
        assert ModelType.TRANSFORMER_BASED.value == "transformer_based"
        assert ModelType.DEEP_LEARNING.value == "deep_learning"


class TestBaseModel:
    """測試 BaseModel 基礎類別"""

    def test_init(self):
        """測試初始化"""
        model = ConcreteModel(model_name="TestModel")
        assert model.model_name == "TestModel"
        assert model.is_fitted is False
        assert isinstance(model.model_params, dict)

    def test_fit_and_predict(self):
        """測試訓練和預測"""
        model = ConcreteModel()

        # 創建測試資料
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y = pd.Series([10, 20, 30, 40, 50])

        # 訓練
        model.fit(X, y)
        assert model.is_fitted is True

        # 預測
        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_predict_without_fit_raises_error(self):
        """測試未訓練前預測會報錯"""
        model = ConcreteModel()
        X = pd.DataFrame({'feature1': [1, 2, 3]})

        with pytest.raises(ValueError, match="模型尚未訓練"):
            model.predict(X)

    def test_get_model_info(self):
        """測試取得模型資訊"""
        model = ConcreteModel(model_name="TestModel")
        info = model.get_model_info()

        assert 'model_name' in info
        assert 'model_type' in info
        assert 'is_fitted' in info
        assert 'model_params' in info
        assert info['model_name'] == 'TestModel'

    def test_predict_with_uncertainty(self):
        """測試帶不確定性的預測"""
        model = ConcreteModel()
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y = pd.Series([10, 20, 30, 40, 50])

        model.fit(X, y)
        result = model.predict_with_uncertainty(X)

        assert 'predictions' in result
        assert 'std' in result
        assert 'lower_bound' in result
        assert 'upper_bound' in result


class TestTimeSeriesModel:
    """測試 TimeSeriesModel 類別"""

    def test_init(self):
        """測試時間序列模型初始化"""
        # TimeSeriesModel 是抽象類，需要創建一個具體實現
        class ConcreteTimeSeriesModel(TimeSeriesModel):
            def fit(self, X, y, validation_data=None):
                self.is_fitted = True
                return self

            def predict(self, X, horizon=1):
                return np.zeros(len(X))

            def predict_with_uncertainty(self, X, horizon=1, confidence_level=0.95):
                return {'predictions': np.zeros(len(X))}

        model = ConcreteTimeSeriesModel()
        assert model.sequence_length is None
        assert isinstance(model.feature_columns, list)

    def test_prepare_sequences(self):
        """測試準備時間序列資料"""
        class ConcreteTimeSeriesModel(TimeSeriesModel):
            def fit(self, X, y, validation_data=None):
                return self

            def predict(self, X, horizon=1):
                return np.zeros(len(X))

            def predict_with_uncertainty(self, X, horizon=1, confidence_level=0.95):
                return {'predictions': np.zeros(len(X))}

        model = ConcreteTimeSeriesModel()

        # 創建時間序列資料
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Close': np.random.randn(100),
            'Volume': np.random.randint(1000, 5000, 100)
        }, index=dates)

        # 準備序列
        X, y = model.prepare_sequences(data, sequence_length=10, target_column='Close')

        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 10  # sequence_length
        assert X.shape[2] == 2   # number of features

    def test_validate_input_shape(self):
        """測試驗證輸入資料形狀"""
        class ConcreteTimeSeriesModel(TimeSeriesModel):
            def fit(self, X, y, validation_data=None):
                self.is_fitted = True
                self.sequence_length = 10
                return self

            def predict(self, X, horizon=1):
                return np.zeros(len(X))

            def predict_with_uncertainty(self, X, horizon=1, confidence_level=0.95):
                return {'predictions': np.zeros(len(X))}

        model = ConcreteTimeSeriesModel()

        # 未訓練時應返回 False
        X = pd.DataFrame({'feature': range(20)})
        assert model.validate_input_shape(X) is False

        # 訓練後
        y = pd.Series(range(20))
        model.fit(X, y)

        # 足夠的資料應返回 True
        assert model.validate_input_shape(X) is True

        # 不足的資料應返回 False
        X_short = pd.DataFrame({'feature': range(5)})
        assert model.validate_input_shape(X_short) is False


class TestSklearnBasedModel:
    """測試 SklearnBasedModel 類別"""

    def test_init(self):
        """測試初始化"""
        # 需要創建具體實現
        class ConcreteSklearnModel(SklearnBasedModel):
            def fit(self, X, y, validation_data=None):
                return self

            def predict(self, X, horizon=1):
                return np.zeros(len(X))

            def predict_with_uncertainty(self, X, horizon=1, confidence_level=0.95):
                return {'predictions': np.zeros(len(X))}

        model = ConcreteSklearnModel()
        assert model.model_type == ModelType.SKLEARN_BASED
        assert model.scaler is None
        assert model.model is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
