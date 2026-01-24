"""
測試 PatchTST 模型

測試 PatchTST 時間序列預測模型的所有功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.currency_predictor.models.patchtst import PatchTST


@pytest.fixture
def sample_training_data():
    """創建樣本訓練資料"""
    np.random.seed(42)
    n_samples = 200

    # 創建時間序列特徵
    features = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })

    # 創建目標變數（與特徵相關）
    target = pd.Series(
        features['feature1'] * 0.5 +
        features['feature2'] * 0.3 +
        np.random.randn(n_samples) * 0.1
    )

    return features, target


class TestPatchTST:
    """測試 PatchTST 模型"""

    def test_init_default_params(self):
        """測試使用默認參數初始化"""
        model = PatchTST()

        assert model.model_name == "PatchTST"
        assert model.seq_len == 168
        assert model.pred_len == 24
        assert model.patch_len == 12
        assert model.stride == 6
        assert model.is_fitted is False

    def test_init_custom_params(self):
        """測試使用自定義參數初始化"""
        model = PatchTST(
            seq_len=100,
            pred_len=10,
            patch_len=20,
            stride=10,
            n_estimators=50,
            max_depth=5
        )

        assert model.seq_len == 100
        assert model.pred_len == 10
        assert model.patch_len == 20
        assert model.stride == 10
        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_fit(self, sample_training_data):
        """測試模型訓練"""
        X, y = sample_training_data

        model = PatchTST(seq_len=50, pred_len=5, patch_len=10, stride=5)
        model.fit(X, y)

        assert model.is_fitted is True
        assert model.scaler is not None
        assert model.target_scaler is not None

    def test_fit_with_validation_data(self, sample_training_data):
        """測試使用驗證資料訓練"""
        X, y = sample_training_data

        # 分割訓練和驗證資料
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        model = PatchTST(seq_len=50, pred_len=5, patch_len=10, stride=5)
        model.fit(X_train, y_train, validation_data=(X_val, y_val))

        assert model.is_fitted is True

    def test_predict_raises_error_when_not_fitted(self, sample_training_data):
        """測試未訓練時預測會報錯"""
        X, _ = sample_training_data

        model = PatchTST(seq_len=50)

        with pytest.raises(ValueError, match="模型尚未訓練"):
            model.predict(X)

    def test_predict_raises_error_with_insufficient_data(self, sample_training_data):
        """測試資料不足時預測會報錯"""
        X, y = sample_training_data

        model = PatchTST(seq_len=50, pred_len=5, patch_len=10, stride=5)
        model.fit(X, y)

        # 使用少於 seq_len 的資料
        X_short = X[:30]

        with pytest.raises(ValueError, match="輸入資料長度"):
            model.predict(X_short)

    def test_get_model_info(self):
        """測試取得模型資訊"""
        model = PatchTST(seq_len=100, pred_len=10)
        info = model.get_model_info()

        assert info['model_name'] == 'PatchTST'
        assert info['model_type'] == 'sklearn_based'
        assert 'seq_len' in info['model_params']
        assert 'pred_len' in info['model_params']
        assert info['model_params']['seq_len'] == 100
        assert info['model_params']['pred_len'] == 10

    def test_create_patches(self, sample_training_data):
        """測試 patch 創建"""
        X, y = sample_training_data

        model = PatchTST(seq_len=50, pred_len=5, patch_len=10, stride=5)

        # 取一個序列
        sequence = X[:50].values

        # 創建 patches
        patches = model._create_patches(sequence)

        # 驗證 patch 形狀
        assert patches.ndim == 3
        assert patches.shape[1] == 10  # patch_len
        assert patches.shape[2] == X.shape[1]  # n_features

    def test_extract_patch_features(self, sample_training_data):
        """測試 patch 特徵提取"""
        X, y = sample_training_data

        model = PatchTST(seq_len=50, pred_len=5, patch_len=10, stride=5)

        # 創建 patches
        sequence = X[:50].values
        patches = model._create_patches(sequence)

        # 提取特徵
        features = model._extract_patch_features(patches)

        # 驗證特徵形狀
        assert features.ndim == 1
        assert len(features) > 0

    def test_model_reproducibility(self, sample_training_data):
        """測試模型的可重現性"""
        X, y = sample_training_data

        # 使用相同的隨機種子訓練兩個模型
        model1 = PatchTST(seq_len=50, pred_len=5, random_state=42)
        model1.fit(X, y)

        model2 = PatchTST(seq_len=50, pred_len=5, random_state=42)
        model2.fit(X, y)

        # 預測結果應該相同
        X_test = X[-60:]
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_save_and_load_model(self, sample_training_data):
        """測試模型儲存和載入"""
        X, y = sample_training_data

        # 訓練模型
        model = PatchTST(seq_len=50, pred_len=5, random_state=42)
        model.fit(X, y)

        # 儲存模型
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            success = model.save_model(str(model_path))
            assert success is True
            assert model_path.exists()

            # 載入模型（這裡只測試檔案存在，實際載入功能需要在子類實現）
            # 注意：save_model 在基類中只是一個接口，具體實現在子類


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
