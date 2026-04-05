"""
測試 PatchTST 配置

測試 PatchTSTConfig 和 TrainingConfig
"""

import pytest

from currency_predictor.models.patchtst.config import PatchTSTConfig, TrainingConfig


class TestPatchTSTConfig:
    """測試 PatchTSTConfig"""

    def test_default_values(self):
        config = PatchTSTConfig()
        assert config.context_length == 64
        assert config.prediction_length == 7
        assert config.patch_length == 8
        assert config.patch_stride == 4
        assert config.d_model == 64
        assert config.n_heads == 4

    def test_custom_values(self):
        config = PatchTSTConfig(
            context_length=128,
            prediction_length=14,
            patch_length=16,
            patch_stride=8,
        )
        assert config.context_length == 128
        assert config.prediction_length == 14

    def test_sklearn_aliases(self):
        config = PatchTSTConfig(context_length=32, prediction_length=5, patch_length=8, patch_stride=4)
        assert config.seq_len == 32
        assert config.pred_len == 5
        assert config.patch_len == 8
        assert config.stride == 4

    def test_num_patches(self):
        config = PatchTSTConfig(context_length=64, patch_length=8, patch_stride=4)
        expected = (64 - 8) // 4 + 1
        assert config.num_patches == expected

    def test_from_sklearn_params(self):
        config = PatchTSTConfig.from_sklearn_params(
            seq_len=30,
            pred_len=7,
            patch_len=10,
            stride=5,
        )
        assert config.context_length == 30
        assert config.prediction_length == 7
        assert config.patch_length == 10
        assert config.patch_stride == 5

    def test_from_sklearn_params_partial(self):
        config = PatchTSTConfig.from_sklearn_params(seq_len=100)
        assert config.context_length == 100
        assert config.prediction_length == 7  # default

    def test_from_sklearn_params_extra_kwargs(self):
        config = PatchTSTConfig.from_sklearn_params(
            seq_len=50,
            d_model=128,
        )
        assert config.context_length == 50
        assert config.d_model == 128

    def test_validate_passes(self):
        config = PatchTSTConfig()
        config.validate()  # should not raise

    def test_validate_bad_context_length(self):
        config = PatchTSTConfig(context_length=0)
        with pytest.raises(ValueError, match="context_length"):
            config.validate()

    def test_validate_bad_prediction_length(self):
        config = PatchTSTConfig(prediction_length=-1)
        with pytest.raises(ValueError, match="prediction_length"):
            config.validate()

    def test_validate_bad_patch_length(self):
        config = PatchTSTConfig(patch_length=0)
        with pytest.raises(ValueError, match="patch_length"):
            config.validate()

    def test_validate_bad_patch_stride(self):
        config = PatchTSTConfig(patch_stride=0)
        with pytest.raises(ValueError, match="patch_stride"):
            config.validate()

    def test_validate_patch_gt_context(self):
        config = PatchTSTConfig(context_length=8, patch_length=16)
        with pytest.raises(ValueError, match="patch_length.*不能大於"):
            config.validate()

    def test_validate_bad_d_model(self):
        config = PatchTSTConfig(d_model=0)
        with pytest.raises(ValueError, match="d_model"):
            config.validate()

    def test_validate_bad_n_heads(self):
        config = PatchTSTConfig(n_heads=0)
        with pytest.raises(ValueError, match="n_heads"):
            config.validate()

    def test_validate_d_model_not_divisible(self):
        config = PatchTSTConfig(d_model=65, n_heads=4)
        with pytest.raises(ValueError, match="整除"):
            config.validate()

    def test_validate_bad_dropout(self):
        config = PatchTSTConfig(dropout=1.5)
        with pytest.raises(ValueError, match="dropout"):
            config.validate()

    def test_to_dict(self):
        config = PatchTSTConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d['context_length'] == 64
        assert d['d_model'] == 64
        assert 'random_state' in d


class TestTrainingConfig:
    """測試 TrainingConfig"""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.validation_split == 0.2
        assert config.num_epochs == 50
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4

    def test_custom_values(self):
        config = TrainingConfig(
            num_epochs=100,
            batch_size=64,
            learning_rate=1e-3,
        )
        assert config.num_epochs == 100
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3

    def test_early_stopping_defaults(self):
        config = TrainingConfig()
        assert config.early_stopping_patience == 10
        assert config.early_stopping_threshold == 0.0001

    def test_lr_scheduler_default(self):
        config = TrainingConfig()
        assert config.lr_scheduler == 'cosine'
