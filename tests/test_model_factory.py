"""
測試模型工廠

測試 ModelFactory 類別和 create_patchtst_model 便利函數
"""

import pytest

from currency_predictor.models.factory import (
    ModelFactory,
    create_patchtst_model,
    PatchTST,
    PatchTSTSklearn,
    HAS_TRANSFORMERS,
    HAS_LIGHTNING,
)
from currency_predictor.models.base import BaseModel, ModelType


class TestModelFactory:
    """測試 ModelFactory"""

    def test_get_available_models_returns_dict(self):
        models = ModelFactory.get_available_models()
        assert isinstance(models, dict)

    def test_sklearn_always_available(self):
        models = ModelFactory.get_available_models()
        assert 'patchtst_sklearn' in models
        assert models['patchtst_sklearn']['available'] is True

    def test_available_models_have_required_keys(self):
        models = ModelFactory.get_available_models()
        required_keys = {'class', 'type', 'description', 'available', 'implementation'}
        for name, info in models.items():
            assert required_keys.issubset(info.keys()), f"{name} missing keys"

    def test_create_sklearn_model(self):
        model = ModelFactory.create_model('patchtst_sklearn')
        assert isinstance(model, BaseModel)
        assert model.model_type == ModelType.SKLEARN_BASED

    def test_create_sklearn_model_with_params(self):
        model = ModelFactory.create_model(
            'patchtst_sklearn',
            seq_len=30,
            pred_len=5,
            patch_len=10,
            stride=5,
        )
        assert isinstance(model, PatchTSTSklearn)

    def test_create_unknown_model_raises(self):
        with pytest.raises(ValueError, match="未知的模型名稱"):
            ModelFactory.create_model('nonexistent_model')

    def test_create_unavailable_model_raises(self):
        models = ModelFactory.get_available_models()
        unavailable = [
            name for name, info in models.items()
            if not info['available']
        ]
        if unavailable:
            with pytest.raises(ValueError, match="不可用"):
                ModelFactory.create_model(unavailable[0])

    def test_get_recommended_model_speed(self):
        name = ModelFactory.get_recommended_model(prefer_accuracy=False)
        assert name == 'patchtst_sklearn'

    def test_get_recommended_model_accuracy(self):
        name = ModelFactory.get_recommended_model(prefer_accuracy=True)
        assert isinstance(name, str)
        assert name in ModelFactory.get_available_models()

    def test_get_model_by_implementation_sklearn(self):
        name = ModelFactory.get_model_by_implementation('sklearn')
        assert name == 'patchtst_sklearn'

    def test_get_model_by_implementation_unknown(self):
        result = ModelFactory.get_model_by_implementation('unknown_impl')
        assert result is None

    def test_print_model_info(self, capsys):
        ModelFactory.print_model_info()
        captured = capsys.readouterr()
        assert '可用的模型' in captured.out
        assert 'patchtst_sklearn' in captured.out

    def test_backward_compat_alias(self):
        """PatchTST should be an alias for PatchTSTSklearn"""
        assert PatchTST is PatchTSTSklearn


class TestCreatePatchtstModel:
    """測試 create_patchtst_model 便利函數"""

    def test_auto_select(self):
        model = create_patchtst_model()
        assert isinstance(model, BaseModel)

    def test_force_sklearn(self):
        model = create_patchtst_model(use_transformer=False)
        assert isinstance(model, PatchTSTSklearn)

    def test_implementation_sklearn(self):
        model = create_patchtst_model(implementation='sklearn')
        assert isinstance(model, PatchTSTSklearn)

    def test_implementation_fallback(self):
        """不可用的 implementation 應 fallback 到 sklearn"""
        model = create_patchtst_model(implementation='nonexistent')
        assert isinstance(model, PatchTSTSklearn)

    def test_with_kwargs(self):
        model = create_patchtst_model(
            use_transformer=False,
            seq_len=20,
            pred_len=3,
        )
        assert isinstance(model, PatchTSTSklearn)
