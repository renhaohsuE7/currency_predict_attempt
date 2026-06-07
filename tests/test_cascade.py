import pytest
from currency_predictor.config.settings import CascadeConfig


def test_cascade_config_defaults():
    cfg = CascadeConfig()
    assert cfg.enabled is False
    assert cfg.crossfit_folds == 5
    assert cfg.stage2_backend == "patchtst_sklearn"


def test_cascade_config_custom():
    cfg = CascadeConfig(enabled=True, crossfit_folds=3, stage2_backend="patchtst_huggingface")
    assert cfg.enabled is True
    assert cfg.crossfit_folds == 3
