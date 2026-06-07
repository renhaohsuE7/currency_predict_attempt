import importlib.util

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from currency_predictor.config.settings import CascadeConfig
from currency_predictor.prediction.cascade import CascadePredictor


def test_cascade_config_defaults():
    cfg = CascadeConfig()
    assert cfg.enabled is False
    assert cfg.crossfit_folds == 5
    assert cfg.stage2_backend == "patchtst_sklearn"


def test_cascade_config_custom():
    cfg = CascadeConfig(
        enabled=True, crossfit_folds=3, stage2_backend="patchtst_huggingface"
    )
    assert cfg.enabled is True
    assert cfg.crossfit_folds == 3


def _ohlcv(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.integers(1000, 10000, n),
        },
        index=idx,
    )


def test_cascade_sklearn_end_to_end():
    df = _ohlcv()
    cfg = {
        "model_name": "patchtst_sklearn",
        "model_params": {"seq_len": 32, "pred_len": 5, "patch_len": 8, "stride": 4},
        "model_training": {"target_transform": "log_return", "test_days": 60},
        "cascade": {
            "enabled": True,
            "crossfit_folds": 3,
            "stage2_backend": "patchtst_sklearn",
        },
    }
    cp = CascadePredictor(cfg)
    with patch.object(cp.predictor.data_storage, "load_raw_data", return_value=df):
        cp.fit("TEST", period="2y")
        out = cp.predict("TEST", horizon=5)
    assert {"price", "vol", "dir"} <= set(out)
    assert len(out["price"]) == 5
    assert np.all(np.isfinite(out["price"]))
    assert 0.0 <= out["dir"] <= 1.0
    assert out["vol"] >= 0.0


def _backend_available(name: str) -> bool:
    if name == "patchtst_sklearn":
        return True
    if name == "patchtst_huggingface":
        return importlib.util.find_spec("transformers") is not None
    if name == "patchtst_lightning":
        return importlib.util.find_spec("pytorch_lightning") is not None
    return False


@pytest.mark.parametrize(
    "backend",
    ["patchtst_sklearn", "patchtst_huggingface", "patchtst_lightning"],
)
def test_cascade_all_backends_end_to_end(backend):
    if not _backend_available(backend):
        pytest.skip(f"{backend} optional dependency not installed")
    df = _ohlcv(n=400, seed=1)
    cfg = {
        "model_name": backend,
        "model_params": {
            "seq_len": 32,
            "pred_len": 5,
            "patch_len": 8,
            "stride": 4,
            "d_model": 32,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
        },
        "model_training": {"target_transform": "log_return", "test_days": 60},
        "cascade": {"enabled": True, "crossfit_folds": 3, "stage2_backend": backend},
    }
    cp = CascadePredictor(cfg)
    with patch.object(cp.predictor.data_storage, "load_raw_data", return_value=df):
        cp.fit("TEST", period="2y")
        out = cp.predict("TEST", horizon=5)
    assert {"price", "vol", "dir"} <= set(out)
    assert len(out["price"]) == 5
    assert np.all(np.isfinite(out["price"]))
