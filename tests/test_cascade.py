import numpy as np
import pandas as pd
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
