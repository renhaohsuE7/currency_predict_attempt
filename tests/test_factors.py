import numpy as np
import pandas as pd
import pytest

from currency_predictor.prediction.factors import (
    realized_volatility_target,
    direction_target,
)


def test_realized_volatility_target_known_values():
    close = pd.Series([100.0 * (1.01**i) for i in range(20)])
    vol = realized_volatility_target(close, horizon=5)
    assert vol.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert vol.iloc[-1] != vol.iloc[-1]  # NaN


def test_realized_volatility_matches_manual():
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.standard_normal(30)))
    vol = realized_volatility_target(close, horizon=4)
    r = np.log(close.astype(float)).diff().to_numpy()
    assert vol.iloc[3] == pytest.approx(np.std(r[4:8]))


def test_direction_target_up_down():
    close = pd.Series([100, 101, 102, 103, 104, 105], dtype=float)
    d = direction_target(close, horizon=2)
    assert d.iloc[0] == 1.0
    down = pd.Series([105, 104, 103, 102, 101, 100], dtype=float)
    assert direction_target(down, horizon=2).iloc[0] == 0.0


def test_direction_target_tail_is_nan():
    close = pd.Series(np.arange(10), dtype=float)
    d = direction_target(close, horizon=3)
    assert d.iloc[-1] != d.iloc[-1]


from currency_predictor.prediction.factors import FactorModel


def _feature_frame(n, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"f{c}": rng.standard_normal(n) for c in range(4)})


def test_factor_model_fit_predict_shapes():
    X = _feature_frame(80)
    vol_y = pd.Series(np.abs(np.random.default_rng(2).standard_normal(80)))
    dir_y = pd.Series((np.random.default_rng(3).random(80) > 0.5).astype(float))
    fm = FactorModel(random_state=0)
    fm.fit(X, vol_y, dir_y)
    vol_pred, dir_pred = fm.predict(X)
    assert len(vol_pred) == len(X) and len(dir_pred) == len(X)
    assert np.all((dir_pred >= 0) & (dir_pred <= 1))


def test_factor_model_crossfit_is_out_of_fold():
    X = _feature_frame(120, seed=5)
    vol_y = pd.Series(X["f0"].abs() + 0.01)
    dir_y = pd.Series((X["f1"] > 0).astype(float))
    fm = FactorModel(random_state=0)
    vol_oof, dir_oof = fm.crossfit_predict(X, vol_y, dir_y, k=4)
    assert len(vol_oof) == len(X) and len(dir_oof) == len(X)
    fm.fit(X, vol_y, dir_y)
    vol_in, _ = fm.predict(X)
    assert not np.allclose(vol_oof, vol_in)
