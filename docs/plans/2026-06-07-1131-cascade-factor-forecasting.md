# Cascade Factor-Augmented Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict realized volatility and direction as factors, then feed them (leak-free, via cross-fitting) into a price/return forecaster — working and tested across the sklearn, HuggingFace and Lightning backends.

**Architecture:** Two-stage cascade. Stage-1 = lightweight sklearn models mapping per-row processed features → future realized-vol (regressor) and future direction (classifier). Their cross-fitted predictions are injected as two extra columns (`Factor_Vol`, `Factor_Dir`) into the processed feature frame; Stage-2 is the existing PatchTST forecaster (any backend) consuming those columns as extra channels.

**Tech Stack:** Python 3.11, pandas/numpy, scikit-learn, existing `currency_predictor` package, pytest, docker compose + uv.

- **Status**: draft
- **Spec**: `docs/superpowers/specs/2026-06-07-cascade-factor-forecasting-design.md`
- **Run convention**: all commands inside container, e.g. `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest ...`

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/currency_predictor/prediction/factors.py` (new) | Pure factor-target functions + `FactorModel` (vol regressor + dir classifier, with cross-fit) |
| `src/currency_predictor/prediction/cascade.py` (new) | `CascadePredictor` — orchestrates stage-1 → factor injection → stage-2; returns {price, vol, dir} |
| `src/currency_predictor/config/settings.py` (modify) | `CascadeConfig` |
| `src/currency_predictor/prediction/__init__.py` (modify) | export `CascadePredictor` |
| `tests/test_factors.py` (new) | factor targets + FactorModel + no-leakage |
| `tests/test_cascade.py` (new) | cross-fit no-leakage, end-to-end per backend, factor-lift |

---

## Task 1: Factor target functions

**Files:**
- Create: `src/currency_predictor/prediction/factors.py`
- Test: `tests/test_factors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_factors.py
import numpy as np
import pandas as pd
import pytest

from currency_predictor.prediction.factors import (
    realized_volatility_target,
    direction_target,
)


def test_realized_volatility_target_known_values():
    # constant up-moves → zero volatility of log-returns
    close = pd.Series([100.0 * (1.01 ** i) for i in range(20)])
    vol = realized_volatility_target(close, horizon=5)
    # first rows have a full future window; vol of constant-rate returns ≈ 0
    assert vol.iloc[0] == pytest.approx(0.0, abs=1e-9)
    # last `horizon` rows have no full future window → NaN
    assert vol.iloc[-1] != vol.iloc[-1]  # NaN


def test_realized_volatility_matches_manual():
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.standard_normal(30)))
    vol = realized_volatility_target(close, horizon=4)
    r = np.log(close.astype(float)).diff().to_numpy()
    # vol[t] = std of r[t+1 : t+1+4]
    assert vol.iloc[3] == pytest.approx(np.std(r[4:8]))


def test_direction_target_up_down():
    close = pd.Series([100, 101, 102, 103, 104, 105], dtype=float)  # always up
    d = direction_target(close, horizon=2)
    assert d.iloc[0] == 1.0
    down = pd.Series([105, 104, 103, 102, 101, 100], dtype=float)
    assert direction_target(down, horizon=2).iloc[0] == 0.0


def test_direction_target_tail_is_nan():
    close = pd.Series(np.arange(10), dtype=float)
    d = direction_target(close, horizon=3)
    assert d.iloc[-1] != d.iloc[-1]  # NaN at the tail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest tests/test_factors.py -q`
Expected: FAIL (ModuleNotFoundError: factors).

- [ ] **Step 3: Write minimal implementation**

```python
# src/currency_predictor/prediction/factors.py
"""Factor targets and Stage-1 factor models for cascade forecasting.

Factor targets are FUTURE quantities (shifted), so a model trained to predict
them uses only past context — no look-ahead within the target itself.
"""

import numpy as np
import pandas as pd


def _log_returns(close: pd.Series) -> np.ndarray:
    return np.log(close.astype(float)).diff().to_numpy()


def realized_volatility_target(close: pd.Series, horizon: int) -> pd.Series:
    """Future realized volatility at each t = std of log-returns over (t, t+horizon].

    The last `horizon` rows are NaN (no complete future window).
    """
    r = _log_returns(close)
    n = len(r)
    out = np.full(n, np.nan)
    for t in range(n - horizon):
        out[t] = float(np.std(r[t + 1 : t + 1 + horizon]))
    return pd.Series(out, index=close.index, name="vol_target")


def direction_target(close: pd.Series, horizon: int) -> pd.Series:
    """Future direction label at each t: 1.0 if cumulative log-return over
    (t, t+horizon] > 0 else 0.0. Last `horizon` rows are NaN.
    """
    r = _log_returns(close)
    n = len(r)
    out = np.full(n, np.nan)
    for t in range(n - horizon):
        out[t] = 1.0 if float(np.sum(r[t + 1 : t + 1 + horizon])) > 0 else 0.0
    return pd.Series(out, index=close.index, name="dir_target")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest tests/test_factors.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/currency_predictor/prediction/factors.py tests/test_factors.py
git commit -m "feat: cascade factor target functions (realized vol, direction)"
```

---

## Task 2: FactorModel (vol regressor + dir classifier, cross-fit)

**Files:**
- Modify: `src/currency_predictor/prediction/factors.py`
- Test: `tests/test_factors.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_factors.py
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
    assert np.all((dir_pred >= 0) & (dir_pred <= 1))  # P(up)


def test_factor_model_crossfit_is_out_of_fold():
    # Target leakage check: cross-fit predictions must differ from in-sample
    # fit-then-predict (a model that memorised would match exactly).
    X = _feature_frame(120, seed=5)
    vol_y = pd.Series(X["f0"].abs() + 0.01)  # learnable
    dir_y = pd.Series((X["f1"] > 0).astype(float))
    fm = FactorModel(random_state=0)
    vol_oof, dir_oof = fm.crossfit_predict(X, vol_y, dir_y, k=4)
    assert len(vol_oof) == len(X) and len(dir_oof) == len(X)
    # OOF preds are produced by models not trained on that row → not identical to
    # an in-sample refit's predictions
    fm.fit(X, vol_y, dir_y)
    vol_in, _ = fm.predict(X)
    assert not np.allclose(vol_oof, vol_in)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest tests/test_factors.py -q`
Expected: FAIL (ImportError: FactorModel).

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/currency_predictor/prediction/factors.py
from typing import Tuple

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import KFold, cross_val_predict


class FactorModel:
    """Stage-1 factor predictors: a regressor for realized volatility and a
    classifier for direction (returns P(up))."""

    def __init__(self, random_state: int = 42) -> None:
        self.vol_model = HistGradientBoostingRegressor(random_state=random_state)
        self.dir_model = HistGradientBoostingClassifier(random_state=random_state)
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, vol_y: pd.Series, dir_y: pd.Series) -> "FactorModel":
        self.vol_model.fit(X.to_numpy(), np.asarray(vol_y, dtype=float))
        self.dir_model.fit(X.to_numpy(), np.asarray(dir_y, dtype=int))
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        vol_pred = np.asarray(self.vol_model.predict(X.to_numpy()), dtype=float)
        proba = self.dir_model.predict_proba(X.to_numpy())
        # P(up) = probability of class 1
        classes = list(self.dir_model.classes_)
        up_idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
        dir_pred = np.asarray(proba[:, up_idx], dtype=float)
        return vol_pred, dir_pred

    def crossfit_predict(
        self, X: pd.DataFrame, vol_y: pd.Series, dir_y: pd.Series, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Out-of-fold predictions to avoid factor leakage in Stage-2 training."""
        kf = KFold(n_splits=k, shuffle=False)
        Xn = X.to_numpy()
        vol_oof = cross_val_predict(
            HistGradientBoostingRegressor(random_state=self.random_state),
            Xn, np.asarray(vol_y, dtype=float), cv=kf,
        )
        dir_oof = cross_val_predict(
            HistGradientBoostingClassifier(random_state=self.random_state),
            Xn, np.asarray(dir_y, dtype=int), cv=kf, method="predict_proba",
        )
        classes = sorted(set(int(v) for v in dir_y))
        up_idx = classes.index(1) if 1 in classes else dir_oof.shape[1] - 1
        return np.asarray(vol_oof, dtype=float), np.asarray(dir_oof[:, up_idx], dtype=float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest tests/test_factors.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/currency_predictor/prediction/factors.py tests/test_factors.py
git commit -m "feat: FactorModel with cross-fit (leak-free) factor predictions"
```

---

## Task 3: CascadeConfig

**Files:**
- Modify: `src/currency_predictor/config/settings.py`
- Test: `tests/test_cascade.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cascade.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest tests/test_cascade.py -q`
Expected: FAIL (ImportError: CascadeConfig).

- [ ] **Step 3: Write minimal implementation**

Add to `settings.py` after `PanelConfig` (around line 128) and register it in `AppSettings` next to `panel`:

```python
class CascadeConfig(BaseModel):
    """Cascade 因子增強預測配置:先預測波動率/方向因子,再注入下游價格模型。"""

    enabled: bool = Field(False, description="啟用 cascade 因子增強預測")
    crossfit_folds: int = Field(5, gt=1, description="Stage-2 訓練因子的 cross-fit 折數")
    stage2_backend: str = Field(
        "patchtst_sklearn",
        description="Stage-2 主預測器 backend(任一 ModelFactory 名稱)",
    )
```

In `AppSettings`, after `panel: PanelConfig = ...`:

```python
    # Cascade 因子增強(預設關閉)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest tests/test_cascade.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/currency_predictor/config/settings.py tests/test_cascade.py
git commit -m "feat: CascadeConfig"
```

---

## Task 4: CascadePredictor (orchestration, sklearn end-to-end)

**Files:**
- Create: `src/currency_predictor/prediction/cascade.py`
- Modify: `src/currency_predictor/prediction/__init__.py`
- Test: `tests/test_cascade.py`

**Design notes for the engineer:**
- Reuse `CurrencyPredictor` (`prediction/predictor.py`) purely for data prep and the Stage-2 model lifecycle. Build the processed feature frame by calling the same `DataProcessor` steps the predictor uses, OR add a small helper `CurrencyPredictor.build_processed_data(symbol, period)` that returns the processed DataFrame (clean → technical indicators → CAPM-if-stock → lagged). Prefer adding that helper (single source of truth) — see Step 3.
- Stage-1 features = all numeric columns of the processed frame EXCEPT the raw target (`Close`), taken at rows that have a non-NaN factor target.
- Cross-fit factor series on training rows; for any rows without an OOF value (the trailing `horizon` rows used only at inference), fall back to the full-data `FactorModel.predict`.
- Inject `Factor_Vol`, `Factor_Dir` columns into the processed frame; for HF/Lightning set `use_multi_channel=True` so the columns become input channels (sklearn already uses all columns).

- [ ] **Step 1: Add `build_processed_data` helper to CurrencyPredictor**

In `prediction/predictor.py`, extract the processing pipeline used at the top of `prepare_training_data` into a reusable method (and call it from `prepare_training_data` and `predict` to stay DRY):

```python
    def build_processed_data(self, symbol: str, period: str) -> "pd.DataFrame":
        """Load raw data and run clean → indicators → (CAPM for stocks) → lagged."""
        clean_symbol = _clean_symbol(symbol)
        raw_data = self.data_storage.load_raw_data(clean_symbol, period)
        if raw_data is None or raw_data.empty:
            raise ValueError(f"找不到 {symbol} 的資料")
        cleaned = self.data_processor.clean_data(raw_data)
        with_ind = self.data_processor.create_technical_indicators(cleaned)
        asset_type = classify_symbol(symbol)
        if asset_type == AssetType.STOCK and self.capm_config.get("enabled", False):
            market_data = self._get_market_index_data(period=period)
            rf_rate = self._get_risk_free_rate()
            rolling_w = self.capm_config.get("rolling_window", 252)
            with_ind = self.data_processor.create_capm_features(
                with_ind, market_data, rf_rate, rolling_w
            )
        return self.data_processor.create_lagged_features(with_ind, lags=[1, 2, 3])
```

- [ ] **Step 2: Write the failing test (sklearn end-to-end)**

```python
# append to tests/test_cascade.py
import numpy as np
import pandas as pd
from unittest.mock import patch

from currency_predictor.prediction.cascade import CascadePredictor


def _ohlcv(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1,
         "Close": close, "Volume": rng.integers(1e3, 1e4, n)},
        index=idx,
    )


def test_cascade_sklearn_end_to_end():
    df = _ohlcv()
    cfg = {
        "model_name": "patchtst_sklearn",
        "model_params": {"seq_len": 32, "pred_len": 5, "patch_len": 8, "stride": 4},
        "model_training": {"target_transform": "log_return", "test_days": 60},
        "cascade": {"enabled": True, "crossfit_folds": 3, "stage2_backend": "patchtst_sklearn"},
    }
    cp = CascadePredictor(cfg)
    # patch data loading to return our synthetic frame
    with patch.object(cp.predictor.data_storage, "load_raw_data", return_value=df):
        result = cp.fit("TEST", period="2y")
        out = cp.predict("TEST", horizon=5)
    assert {"price", "vol", "dir"} <= set(out)
    assert len(out["price"]) == 5
    assert np.all(np.isfinite(out["price"]))
    assert 0.0 <= out["dir"] <= 1.0
    assert out["vol"] >= 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest tests/test_cascade.py::test_cascade_sklearn_end_to_end -q`
Expected: FAIL (ImportError: cascade).

- [ ] **Step 4: Write minimal implementation**

```python
# src/currency_predictor/prediction/cascade.py
"""Cascade factor-augmented forecasting.

Stage-1 predicts future realized volatility and direction (leak-free via
cross-fitting); these are injected as extra feature channels; Stage-2 (any
PatchTST backend) forecasts price/return from the augmented features.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..models.factory import ModelFactory
from ..models.patchtst.config import TrainingConfig
from .factors import FactorModel, direction_target, realized_volatility_target
from .predictor import CurrencyPredictor

logger = logging.getLogger(__name__)

_MULTI_CHANNEL_BACKENDS = {"patchtst_huggingface", "patchtst_lightning"}


class CascadePredictor:
    """Two-stage cascade: factors (vol, dir) → price/return."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cascade_cfg = config.get("cascade", {})
        self.backend = self.cascade_cfg.get("stage2_backend", "patchtst_sklearn")
        self.k = int(self.cascade_cfg.get("crossfit_folds", 5))
        self.model_params = dict(config.get("model_params", {}))
        self.horizon = int(self.model_params.get("pred_len", 15))
        self.target_transform = config.get("model_training", {}).get(
            "target_transform", "log_return"
        )
        # Stage-2 needs multi-channel for HF/Lightning so factor columns are used
        if self.backend in _MULTI_CHANNEL_BACKENDS:
            self.model_params["use_multi_channel"] = True

        # predictor used for data prep + Stage-2 model lifecycle
        self.predictor = CurrencyPredictor(
            model_name=self.backend,
            model_params=self.model_params,
            data_storage_path=config.get("data_storage_path", "data"),
            capm_config=config.get("capm", {}),
            target_transform=self.target_transform,
        )
        self.factor_model = FactorModel()
        self._test_days = config.get("model_training", {}).get("test_days")

    def _augment(self, processed: pd.DataFrame, training: bool) -> pd.DataFrame:
        """Add Factor_Vol / Factor_Dir columns to the processed frame."""
        close = processed["Close"]
        vol_t = realized_volatility_target(close, self.horizon)
        dir_t = direction_target(close, self.horizon)
        feat_cols = [
            c for c in processed.select_dtypes(include=[np.number]).columns
            if c not in ("Close", "Factor_Vol", "Factor_Dir")
        ]
        Xf = processed[feat_cols]
        valid = vol_t.notna() & dir_t.notna()

        vol_series = pd.Series(np.nan, index=processed.index)
        dir_series = pd.Series(np.nan, index=processed.index)
        if training:
            vol_oof, dir_oof = self.factor_model.crossfit_predict(
                Xf[valid], vol_t[valid], dir_t[valid], k=self.k
            )
            vol_series.loc[valid] = vol_oof
            dir_series.loc[valid] = dir_oof
            # fit on all valid rows for inference-time predictions
            self.factor_model.fit(Xf[valid], vol_t[valid], dir_t[valid])
        # rows without OOF (and all rows at inference) → full-model predict
        need = vol_series.isna()
        if need.any():
            v_pred, d_pred = self.factor_model.predict(Xf[need])
            vol_series.loc[need] = v_pred
            dir_series.loc[need] = d_pred

        out = processed.copy()
        out["Factor_Vol"] = vol_series.values
        out["Factor_Dir"] = dir_series.values
        return out

    def fit(self, symbol: str, period: str = "2y", **train_kwargs) -> Dict[str, Any]:
        processed = self.predictor.build_processed_data(symbol, period)
        augmented = self._augment(processed, training=True)
        self._augmented_cache = augmented  # reused by predict()
        self._symbol, self._period = symbol, period

        # Stage-2: train on augmented features. target = Close (transform applied
        # inside predictor). Build X/y the same way prepare_training_data does.
        target_col = "Close"
        feature_cols = [c for c in augmented.columns if c != target_col]
        X = augmented[feature_cols]
        if self.target_transform == "log_return":
            y = self.predictor.data_processor.to_log_returns(augmented[target_col])
            mask = y.notna()
            X, y, augmented = X[mask], y[mask], augmented[mask]
        else:
            y = augmented[target_col]
        # chronological split
        test_days = self._test_days or max(2 * self.horizon, 30)
        split = len(X) - min(test_days, len(X) // 2)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        self.predictor.model.fit(X_train, y_train, training_config=TrainingConfig())
        self.predictor._target_column = target_col
        return {"symbol": symbol, "n_train": len(X_train), "trained": True}

    def predict(self, symbol: str, horizon: int = None, period: str = None) -> Dict[str, Any]:
        horizon = horizon or self.horizon
        processed = self.predictor.build_processed_data(symbol, period or self._period)
        augmented = self._augment(processed, training=False)
        last = augmented.iloc[-1]
        # Stage-2 price/return forecast (predictor handles log_return reconstruction)
        feature_cols = [c for c in augmented.columns if c != "Close"]
        preds = self.predictor.model.predict(augmented[feature_cols], horizon)
        preds = np.asarray(preds, dtype=float)
        if self.target_transform == "log_return":
            last_close = float(augmented["Close"].iloc[-1])
            preds = self.predictor.data_processor.from_log_returns(last_close, preds)
        return {
            "symbol": symbol,
            "price": preds,
            "vol": float(last["Factor_Vol"]),
            "dir": float(last["Factor_Dir"]),
        }
```

Add to `prediction/__init__.py`: `from .cascade import CascadePredictor` and add `"CascadePredictor"` to `__all__`.

- [ ] **Step 5: Run test to verify it passes**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest tests/test_cascade.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/currency_predictor/prediction/cascade.py src/currency_predictor/prediction/predictor.py src/currency_predictor/prediction/__init__.py tests/test_cascade.py
git commit -m "feat: CascadePredictor (sklearn end-to-end, cross-fit factors)"
```

---

## Task 5: Cross-backend end-to-end test (sklearn / HF / Lightning)

**Files:**
- Test: `tests/test_cascade.py`

- [ ] **Step 1: Write the failing/parametrized test**

```python
# append to tests/test_cascade.py
import importlib.util


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
        "model_params": {"seq_len": 32, "pred_len": 5, "patch_len": 8, "stride": 4,
                         "d_model": 32, "num_attention_heads": 2, "num_hidden_layers": 1},
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
```

- [ ] **Step 2: Run it**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest tests/test_cascade.py::test_cascade_all_backends_end_to_end -q`
Expected: 3 passed (or HF/Lightning skipped if deps absent; in CI `--all-extras` they run).

- [ ] **Step 3: Fix integration if HF/Lightning fail**

If an HF/Lightning run fails because the factor columns are not consumed, verify `use_multi_channel=True` is being set (it is, in `CascadePredictor.__init__`) and that the wrapper reads all numeric columns (see `huggingface/model.py:_infer_num_features` / `lightning/wrapper.py` `_prepare_*`). Adjust the wrapper's multi-channel path only if needed; keep changes minimal and covered by this test.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cascade.py src/currency_predictor/models/
git commit -m "test: cascade end-to-end across sklearn/HF/Lightning backends"
```

---

## Task 6: Factor-lift evaluation (cascade vs no-factor baseline)

**Files:**
- Modify: `src/currency_predictor/prediction/cascade.py`
- Test: `tests/test_cascade.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_cascade.py
def test_factor_lift_report_structure():
    df = _ohlcv(n=400, seed=2)
    cfg = {
        "model_name": "patchtst_sklearn",
        "model_params": {"seq_len": 32, "pred_len": 5, "patch_len": 8, "stride": 4},
        "model_training": {"target_transform": "log_return", "test_days": 80},
        "cascade": {"enabled": True, "crossfit_folds": 3, "stage2_backend": "patchtst_sklearn"},
    }
    cp = CascadePredictor(cfg)
    with patch.object(cp.predictor.data_storage, "load_raw_data", return_value=df):
        report = cp.evaluate_factor_lift("TEST", period="2y")
    assert "with_factors" in report and "without_factors" in report
    assert "rmse" in report["with_factors"] and "rmse" in report["without_factors"]
    assert "vol_rmse" in report and "dir_accuracy" in report
```

- [ ] **Step 2: Run it (fails — no method)**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest tests/test_cascade.py::test_factor_lift_report_structure -q`
Expected: FAIL (AttributeError: evaluate_factor_lift).

- [ ] **Step 3: Implement `evaluate_factor_lift`**

Add a method that: builds processed data; trains two Stage-2 models on the SAME chronological train split — one on augmented features (with `Factor_Vol`/`Factor_Dir`), one on the plain features — evaluates both on the test split via the model's `evaluate_rolling` (return space), and also reports Stage-1 factor quality (vol RMSE vs realized vol, direction accuracy) on the test rows. Return:

```python
    def evaluate_factor_lift(self, symbol: str, period: str = "2y") -> Dict[str, Any]:
        import numpy as np
        from .metrics import mda  # noqa: F401  (kept for parity; not required)

        processed = self.predictor.build_processed_data(symbol, period)
        augmented = self._augment(processed, training=True)
        seq_len, pred_len = int(self.model_params.get("seq_len", 32)), self.horizon

        def split_xy(frame):
            y = self.predictor.data_processor.to_log_returns(frame["Close"])
            mask = y.notna()
            X = frame[[c for c in frame.columns if c != "Close"]][mask]
            y = y[mask]
            test_days = self._test_days or max(2 * pred_len, 30)
            s = len(X) - min(test_days, len(X) // 2)
            return X.iloc[:s], y.iloc[:s], X.iloc[s:], y.iloc[s:]

        def fit_eval(frame):
            Xtr, ytr, Xte, yte = split_xy(frame)
            m = ModelFactory.create_model(self.backend, **self.model_params)
            m.fit(Xtr, ytr, training_config=TrainingConfig())
            if len(Xte) >= seq_len + pred_len:
                return m.evaluate_rolling(Xte, yte, seq_len, pred_len, y_train=ytr)["aggregate"]
            return m.evaluate_single_shot(Xte, yte, y_train=ytr)

        plain = augmented[[c for c in augmented.columns if c not in ("Factor_Vol", "Factor_Dir")]]
        with_f = fit_eval(augmented)
        without_f = fit_eval(plain)

        # Stage-1 factor quality on test rows
        vol_t = realized_volatility_target(processed["Close"], pred_len)
        dir_t = direction_target(processed["Close"], pred_len)
        valid = vol_t.notna()
        vol_rmse = float(np.sqrt(np.mean(
            (augmented["Factor_Vol"][valid] - vol_t[valid]) ** 2)))
        dir_acc = float(np.mean(
            (augmented["Factor_Dir"][valid] > 0.5).astype(float).values
            == dir_t[valid].values))
        return {
            "with_factors": with_f,
            "without_factors": without_f,
            "vol_rmse": vol_rmse,
            "dir_accuracy": dir_acc,
        }
```

- [ ] **Step 4: Run it**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest tests/test_cascade.py::test_factor_lift_report_structure -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/currency_predictor/prediction/cascade.py tests/test_cascade.py
git commit -m "feat: factor-lift evaluation (cascade vs no-factor baseline)"
```

---

## Task 7: Quality gate + real-data smoke

- [ ] **Step 1: Lint & type-check the new/changed files**

Run:
```
APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test bash -lc \
 "uv run black src/currency_predictor/prediction/factors.py src/currency_predictor/prediction/cascade.py src/currency_predictor/config/settings.py tests/test_factors.py tests/test_cascade.py && \
  uv run flake8 src/currency_predictor/prediction/factors.py src/currency_predictor/prediction/cascade.py && \
  uv run mypy src/currency_predictor/"
```
Expected: black OK, flake8 clean, mypy `Success`.

- [ ] **Step 2: Full suite**

Run: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest -q`
Expected: all pass (previous 670 + new cascade/factor tests).

- [ ] **Step 3: Real-data factor-lift smoke (2330.TW, all three backends)**

Run a short script in the container that, for each available backend, calls `CascadePredictor(...).evaluate_factor_lift("2330.TW", "2y")` and prints `with_factors` vs `without_factors` RMSE/MDA + `vol_rmse`/`dir_accuracy`. Record the honest result (does the factor help? does vol beat naive? is direction > 0.5?) in a short note under `docs/issues/2026-06-07-cascade-factor-lift-results.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/issues/2026-06-07-cascade-factor-lift-results.md
git commit -m "docs: cascade factor-lift real-data results (honest evaluation)"
```

---

## Self-Review Notes

- **Spec coverage:** factors (Task 1), FactorModel + cross-fit no-leakage (Task 2), CascadeConfig (Task 3), CascadePredictor returning {price,vol,dir} (Task 4), all-three-backends test (Task 5), factor-lift vs baseline + factor quality (Task 6), quality gate + honest real-data results (Task 7). All spec sections covered.
- **No silent-swallow:** none of the new code catches→empty; factor-lift surfaces real numbers (per `.claude/rules/fail-loud.md`).
- **Type consistency:** `realized_volatility_target`/`direction_target` return `pd.Series`; `FactorModel.predict`/`crossfit_predict` return `(np.ndarray, np.ndarray)`; `CascadePredictor.predict` returns `{"price","vol","dir"}` used consistently in Tasks 4–6.
- **Risk (HF/Lightning channels):** handled by `use_multi_channel=True` in `CascadePredictor.__init__` + Task 5 Step 3 contingency.
