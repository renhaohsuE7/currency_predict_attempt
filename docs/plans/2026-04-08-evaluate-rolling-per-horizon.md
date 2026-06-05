# Plan: evaluate_rolling + Backtest Alignment Fix + Per-Horizon Metrics

- **Date**: 2026-04-08
- **Status**: completed
- **Module**: base model, backtesting, predictor, reporting

## Context

`evaluate_single_shot()` 只用 `pred_len=15` 個點算 metrics，統計上不可靠且有時間對齊問題。Backtest runner 的 `_run_fold()` 也有相同的對齊 bug（`actual[:min_len]` start-aligned，但 predictions 是 end-of-input 之後的值）。

需要：
1. 新增 `evaluate_rolling()` — rolling origin evaluation
2. 修 backtest runner alignment — 改用 rolling evaluation
3. 新增 per-horizon metrics — h=1, h=2, ..., h=pred_len 各自的 error

---

## 設計

### `evaluate_rolling()` 滑動邏輯

```
test set: [t0, t1, ..., t59]  (60 天, seq_len=32, pred_len=15)

origin 0:  X=[t0..t31]   → predict → compare with [t32..t46]
origin 1:  X=[t1..t32]   → predict → compare with [t33..t47]
...
origin 13: X=[t13..t44]  → predict → compare with [t45..t59]

n_origins = len(X) - seq_len - pred_len + 1 = 60 - 32 - 15 + 1 = 14
```

每個 origin 產生 `pred_len` 個 (predicted, actual) 配對。

**Return structure:**
```python
{
    'aggregate': {rmse, mae, mape, mase?, mda, direction_accuracy, n_origins},
    'per_horizon': {1: {rmse, mae, mape}, 2: {...}, ..., pred_len: {...}},
    'h1_actual': np.ndarray,      # 每個 origin 的 h=1 actual (for visualization)
    'h1_predicted': np.ndarray,   # 每個 origin 的 h=1 predicted
}
```

### Backtest Runner Fix

`_run_fold()` 改用 `model.evaluate_rolling(X_test, y_test, seq_len, pred_len, y_train)` 取代舊的 single-shot predict + misaligned compare。`actual_prices`/`predicted_prices` 改用 rolling result 的 `h1_actual`/`h1_predicted`。

### Per-Horizon Metrics

`FoldResult` 和 `BacktestResult` 新增 optional 欄位存放 per-horizon data。Formatter 新增 per-horizon table 顯示。

---

## 影響範圍

| 檔案 | 修改 |
| --- | --- |
| `src/currency_predictor/models/base.py` | 新增 `evaluate_rolling()` method |
| `src/currency_predictor/backtesting/result.py` | `FoldResult` + `BacktestResult` 新增 `per_horizon_metrics`, `n_origins` 欄位 |
| `src/currency_predictor/backtesting/runner.py` | `_run_fold()` 改用 rolling evaluation；`_run_single_backtest()` 聚合 per-horizon metrics |
| `src/currency_predictor/prediction/predictor.py` | `_evaluate_model()` 改用 rolling（test set 夠大時），fallback single-shot |
| `src/currency_predictor/reporting/formatter.py` | 新增 per-horizon metrics 顯示 |
| `tests/test_evaluate_rolling.py` | **新檔** — rolling evaluation 單元測試 |
| `tests/test_backtest_runner.py` | 更新 mock model 和 assertions |

### 不受影響

- Model 子類別 (naive, sklearn, HF, lightning) — 不需修改，`evaluate_rolling()` 用外部切 window + 現有 `predict()` API
- `comparer.py` — `compare()`/`train()` flow 不變（它們走 predictor 的 `_evaluate_model` 路徑）
- `compute_unified_metrics()` 靜態方法 — 保持不變

---

## 實作步驟

### Step 1: `base.py` — 新增 `evaluate_rolling()`

在 `evaluate_single_shot()` 之後新增。核心邏輯：

```python
def evaluate_rolling(self, X, y_true, seq_len, pred_len, y_train=None, step=1):
    n = len(X)
    if n < seq_len + pred_len:
        raise ValueError(f"Data length ({n}) < seq_len ({seq_len}) + pred_len ({pred_len})")

    origins = range(seq_len, n - pred_len + 1, step)
    all_pred = np.zeros((len(origins), pred_len))
    all_actual = np.zeros((len(origins), pred_len))

    for i, origin in enumerate(origins):
        X_window = X.iloc[origin - seq_len : origin]
        preds = self.predict(X_window)[:pred_len]
        actuals = y_true.iloc[origin : origin + pred_len].values
        all_pred[i] = preds
        all_actual[i] = actuals

    # Per-horizon metrics (column-wise)
    per_horizon = {}
    for h in range(pred_len):
        errors = all_actual[:, h] - all_pred[:, h]
        per_horizon[h + 1] = {
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'mae': float(np.mean(np.abs(errors))),
        }

    # Aggregate metrics (flatten all pairs)
    flat_actual = all_actual.flatten()
    flat_pred = all_pred.flatten()
    aggregate = _compute_metrics(flat_actual, flat_pred, y_train)
    aggregate['n_origins'] = len(origins)

    return {
        'aggregate': aggregate,
        'per_horizon': per_horizon,
        'h1_actual': all_actual[:, 0],
        'h1_predicted': all_pred[:, 0],
    }
```

Reuse existing `mase()` and `mda()` from `prediction/metrics.py`。

### Step 2: `result.py` — 新增欄位

```python
@dataclass
class FoldResult:
    ...existing fields...
    per_horizon_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    n_origins: int = 0

@dataclass
class BacktestResult:
    ...existing fields...
    avg_per_horizon_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
```

使用 `field(default_factory=...)` 確保向後相容。

### Step 3: `runner.py` — 修 `_run_fold()` alignment

替換 lines 252-263：

```python
# 取得 seq_len / pred_len
seq_len = model_params.get('seq_len', model_params.get('context_length', 64))
pred_len = model_params.get('pred_len', model_params.get('prediction_length', 15))

# Rolling evaluation (fixes temporal alignment)
if len(X_test) >= seq_len + pred_len:
    rolling = model.evaluate_rolling(X_test, y_test, seq_len, pred_len, y_train=y_train)
    pred_metrics = rolling['aggregate']
    actual = rolling['h1_actual']
    pred = rolling['h1_predicted']
    per_horizon = rolling['per_horizon']
    n_origins = pred_metrics.get('n_origins', 0)
else:
    # Fallback: single-shot (test window too small)
    predictions = model.predict(X_test)
    pred = np.asarray(predictions, dtype=float)
    actual = y_test.values[-len(pred):]
    pred_metrics = ModelComparer.compute_unified_metrics(actual, pred)
    per_horizon = {}
    n_origins = 0
```

更新 `_run_single_backtest()` 聚合 per-horizon metrics across folds。

### Step 4: `predictor.py` — `_evaluate_model()` 改用 rolling

```python
def _evaluate_model(self, X, y_true, dataset_name, y_train=None):
    seq_len = self.model_params.get('seq_len', self.model_params.get('context_length', 64))
    pred_len = self.model_params.get('pred_len', self.model_params.get('prediction_length', 15))

    if len(X) >= seq_len + pred_len:
        rolling = self.model.evaluate_rolling(X, y_true, seq_len, pred_len, y_train)
        metrics = rolling['aggregate']
        metrics['per_horizon'] = rolling.get('per_horizon', {})
    else:
        metrics = self.model.evaluate_single_shot(X, y_true, y_train=y_train)

    logger.info(f"{dataset_name}集評估: RMSE={metrics.get('rmse', 0):.6f} (n_origins={metrics.get('n_origins', 'N/A')})")
    return dict(metrics)
```

### Step 5: `formatter.py` — 顯示 per-horizon metrics

在 comparison report 和 backtest report 中，如果 `per_horizon` 存在，新增一個表格：

```
--- Per-Horizon RMSE ---
  h=1: 0.042   h=2: 0.058   h=3: 0.071   ...   h=15: 0.156
```

### Step 6: 測試

**新檔 `tests/test_evaluate_rolling.py`：**
- `test_rolling_returns_correct_structure` — 檢查 keys
- `test_rolling_n_origins_math` — n_origins = len(X) - seq_len - pred_len + 1
- `test_rolling_per_horizon_count` — len(per_horizon) == pred_len
- `test_rolling_insufficient_data_raises` — ValueError
- `test_rolling_with_naive_model` — deterministic 驗證
- `test_rolling_step_reduces_origins` — step=5 產生更少 origins
- `test_rolling_h1_series_length` — h1_actual 長度 = n_origins

**更新 `tests/test_backtest_runner.py`：**
- mock model 需要支援 `evaluate_rolling` 或讓 `predict` 被 rolling 呼叫
- 驗證 `FoldResult.per_horizon_metrics` 有值
- 驗證 `FoldResult.n_origins > 0`

### Step 7: 跑全部測試

---

## 完成標準

- [x] `evaluate_rolling()` 可正確計算多 origin 的 metrics
- [x] Per-horizon metrics 報告 h=1 到 h=pred_len
- [x] Backtest runner 使用 rolling evaluation，temporal alignment 正確
- [x] `_evaluate_model()` 預設用 rolling，小 test set fallback single-shot
- [x] 全部測試通過
- [x] Pipeline re-run 確認 metrics 合理
