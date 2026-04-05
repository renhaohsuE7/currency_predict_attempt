# Plan #0110: Adaptive Feature Engineering

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: data_processor.py
- **Priority**: P1

## 目標

根據資料長度動態調整 feature engineering 參數（lag features、rolling window sizes），避免短期資料產生過多 NaN 或特徵不足。

## 設計決策

| 問題 | 決策 |
| --- | --- |
| 動態調整什麼 | lag periods + rolling window sizes |
| 調整規則 | `max_lag = max(1, int(data_length * 0.1))`，`max_window = max(3, int(data_length * 0.15))` |
| API 設計 | 新增 `auto_lags=True` 參數，預設開啟；`lags` 參數仍可手動覆蓋 |
| EMA/MACD | 永遠產生（ewm 不產生 NaN） |
| BB 最小窗口 | `>= 5`（低於此 std 無意義） |
| RSI/Volatility 最小窗口 | `>= 3` |

## 修改範圍

| 檔案 | 說明 |
| --- | --- |
| `data_processor.py` | `_compute_adaptive_lags(data_length)` helper、`_compute_adaptive_windows(data_length)` helper、重寫 `create_technical_indicators()` 和 `create_lagged_features()` |
| `tests/test_data_processor.py` | 新增 `TestAdaptiveFeatureEngineering` class（12 tests） |
| `tests/test_currency_predictor.py` | 更新 `test_create_technical_indicators` 預期（100 行資料不產生 MA_20） |

## 實作摘要

### `_compute_adaptive_windows(data_length)` → dict

```python
max_rolling = max(3, int(data_length * 0.15))
{
    'ma_windows': [w for w in [5, 10, 20] if w <= max_rolling],
    'rsi_window': min(10, max_rolling),
    'bb_window': min(15, max_rolling),
    'volatility_window': min(15, max_rolling),
    'price_change_periods': [p for p in [1, 5, 10] if p <= max_rolling],
    'volume_window': min(15, max_rolling),
}
```

### `_compute_adaptive_lags(data_length)` → list

```python
max_lag = max(1, int(data_length * 0.1))
[lag for lag in [1, 2, 3, 5] if lag <= max_lag]
```

### `create_technical_indicators(data)` — 改用 adaptive windows

- MA: 只建立 `ma_windows` 中的窗口
- EMA/MACD: 永遠建立（ewm 無 NaN）
- RSI: `rsi_window >= 3` 才建立
- BB: `bb_window >= 5` 才建立
- Volatility: `volatility_window >= 3` 才建立
- Price changes: 只建立在 `price_change_periods` 中的週期

### `create_lagged_features(data, lags=None, auto_lags=True)` — 改用 adaptive lags

- `lags` 明確指定 → 使用指定值（手動覆蓋）
- `lags=None, auto_lags=True` → 使用 `_compute_adaptive_lags()`
- `lags=None, auto_lags=False` → 使用舊預設 `[1, 2, 3, 5]`

### 資料長度 → 特徵行為

| 資料長度 | max_rolling | max_lag | MA | BB | RSI | Lags |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 4 | 3 | — | — | ✓(4) | [1,2,3] |
| 100 | 15 | 10 | 5,10 | ✓(15) | ✓(10) | [1,2,3,5] |
| 200 | 30 | 20 | 5,10,20 | ✓(15) | ✓(10) | [1,2,3,5] |
| 300 | 45 | 30 | 5,10,20 | ✓(15) | ✓(10) | [1,2,3,5] |

## 測試結果

- **292 passed, 0 failed**（排除 HuggingFace pre-existing PermissionError）
- 12 新增 adaptive feature tests
- 既有 9 DataProcessor tests 全部通過
- `test_currency_predictor.py` 更新後通過

## 完成標準

- [x] 30 行資料不會因 lag/window 產生 >50% NaN
- [x] 300 行資料的行為與現有一致（向後相容）
- [x] `lags` 參數手動覆蓋時不觸發 adaptive 邏輯
- [x] 既有測試通過（292 passed）
