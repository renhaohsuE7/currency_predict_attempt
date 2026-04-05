# Fix: PatchTSTSklearn predict() Feature Mismatch Bug

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: `models/patchtst/sklearn/`

## Bug

`PatchTSTSklearn.predict()` 因 feature 維度不一致而失敗：

```
ValueError: X has 1554 features, but StandardScaler is expecting 1596 features as input
```

**根本原因**：`_extract_features_from_data(X, y)` 做 `pd.concat([X, y.to_frame()], axis=1)` 將 y 合併回 X，使得訓練時的 input sequence 有 `N+1` 欄位。但 `predict()` 用 `X.iloc[-seq_len:].values` 只有 `N` 欄位。feature vector size = `n_patches * 6 * n_columns`，欄位數不同導致 StandardScaler 維度不 match。

**影響範圍**：

- 直接 `model.fit(X, y)` 後 `model.predict(X)` → 失敗
- `CurrencyPredictor._evaluate_model(X, y)` → 呼叫 `model.evaluate(X)` → 失敗
- `CurrencyPredictor.predict()` → 傳入含 Close 的 processed_data（剛好 N+1 欄）→ 意外成功但欄位語義不同

## Fix Summary

| 變更 | 檔案 |
| --- | --- |
| `_extract_features_from_data()` 不再 concat y，改用 X 作 input、y 獨立取 target | `models/patchtst/sklearn/model.py` |
| `fit()` 儲存 `self._feature_columns` | 同上 |
| `predict()` 自動過濾到訓練欄位，缺欄位報清楚錯誤 | 同上 |
| `save_model()` / `load_model()` 持久化 `_feature_columns` | 同上 |
| 移除 test workaround（手動 concat X+y） | `test_e2e_offline.py`, `test_models_patchtst.py` |
| 新增 3 個 predict 欄位測試 | `test_models_patchtst.py` |

## Test Results

- **327 passed**, 0 failed（non-slow, non-e2e）
- 3 new tests: `TestPatchTSTPredictColumns`
