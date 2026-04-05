# Plan #0170: Fix Multi-Step Prediction — Flat Line Problem

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: models/patchtst, prediction, main, visualization

## 目標

修正三個 PatchTST 模型的多步預測問題：預測結果在比較圖中幾乎都是直線，以及比較圖的 actual 線用了 placeholder zeros。

## 問題分析

| 模型 | 現象 | 根因 |
| --- | --- | --- |
| **sklearn** | 完全是直線（N 個相同值） | `np.mean(y_sequences, axis=1)` 把 N 步目標壓成 1 值，`np.full(pred_len, prediction[0])` 重複填充 |
| **HuggingFace** | 幾乎是直線（range=0.15, std=0.044） | 只用 Close 1 channel（忽略 27 個 technical indicators），小資料量學到「預測均值」 |
| **Lightning** | 有波動但不準（RMSE=1.45） | 同 HuggingFace，只用 Close 1 channel |
| **比較圖 actual** | Y 軸 0 的直線 | `main.py` 用 `np.zeros(n)` 當 actual（comparer 計算了 y_test 卻沒存） |

## 設計決策

| 問題 | 決策 |
| --- | --- |
| sklearn 如何做真正多步預測？ | `MultiOutputRegressor(GradientBoostingRegressor(...))` — 每個 timestep 獨立一個 regressor |
| HuggingFace/Lightning 如何改善？ | 新增 `use_multi_channel` config，支援 27+ features 作為多 channel 輸入 |
| 比較圖 actual 怎麼修？ | comparer 存 `actual_series` (y_test)，main.py 改用真實資料 |
| 向後相容？ | `use_multi_channel` 預設 False，不影響現有行為 |

---

## Phase 1: sklearn 多步預測 + 比較圖 actual

### 1.1 sklearn MultiOutputRegressor

**File**: `src/currency_predictor/models/patchtst/sklearn/model.py`

**變更點**:

1. `import` — 新增 `from sklearn.multioutput import MultiOutputRegressor`

2. `__init__()` (line 110-114) — wrap GBR in MultiOutputRegressor:
   ```python
   base_regressor = GradientBoostingRegressor(
       n_estimators=..., max_depth=..., random_state=...
   )
   self.ensemble_model = MultiOutputRegressor(base_regressor)
   ```

3. `_extract_features_from_data()` (line 232) — 保留完整目標矩陣:
   ```python
   # Before: targets = np.mean(y_sequences, axis=1)  # (N,)
   # After:
   targets = y_sequences  # (N, pred_len)
   ```

4. `fit()` target_scaler (lines 280-282, 298-300) — 處理 2D targets:
   ```python
   # Before: .fit_transform(targets.reshape(-1, 1)).flatten()
   # After:
   training_targets_scaled = self.target_scaler.fit_transform(training_targets)
   ```

5. `predict()` (lines 355-367) — 移除 np.full:
   ```python
   prediction_scaled = self.ensemble_model.predict(features_scaled)    # (1, pred_len)
   prediction = self.target_scaler.inverse_transform(prediction_scaled) # (1, pred_len)
   multi_step_prediction = prediction.flatten()                         # (pred_len,)
   ```

6. `predict_with_uncertainty()` — 同步更新 inverse_transform shape

### 1.2 比較圖使用真實 actual

**File**: `src/currency_predictor/prediction/comparer.py`

- line 205 之後新增: `symbol_result['actual'] = actual_series`

**File**: `main.py` `_generate_comparison_charts()`

- lines 193-198: 用 `sym_data.get('actual')` 取代 `np.zeros(n)` placeholder
- 需處理 actual 長度 ≈ test_size × rows vs predictions 長度 = pred_len 的對齊

### 1.3 測試

- `tests/test_models_patchtst.py` — 新增 `test_predict_non_flat_multistep`: 確認預測值不全相同
- `tests/test_model_comparer.py` — 新增 `test_compare_result_contains_actual`: 確認結果含 actual

---

## Phase 2: HuggingFace / Lightning 多 channel 輸入

### 2.1 Config

**File**: `src/currency_predictor/models/patchtst/config.py`

- `PatchTSTConfig` 新增 `use_multi_channel: bool = False`

### 2.2 HuggingFace

**File**: `src/currency_predictor/models/patchtst/huggingface/model.py`

- `_prepare_data()` — `use_multi_channel=True` 時用所有 numeric columns，記錄 `_target_channel_idx`
- `_setup_model()` — `num_input_channels` 動態設定
- `predict()` — 多 channel 時從 output 取 target channel
- scaler — 多 channel 時 fit 所有 features，inverse_transform 只取 target column

### 2.3 Lightning

**File**: `src/currency_predictor/models/patchtst/lightning/wrapper.py`

- 同 HuggingFace 的修改模式

### 2.4 Config.json

```json
{
    "model_params": {
        "use_multi_channel": true
    }
}
```

### 2.5 測試

- 新增 `test_huggingface_multi_channel_predict`, `test_lightning_multi_channel_predict`
- 確認 `use_multi_channel=false` 時現有 tests 不受影響

---

## 修改檔案清單

| 檔案 | 變更 | Phase |
| --- | --- | --- |
| `src/currency_predictor/models/patchtst/sklearn/model.py` | MultiOutputRegressor + 移除 np.full | 1 |
| `src/currency_predictor/prediction/comparer.py` | 儲存 actual_series | 1 |
| `main.py` | 比較圖用真實 actual | 1 |
| `tests/test_models_patchtst.py` | 新增 non-flat test | 1 |
| `tests/test_model_comparer.py` | 新增 actual test | 1 |
| `src/currency_predictor/models/patchtst/config.py` | 新增 use_multi_channel | 2 |
| `src/currency_predictor/models/patchtst/huggingface/model.py` | 多 channel 支援 | 2 |
| `src/currency_predictor/models/patchtst/lightning/wrapper.py` | 多 channel 支援 | 2 |

## 執行順序

1. Phase 1.1: sklearn MultiOutputRegressor
2. Phase 1.3: 新增 sklearn non-flat test → 驗證通過
3. Phase 1.2: comparer 存 actual + main.py 用真實 actual
4. Phase 1.3: 新增 comparer actual test → 驗證通過
5. 全回歸測試: `uv run pytest -x`
6. 重跑 `uv run main.py --symbols USDTWD=X --compare -v` 驗證圖表
7. Phase 2: HuggingFace/Lightning 多 channel
8. 全回歸測試 + 重跑比較

## 驗證

```bash
# 單元測試
uv run pytest -x

# 視覺驗證
uv run main.py --symbols USDTWD=X --compare -v
# 檢查 results/figures/USDTWD_model_comparison.png — 不再有直線
```

## 完成標準

- [ ] sklearn 預測 N 步產生 N 個不同值（非直線）
- [ ] 比較圖 actual 線用真實 y_test 資料
- [ ] HuggingFace/Lightning 支援 `use_multi_channel` 多 channel 輸入
- [ ] 全部既有 tests 通過 + 新增 tests 通過
- [ ] 視覺驗證比較圖不再全是直線
