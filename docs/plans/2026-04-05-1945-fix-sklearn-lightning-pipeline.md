# Fix sklearn & Lightning Prediction Pipeline

- **Date**: 2026-04-05 19:45
- **Status**: completed
- **Module**: models/patchtst/sklearn, models/patchtst/lightning

## 目標

修復 sklearn 和 Lightning 的預測精準度問題，使三個 backend 輸出合理且一致的預測結果。

## 問題分析

見 `docs/issues/2026-04-05-model-accuracy-analysis.md`

| Model | RMSE | 問題 |
|-------|------|------|
| HuggingFace | 31 | 正常 |
| Lightning | 875 | 缺少 RevIN |
| sklearn | 897 | 多通道 feature explosion (1,134 features) |

## 影響範圍

| 檔案 | 修改內容 |
|------|----------|
| `src/currency_predictor/models/patchtst/sklearn/model.py` | 改為 Close-only pipeline，移除 target_scaler |
| `src/currency_predictor/models/patchtst/lightning/modules.py` | 新增 RevIN 到 PatchTSTModel |
| `tests/test_models_patchtst.py` | 更新 sklearn 相關測試 |
| `tests/test_patchtst_lightning.py` | 新增 RevIN 測試 |

## 實作步驟

### Step 1: sklearn — Close-only pipeline

1. `_extract_features_from_data()`: 只從 Close column 提取 values → (N, 1)
2. `fit()`: 移除 `target_scaler`，改用簡單的 mean/std 正規化 target
3. `predict()`: 只從 Close column 取 input sequence
4. `save_model()` / `load_model()`: 移除 target_scaler，新增 _target_mean/_target_std

Feature count: 1,134 → 42（6 stats × 1 channel × 7 patches）

### Step 2: Lightning — 新增 RevIN

1. 在 `modules.py` 新增 `RevIN` class（Reversible Instance Normalization）
2. 在 `PatchTSTModel.forward()` 中加入 RevIN normalize/denormalize
3. RevIN 是 PatchTST 論文的核心組件，HuggingFace 有內建（`scaling='std'`），Lightning 目前缺少

### Step 3: 測試更新

- sklearn: 驗證 Close-only pipeline 和新的 save/load
- Lightning: 驗證 RevIN 可逆性和 model forward pass

## 完成標準

- sklearn 和 Lightning 預測在 Close price ±10% 以內
- Feature count 從 1,134 降到 42
- 所有 tests 通過
- Forecast chart 三個模型都在合理範圍
