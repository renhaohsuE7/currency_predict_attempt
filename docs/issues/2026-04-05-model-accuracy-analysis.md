# Model Accuracy Analysis — 2330.TW 15-Day Forecast

- **Date**: 2026-04-05
- **Status**: resolved
- **Symbol**: 2330.TW (台積電)
- **Config**: tw2330_config.json (seq_len=64, pred_len=15, data_period=2y)

## Comparison Results

| Model | RMSE | MAE | Training Time |
|-------|------|-----|---------------|
| **patchtst_huggingface** | **31.07** | **25.26** | 1.9s |
| patchtst_lightning | 875.87 | 873.76 | 0.8s |
| patchtst_sklearn | 897.56 | 894.81 | 403.4s |

## Forecast Chart 觀察

- **HuggingFace**: 預測維持在 ~1800 附近，合理（RMSE ~1.7% of price）
- **sklearn**: 預測掉到 ~800-1000，偏離 45-55%
- **lightning**: 預測掉到 ~900-1050，偏離 40-50%

## Root Cause Analysis

### 三個 backend 的 data pipeline 完全不同

#### HuggingFace (正常)

```
Close prices → 單一 scaler 正規化 → model predict in scaled space → inverse_transform
```

全程只用 Close price 單通道，scaler distribution 穩定。

#### sklearn (壞掉)

```
多通道 data → 提取 patch 統計特徵 (mean/std/min/max/median/trend) → 800+ features
→ feature scaler + 獨立 target scaler → GradientBoosting predict → inverse target scaler
```

**問題：patch 統計特徵的 distribution 非常不穩定**

- Training 時 Close 平均 ~1850，patch mean features 圍繞 1850
- Prediction 時 Close 移到 ~1800，patch mean features 整體偏移
- Feature scaler 用的是舊的 fit，新 features 變成 **out-of-distribution** 輸入
- GradientBoosting 對 OOD 輸入外推出極端值（scaled space 中 -30 ~ -40）
- Inverse transform: `(-35 * 30) + 1850 = 800`

**關鍵程式碼：**
- `sklearn/model.py` lines 157-196: `_extract_patch_features()` 產生異質統計特徵
- `sklearn/model.py` lines 279-282: 使用兩個獨立 scaler（features vs targets）
- `sklearn/model.py` lines 342-357: Prediction 用舊 scaler 處理新 distribution 的 features

#### Lightning (缺少 RevIN)

```
Close prices → 外部 StandardScaler 正規化 → PatchTSTModel (無 instance norm) → predict
```

**問題：缺少 Reversible Instance Normalization (RevIN)**

- HuggingFace 的 `PatchTSTForPrediction` 內建 `scaling='std'`，每個 input sequence 獨立做 instance normalization
- Lightning 的自製 `PatchTSTModel` 完全沒有 RevIN，只依賴外部 StandardScaler
- RevIN 是 PatchTST 論文的核心組件，對 non-stationary time series 至關重要
- 缺少 RevIN 導致 model 無法學習 price level invariant patterns
- val_loss 最佳只有 2.944（scaled space），遠不如 HuggingFace

#### Evaluation pipeline (也壞掉)

```
prepare_training_data → X = all columns except Close, y = Close
_evaluate_model(X, y) → model.predict(X)  ← X 沒有 Close!
```

**問題：`_evaluate_model` 傳入的 X 不含 Close 欄位**

- 所有 PatchTST backend 的 predict 都需要 Close 欄位作為 input
- 但 `prepare_training_data` 在 line 260 把 Close 排除在 X 之外
- sklearn predict fallback 到 `numeric_cols[0]`（可能是 Open），用錯誤的欄位計算 RMSE
- 最終預測路徑（`predict_symbol`）傳入 `processed_data`（含 Close）不受影響，但 evaluation RMSE 不準確

## 問題 2: 缺乏完善評估框架

- **沒有 walk-forward backtesting 結果** — 單次 train/test split 的 RMSE 無法反映穩定性
- **沒有 baseline 比較** — 缺少 naive forecast（last value repeat）作為下界
- **Test RMSE ≠ forecast accuracy** — test set 是歷史資料，future prediction 是 out-of-sample

## Fix Results (2026-04-05)

### Changes Applied

1. **sklearn**: Close-only pipeline + instance normalization (RevIN-like)
   - Feature count: 1,134 → 42
   - 每個滑動窗口獨立 normalize，避免 GradientBoosting extrapolation 問題
   - 移除 target_scaler，改用 per-instance mean/std
2. **Lightning**: 新增 RevIN (Reversible Instance Normalization)
   - 與 HuggingFace 的 `scaling='std'` 等效
3. **Evaluation**: `_evaluate_model` 加入 Close 欄位到 eval data

### After Fix

| Model       | RMSE (before) | RMSE (after) | Improvement |
| ----------- | ------------- | ------------ | ----------- |
| HuggingFace | 31            | 31           | baseline    |
| Lightning   | 875           | **83**       | 90%         |
| sklearn     | 888           | **224**      | 75%         |

三個模型預測都在合理範圍（Close price ~1800 附近）。

## Remaining Action Items

1. **加入 naive baseline** — 作為最低評估標準
2. **啟用 walk-forward backtesting** — 驗證模型在不同市場環境下的穩定性
