# Prediction Intervals / Uncertainty Quantification

- **Date**: 2026-04-05 20:34
- **Status**: draft
- **Module**: models/, prediction/
- **Reference**: [docs/reference/production-forecasting-best-practices.md](../reference/production-forecasting-best-practices.md) §5

## 目標

以 Conformal Prediction 為基礎，為所有 PatchTST backend 提供具統計保證的 prediction intervals，取代目前 `predict_with_uncertainty` 中 `std * 0.1` 的假設。

## 背景

- 目前 sklearn 的 `predict_with_uncertainty` 用 `np.std(predictions) * 0.1` 估計不確定性，毫無統計根據
- 點預測（"明天 1800"）對決策者不夠有用，需要信賴區間（"明天 1750-1850, 90% confidence"）
- Conformal Prediction 不需要分布假設，保證 coverage rate

## 影響範圍

| 檔案 | 修改內容 |
| ---- | -------- |
| `src/currency_predictor/prediction/conformal.py` | 新增 `ConformalPredictor` |
| `src/currency_predictor/models/patchtst/sklearn/model.py` | 改寫 `predict_with_uncertainty` |
| `src/currency_predictor/visualization/visualizer.py` | forecast chart 加入 confidence band |
| `tests/test_conformal.py` | Conformal prediction 測試 |

## 實作步驟

### Step 1: ConformalPredictor

- 使用 MAPIE（scikit-learn-contrib）或自製 split conformal
- 在 walk-forward backtesting 的 calibration set 上計算 nonconformity scores
- 根據指定 confidence level（90%, 95%）產出 prediction intervals

### Step 2: 整合到 predict_with_uncertainty

- 取代 `std * 0.1` 假設
- 回傳 `{'predictions', 'lower_bound', 'upper_bound', 'confidence_level'}`

### Step 3: Forecast chart

- 在 forecast chart 加入 shaded confidence band

## 完成標準

- 90% prediction interval 的 empirical coverage ≥ 85%（允許小偏差）
- Forecast chart 顯示 confidence band
- 不依賴分布假設
- 所有現有 tests 通過
