# Forecast Chart + prediction_horizon 15 天

- **Date**: 2026-04-05 18:00
- **Status**: completed
- **Module**: visualization, prediction/comparer, main.py, config

## 目標

新增一張 **forecast chart**，展示「最新交易日前 15 天的 actual 資料 + 後 15 天的多模型預測」。同時將 `prediction_horizon` 從 30 天改為 15 天。

既有圖表（dashboard + comparison chart）保留不動。

## 問題分析

目前的 comparison chart (`plot_model_comparison`) 將 future predictions 疊在 test split actual 上——兩者時間軸不同步，意義不大。

Forecast chart 以最新交易日為分界，左邊是 actual close prices，右邊是各模型的未來預測。

## 影響範圍

| 檔案 | 修改內容 |
|------|----------|
| `config.json` | `prediction_horizon: 30→15`, `pred_len: 30→15` |
| `src/currency_predictor/prediction/comparer.py` | `compare()` 和 `predict_only()` 額外存 `prediction_dates`, `last_known_date`, `last_known_value` |
| `src/currency_predictor/visualization/visualizer.py` | 新增 `plot_forecast()` 方法 |
| `main.py` | 新增 `_generate_forecast_charts()` 函數，在 visualize 時呼叫 |
| `tests/test_visualizer.py` | 新增 `plot_forecast()` 單元測試 |

## 實作步驟

### Step 1: config.json — 改預設

- `prediction_horizon`: 30 → 15
- `model_params.pred_len`: 30 → 15

### Step 2: comparer.py — 擴充回傳結構

在 `compare()` 和 `predict_only()` 的 model result 中增加：

```python
model_result['prediction_dates'] = pred_result.get('prediction_dates')
model_result['last_known_date'] = pred_result.get('last_known_date')
model_result['last_known_value'] = pred_result.get('last_known_value')
```

### Step 3: visualizer.py — 新增 `plot_forecast()`

- 左半：actual close prices（黑色實線 + 圓點，最近 15 天）
- 右半：每個模型的 predictions（虛線 + 不同顏色）
- 垂直虛線 at `last_known_date`，標註 "Latest"
- 連接點：`last_known_value` 同時出現在 actual 尾端和 prediction 起點

### Step 4: main.py — 新增 `_generate_forecast_charts()`

- 從 DataStorage 載入原始資料取最近 15 天 close
- 從 comparison_results 取各模型 predictions + dates
- 呼叫 `visualizer.plot_forecast()`
- 在 `_run_compare_mode()` 的 visualize 區塊中呼叫

### Step 5: 測試

- `tests/test_visualizer.py` 新增 forecast chart 測試

## 產出檔案結構

```
results/latest/figures/
├── USDTWD_dashboard.png           # 既有
├── USDTWD_model_comparison.png    # 既有
└── USDTWD_forecast.png            # 新增
```

## 完成標準

- `prediction_horizon` 和 `pred_len` 改為 15
- Forecast chart 顯示 15 天 actual + 15 天 predicted
- 圖表有分界線標示最新交易日
- 既有 dashboard 和 comparison chart 不受影響
- 所有 tests 通過
