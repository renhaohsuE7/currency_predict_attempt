# Directional Accuracy (MDA) Metric

- **Date**: 2026-04-05 20:35
- **Status**: completed
- **Module**: prediction/
- **Reference**: [docs/reference/production-forecasting-best-practices.md](../reference/production-forecasting-best-practices.md) §1

## 目標

新增 MDA (Mean Directional Accuracy) metric，衡量模型預測漲跌方向的準確度。

## 背景

- RMSE/MAE 只衡量幅度誤差，但在 trading 場景中，方向比幅度更重要
- 一個 RMSE=100 但方向 80% 正確的模型，可能比 RMSE=50 但方向 50% 的模型更有價值
- MDA = 預測方向正確的比例（> 50% 代表比隨機猜好）

## 影響範圍

| 檔案 | 修改內容 |
| ---- | -------- |
| `src/currency_predictor/prediction/metrics.py` | 新增 `mda()` 函式（若 #0250 已建立此檔案，直接加入） |
| `src/currency_predictor/prediction/predictor.py` | `_evaluate_model` 加入 MDA |
| `src/currency_predictor/reporting/formatter.py` | 比較報告加入 MDA 欄位 |
| `tests/test_metrics.py` | MDA 測試 |

## 實作步驟

### Step 1: MDA 計算

```python
def mda(y_true, y_pred):
    """Mean Directional Accuracy: 預測漲跌方向的正確率"""
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return np.mean(true_direction == pred_direction)
```

### Step 2: 整合到 evaluation pipeline

- `_evaluate_model` 回傳新增 `mda` 欄位
- `ResultFormatter` 比較表加入 MDA 欄位

## 完成標準

- MDA 計算正確（全漲/全跌/混合 case）
- 比較報告顯示 MDA 欄位
- 所有現有 tests 通過

## 備註

- 此 plan 可與 #0250 合併實作（MASE + MDA 在同一個 metrics module）
