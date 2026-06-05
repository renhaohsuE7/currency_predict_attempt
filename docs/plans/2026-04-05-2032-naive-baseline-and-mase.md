# Naive Baseline Models + MASE Metric

- **Date**: 2026-04-05 20:32
- **Status**: completed
- **Module**: models/, prediction/, reporting/
- **Reference**: [docs/reference/production-forecasting-best-practices.md](../reference/production-forecasting-best-practices.md) §1, §3

## 目標

加入 Naive baseline model 和 MASE metric，讓每次模型比較都能回答：「模型有比最簡單的預測方法好嗎？」

## 背景

- 目前只有 RMSE / MAE，無法判斷模型是否比「重複最後一個值」更好
- sklearn RMSE=224 看似不差，但如果 naive RMSE=50 那就代表 sklearn 反而更爛
- MASE (Mean Absolute Scaled Error) 直接以 naive forecast 為分母，MASE < 1.0 代表贏過 naive

## 影響範圍

| 檔案 | 修改內容 |
| ---- | -------- |
| `src/currency_predictor/models/naive.py` | 新增 NaiveModel（persistence + drift） |
| `src/currency_predictor/models/factory.py` | 註冊 naive model |
| `src/currency_predictor/prediction/metrics.py` | 新增 MASE, MDA 計算函式 |
| `src/currency_predictor/prediction/predictor.py` | `_evaluate_model` 加入 MASE |
| `src/currency_predictor/reporting/formatter.py` | 比較報告加入 MASE 欄位 |
| `tests/test_naive_model.py` | Naive model 測試 |
| `tests/test_metrics.py` | MASE / MDA 測試 |

## 實作步驟

### Step 1: NaiveModel

- 實作 `BaseModel` ABC
- `predict(X)`: 回傳 `X['Close'].iloc[-1]` 重複 pred_len 次
- 不需要 `fit()`（或 fit 為 no-op）
- 可選：加入 drift variant（最後一個值 + 線性趨勢）

### Step 2: Metrics module

- `mase(y_true, y_pred, y_train)`: MAE / naive_MAE
- `mda(y_true, y_pred)`: 方向準確度（漲跌是否一致）
- 純函式，不依賴 model

### Step 3: 整合到 pipeline

- `_evaluate_model` 新增 MASE 輸出
- `ResultFormatter` 比較表加入 MASE 欄位
- `ModelFactory` 新增 `naive` 選項，讓 multi-model comparison 自動包含 naive

## 完成標準

- `NaiveModel` 通過所有 BaseModel 介面測試
- MASE 計算正確（naive 的 MASE 恆為 1.0）
- 比較報告顯示 MASE 欄位
- 所有現有 tests 通過
