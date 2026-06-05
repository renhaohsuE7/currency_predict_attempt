# Drift Monitoring

- **Date**: 2026-04-05 20:37
- **Status**: draft
- **Module**: prediction/
- **Reference**: [docs/reference/production-forecasting-best-practices.md](../reference/production-forecasting-best-practices.md) §4

## 目標

實作 drift detection 機制，當模型效能下降或資料分布改變時自動發出警告。

## 背景

- 金融市場有明顯的 regime changes（牛市/熊市/盤整/危機），模型在某個 regime 訓練後對新 regime 可能失效
- 目前沒有任何監控機制，模型部署後無法知道何時需要重新訓練
- Drift 分兩類：
  - **Data drift**: input features 的分布改變（例如波動率從低變高）
  - **Concept drift**: input→output 的關係改變（同樣的 features 對應不同的 target）

## 影響範圍

| 檔案 | 修改內容 |
| ---- | -------- |
| `src/currency_predictor/prediction/drift.py` | 新增 `DriftDetector` class |
| `src/currency_predictor/prediction/predictor.py` | 預測後檢查 drift |
| `tests/test_drift.py` | Drift detection 測試 |

## 實作步驟

### Step 1: DriftDetector

- 追蹤 rolling MASE：若超過閾值（如 1.2）表示模型退化到接近 naive
- 追蹤 input feature 分布：用 PSI (Population Stability Index) 或 KS test
- 儲存 reference distribution（training 時的 feature 統計量）

### Step 2: 整合到 prediction pipeline

- 每次預測後計算 drift score
- 若超過閾值，在 log 中發出 warning
- 可選：回傳 drift 資訊在 prediction result 中

### Step 3: 可選 — Evidently AI 整合

- 使用 Evidently AI 產生 drift report（HTML dashboard）
- 需要 `uv add --optional monitoring evidently`

## 完成標準

- 能偵測 input distribution shift（data drift）
- 能偵測 prediction accuracy 下降（concept drift）
- 超過閾值時發出 warning log
- 所有現有 tests 通過

## 依賴

- #0250 (MASE metric) — drift 以 MASE 退化作為觸發指標
