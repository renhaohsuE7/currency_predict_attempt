# Look-Ahead Bias Audit

- **Date**: 2026-04-05 20:36
- **Status**: draft
- **Module**: data_processor, prediction/
- **Reference**: [docs/reference/production-forecasting-best-practices.md](../reference/production-forecasting-best-practices.md) §6

## 目標

審核整個 data pipeline 的 feature engineering，確認沒有 look-ahead bias（未來資訊洩漏）。

## 背景

- `DataProcessor.create_technical_indicators()` 計算 RSI, MACD, Bollinger Bands 等
- `DataProcessor.create_lagged_features()` 建立滯後特徵
- 若這些操作使用了未來資料（如 rolling window 包含當前或未來值），模型的 test RMSE 會虛假地好看
- 常見的 leakage 來源：
  - `rolling().mean()` 沒有先 `.shift(1)` → 包含當前觀測值
  - `fillna(method='bfill')` → 用未來值填補過去的缺失值
  - Scaler 在全部資料上 fit 再 split → test set 的統計已洩漏
  - 技術指標（如 EMA）使用整個序列計算而非只用歷史資料

## 影響範圍

| 檔案 | 審核內容 |
| ---- | -------- |
| `src/currency_predictor/data_processor.py` | 所有 rolling/shift/fillna/indicator 操作 |
| `src/currency_predictor/prediction/predictor.py` | train/test split 順序 vs scaler fit 順序 |
| `src/currency_predictor/models/patchtst/*/model.py` | 各 backend 內部 scaler 是否只用 train data fit |

## 實作步驟

### Step 1: 逐一審核 DataProcessor 的操作

- `create_technical_indicators()`: 檢查 RSI, MACD, MA, Bollinger, ATR 的實作
- `create_lagged_features()`: 檢查 shift 方向和 fillna 方法
- 確認所有 rolling 操作用 `shift(1)` 或只看歷史窗口

### Step 2: 檢查 scaler fit 順序

- `prepare_training_data()` 裡 scaler 是在 split 之前還是之後 fit？
- 各 model 的 `fit()` 裡 scaler 是否只用 training data？

### Step 3: 修復發現的問題

- 修改有 leakage 風險的操作
- 新增測試確認修復

## 完成標準

- 所有 rolling features 確認只使用歷史資料
- 所有 fillna 確認只用 forward-fill（不用 backfill）
- Scaler 確認只在 training data 上 fit
- 審核結果文件化
