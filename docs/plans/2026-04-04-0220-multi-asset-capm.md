# Plan #0220: Multi-Asset Support + CAPM 準備

- **Date**: 2026-04-04
- **Status**: completed
- **Module**: utils, config, data_processor, prediction

## 目標

新增 asset type 分類機制（forex/stock/crypto/index），並為股市支援準備 CAPM 特徵工程（Beta、Alpha、Sharpe Ratio）。CAPM 預設 disabled，不影響現有 forex 流程。

## 影響範圍

| 檔案 | 變更 |
|------|------|
| `src/currency_predictor/utils/asset_type.py` | **新增** AssetType enum + classify_symbol() |
| `src/currency_predictor/config/settings.py` | 新增 CAPMConfig model |
| `src/currency_predictor/data_processor.py` | 新增 `create_capm_features()` method |
| `src/currency_predictor/prediction/predictor.py` | 新增 asset-type routing + market data helpers |
| `config.json` / `config_example.json` | 新增 `capm` 區塊（disabled by default） |
| `tests/test_data_processor.py` | 新增 CAPM feature tests |
| `tests/test_asset_type.py` | **新增** classify_symbol tests |

## 實作步驟

1. 新增 `utils/asset_type.py` — AssetType enum + classify_symbol()
2. 新增 `CAPMConfig` 到 settings.py
3. 新增 `create_capm_features()` 到 DataProcessor
4. 修改 predictor.py — asset-type routing + market data helpers
5. 更新 config.json / config_example.json
6. 新增測試
7. Docker 跑全套測試驗證

## 相容性設計

- CAPM 預設 disabled — 不影響現有 forex 流程
- classify_symbol() 是 pure function — 不修改現有 collector/storage 介面
- create_capm_features() 是 additive — 只新增 columns，不影響既有 features
- Models 不需修改 — 接受任何 feature columns
- Forex 路徑完全不變 — asset_type == FOREX 時跳過所有 CAPM 邏輯

## 完成標準

- classify_symbol() 正確分類 forex/stock/crypto/index
- CAPMConfig 可透過 config.json 設定
- create_capm_features() 計算 Beta, Alpha, Sharpe
- CAPM disabled 時不影響 forex 流程
- 全套測試通過（含新增 CAPM 測試）
