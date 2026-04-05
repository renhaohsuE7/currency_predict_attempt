# Plan #0230: Stock E2E Validation

- **Date**: 2026-04-04
- **Status**: completed
- **Module**: prediction, tests

## 目標

修復 CAPM config 傳遞斷鏈（pipeline/comparer 未傳 capm_config），並用真實股票資料跑完整 E2E 驗證。

## 影響範圍

| 檔案 | 變更 |
|------|------|
| `src/currency_predictor/prediction/pipeline.py` | 傳 `capm_config` 到 CurrencyPredictor |
| `src/currency_predictor/prediction/comparer.py` | 傳 `capm_config` 到 CurrencyPredictor |
| `tests/test_e2e_stock.py` | **新增** Stock + CAPM E2E 測試 |

## 實作步驟

1. 修復 pipeline.py — 傳 `capm_config`
2. 修復 comparer.py — 傳 `capm_config`
3. 新增 `tests/test_e2e_stock.py`（@pytest.mark.e2e）
4. Docker 跑全套測試驗證

## 完成標準

- capm_config 完整傳遞至 CurrencyPredictor
- AAPL E2E: collect → CAPM features → train → predict 成功
- Forex 不受影響
- 全套 non-e2e 測試通過
