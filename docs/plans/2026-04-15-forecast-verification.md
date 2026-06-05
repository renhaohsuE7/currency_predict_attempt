# Plan: Forecast Verification — Out-of-Sample Prediction Accuracy Check

- **Date**: 2026-04-15
- **Status**: completed
- **Module**: data/collectors, prediction/comparer, prediction/pipeline, verification (new), main CLI

## Context

目前 pipeline 跑完後，`predictions_*.csv` 的 "Actual" 欄位其實是 test set 的 `y_test[-n:]`，不是真正的未來實際資料。當模型預測未來 N 天後，user 無法在 N 天後驗證預測準確度。

需要：
1. Pipeline 結束時，儲存 **純粹的 out-of-sample forecast**（不含 test set "Actual"）
2. N 天後，user 可以下載新資料，跟之前的 forecast 比較
3. 這個功能要做成 **module + CLI command**

## 影響範圍

| 檔案 | 修改 |
| --- | --- |
| `src/currency_predictor/data/collectors.py` | 新增 `get_currency_data_range()` |
| `src/currency_predictor/prediction/comparer.py` | 新增 `save_forecast_json()` |
| `src/currency_predictor/prediction/pipeline.py` | 新增 `save_forecast_json()` |
| `src/currency_predictor/verification/__init__.py` | **新模組** |
| `src/currency_predictor/verification/verifier.py` | **新檔** — `ForecastVerifier` |
| `main.py` | 新增 `--verify`, `--run` flags + `_run_verify_mode()` |
| `tests/test_forecast_verification.py` | **新檔** |

## 實作步驟

1. `collectors.py` — 新增 `get_currency_data_range(symbol, start_date, end_date)`
2. `comparer.py` + `pipeline.py` — 新增 `save_forecast_json()`，所有模式都存 forecast JSON
3. `verification/verifier.py` — `ForecastVerifier` 類別：載入 forecast → 下載 actual → 比對 metrics
4. `main.py` — `--verify` CLI + `_run_verify_mode()`
5. 測試
6. 跑全部測試

## 完成標準

- [ ] `get_currency_data_range()` 可下載指定日期範圍資料
- [ ] Pipeline 完成時自動存 `forecast_{SYMBOL}.json`
- [ ] `ForecastVerifier.verify()` 可對比 forecast vs actual
- [ ] `--verify` CLI 正常運作
- [ ] 非交易日正確處理
- [ ] 部分覆蓋情況正確處理
- [ ] 全部測試通過
