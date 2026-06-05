# Save Predictions & Metrics CSV for Compare Mode

- **Date**: 2026-04-05 22:00
- **Status**: completed
- **Module**: prediction/comparer.py, main.py

## 目標

在 compare mode 的結果目錄下額外輸出 predictions CSV 和 metrics CSV，讓事後分析、歷史比較不需要重跑 pipeline。

## 影響範圍

| 檔案 | 修改 |
| --- | --- |
| `src/currency_predictor/prediction/comparer.py` | 新增 `save_predictions_csv()` 和 `save_metrics_csv()` |
| `main.py` | `_run_compare_mode()` 呼叫新方法 |
| `tests/test_comparer_csv.py` | 新增 CSV 儲存測試 |

## 實作步驟

1. ModelComparer 新增 `save_predictions_csv(comparison_results, output_dir)` — wide format，每 symbol 一個檔
2. ModelComparer 新增 `save_metrics_csv(comparison_results, output_dir)` — long format，所有 symbol/model 合併
3. `_run_compare_mode()` 在報告儲存後呼叫兩個方法
4. 新增單元測試

## 風險評估

- 低風險：純新增功能，CSV 寫入失敗用 try/except 保護，不影響現有流程

## 完成標準

- 新增測試全部通過
- 預設 pipeline 跑完後 `results/runs/` 下有 `predictions_*.csv` 和 `metrics.csv`
