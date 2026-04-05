# Fix ResultFormatter Stage Status for predict-only / train-only

- **Date**: 2026-04-05 17:01
- **Status**: completed
- **Module**: prediction/pipeline, reporting/formatter

## 目標

修正 `--predict-only` 和 `--train-only` 模式下 ResultFormatter 顯示所有 stage 為 `[FAIL]` 的問題。

## 問題

`run_predict_only()` 和 `run_train_only()` 回傳的 results dict 沒有 `pipeline_status` key。`format_stage_status()` 讀取 `results.get('pipeline_status', {})` 得到空 dict，所有 stage 的 `.get()` 回傳 default False → 全顯示 `[FAIL]`。

## 影響範圍

| 檔案 | 修改內容 |
|------|----------|
| `src/currency_predictor/prediction/pipeline.py` | `run_predict_only()` 和 `run_train_only()` 加入 `pipeline_status` |
| `tests/test_prediction_pipeline.py` | 驗證 `pipeline_status` 結構 |

## 實作

為 `run_predict_only()` 和 `run_train_only()` 加入 `pipeline_status` dict：

- 不執行的階段視為通過（True）
- 實際執行的階段在完成後更新狀態
- `success` 改為 `all(pipeline_status.values())`

### Comparison chart 說明

`plot_model_comparison()` 只在 `--compare` 模式才有意義（需要多模型 predictions + actual series）。單一模型模式只產生 dashboard，這是預期行為。

## 驗證

- 539 tests passed
