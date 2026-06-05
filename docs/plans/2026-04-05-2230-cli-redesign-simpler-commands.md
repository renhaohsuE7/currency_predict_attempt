# CLI Redesign — Simpler Commands + `--days`

- **Date**: 2026-04-05 22:30
- **Status**: completed
- **Module**: main.py, prediction/comparer.py, prediction/pipeline.py

## 目標

簡化 CLI 命名並新增 `--days` 參數覆蓋 config 中的 `prediction_horizon`。

## 影響範圍

| 檔案 | 修改 |
| --- | --- |
| `main.py` | `--predict`/`--train`/`--days` 新增；op_type 改為 `"train"`/`"predict"` |
| `src/currency_predictor/prediction/comparer.py` | `predict_only()` → `predict()`（加 evaluate）；`train_only()` → `train()` |
| `src/currency_predictor/prediction/pipeline.py` | `run_predict_only()` → `run_predict()`；`run_train_only()` → `run_train()` |
| `src/currency_predictor/prediction/run_manager.py` | `get_latest_model_path` 接受 `"train"` op_type |
| `tests/test_e2e_cli.py` | 新增 `TestNewShortFlags` 測試類 |
| `tests/test_model_comparer.py` | 更新 `operation_type` 斷言 |

## 實作步驟

1. CLI: `--predict`/`--train` 為新名稱，`--predict-only`/`--train-only` 保留為隱藏 alias
2. `--days N` 覆蓋 config 中的 `prediction_horizon`
3. Method rename + 舊名保留為 alias
4. `comparer.predict()` 新增 evaluate（含 MASE/MDA metrics）

## 完成標準

- 559 tests passed
- `--predict --days 60` 可正常執行並顯示 metrics
