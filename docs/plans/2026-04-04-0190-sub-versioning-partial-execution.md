# Plan #0190: Sub-versioning + Partial Execution

- **Date**: 2026-04-04
- **Status**: completed
- **Module**: prediction/run_manager, prediction/pipeline, prediction/comparer, main

## 目標

在 RunManager 的 run 目錄內加入操作子目錄（operation sub-directory），讓每次 train/predict 操作的產出互不覆蓋。同時新增 `--train-only` / `--predict-only` CLI flags 支援部分執行。

## 問題分析

| 現狀 | 問題 |
| --- | --- |
| run 目錄內產出每次覆蓋 | 重跑 train 會覆蓋舊模型，無法比較不同次訓練 |
| Pipeline 只有 `run_full_pipeline()` | 無法只跑 train 或只跑 predict |
| Comparer 不儲存模型 | 比較模式訓練的模型無法復用 |

## 設計決策

| 問題 | 決策 |
| --- | --- |
| 如何隔離操作產出？ | 每次操作在 run 目錄內建立 `{op_id}/` 子目錄（8 字元 UUID） |
| 檔名是否加 suffix？ | 不需要，子目錄已提供隔離，檔名不變 |
| `results/latest` 指向？ | 最新操作目錄（直接存取最新產出） |
| 如何追蹤操作歷史？ | `manifest.json` 在 run 目錄根部 |
| predict-only 如何載入模型？ | 從 manifest 反向掃描找最新訓練操作的模型 |

## 影響範圍

| 檔案 | 變更 |
| --- | --- |
| `src/currency_predictor/prediction/run_manager.py` | op_id/op_dir、manifest、continue_latest、update_latest_symlink |
| `src/currency_predictor/prediction/pipeline.py` | run_train_only、run_predict_only |
| `src/currency_predictor/prediction/comparer.py` | 儲存模型、train_only、predict_only |
| `main.py` | --train-only、--predict-only、--continue、--use-op CLI flags |
| `tests/test_run_manager.py` | operation lifecycle、manifest、continue_latest |
| `tests/test_prediction_pipeline.py` | partial execution tests |
| `tests/test_model_comparer.py` | model saving、partial execution tests |
| `tests/test_e2e_cli.py` | CLI flag tests |

## 實作步驟

1. RunManager — op_id/op_dir + manifest + continue_latest + update_latest_symlink
2. RunManager tests
3. Pipeline — run_train_only、run_predict_only
4. Pipeline tests
5. Comparer — 儲存模型、train_only、predict_only
6. Comparer tests
7. main.py — CLI flags + 操作生命週期 + op_dir routing
8. CLI tests
9. 全回歸測試

## 完成標準

- [ ] 每次操作產出在獨立的 `{op_id}/` 子目錄
- [ ] `manifest.json` 正確追蹤操作歷史
- [ ] `results/latest` 指向最新操作目錄
- [ ] `--train-only` 只跑 collect + train
- [ ] `--predict-only` 載入既有模型並預測
- [ ] `--continue` 在既有 run 內繼續
- [ ] `--use-op` 指定使用特定操作的模型
- [ ] Comparer 訓練後儲存模型
- [ ] 全部既有 tests 通過 + 新增 tests 通過
