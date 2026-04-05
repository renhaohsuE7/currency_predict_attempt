# Plan #0180: One-Command Pipeline + Data Validation + Run Versioning

- **Date**: 2026-04-04
- **Status**: completed
- **Module**: main, prediction, data

## 目標

簡化完整工作流程：使用者只需填 config.json，一個指令完成所有步驟。同時加入智慧資料快取驗證和按時間戳組織每次執行結果。

## 問題分析

| 現狀 | 問題 |
| --- | --- |
| 需記住 `--compare -v` flags | 新手不知道該用哪些參數 |
| `force_update=False` 只看檔案存在 | 不驗證快取資料正確性 |
| 模型/報告每次覆蓋 | 無法比較不同次的執行結果 |

## 設計決策

| 問題 | 決策 |
| --- | --- |
| 一鍵指令如何設計？ | `--full` = `--compare -v` 的語意化捷徑 |
| 強制重新開始？ | `--fresh` flag → `force_update=True` + `force_retrain=True` |
| 資料驗證方式？ | 抽樣 3 個日期從 Yahoo Finance 重新下載比對 Close 值 |
| 結果如何組織？ | `results/runs/{YYYYMMDD_HHMMSS}/` + `results/latest` symlink |
| 向後相容？ | 舊的 `results/models/`、`results/figures/` 不影響 |

## 實作完成

### Feature 1: `--full` + `--fresh` CLI Flags

- `main.py` — 新增 `--full`（= `--compare -v`）和 `--fresh`（force re-download + re-train）
- `comparer.py` — `compare()` 新增 `force_update` 參數
- `pipeline.py` — `_collect_data_phase()` 新增 `force_update` 參數，`force_retrain` 連動 `force_update`

### Feature 2: 智慧資料快取驗證

- `collectors.py` — 新增 `get_spot_check_data(symbol, target_date)` 方法
- `predictor.py` — 新增 `_validate_cached_data()` 方法：
  - 從快取資料隨機抽 3 個日期
  - 從 Yahoo Finance 下載 ±3 天的小範圍資料
  - 比較 Close 值，允許 ±0.01 誤差
  - 全部匹配 → 跳過下載；任一不匹配 → 重新下載
  - 網路問題 → 跳過該 sample（不阻塞流程）

### Feature 3: Run Versioning

- `run_manager.py`（新增）— `RunManager` class：
  - `setup(config)` — 建立 `results/runs/{timestamp}/` 目錄結構 + config 快照
  - `models_dir`, `figures_dir` properties
  - `update_latest_symlink()` — 相對路徑 symlink
  - `list_runs()` — 列出歷史 runs
- `main.py` — 所有產出（報告、模型、圖表）寫入 run 目錄
- `pipeline.py` — 檔名不再需要時間戳後綴（run 目錄已含）

## 新增檔案

| 檔案 | 說明 |
| --- | --- |
| `src/currency_predictor/prediction/run_manager.py` | RunManager class |
| `tests/test_run_manager.py` | 13 tests |
| `tests/test_data_validation.py` | 9 tests |

## 修改檔案

| 檔案 | 變更 |
| --- | --- |
| `main.py` | `--full`, `--fresh` flags + RunManager 整合 |
| `prediction/__init__.py` | 匯出 RunManager |
| `prediction/comparer.py` | `compare()` 新增 `force_update` 參數 |
| `prediction/pipeline.py` | run_dir 整合 + 移除檔名時間戳 |
| `prediction/predictor.py` | `_validate_cached_data()` |
| `data/collectors.py` | `get_spot_check_data()` |
| `tests/test_prediction_pipeline.py` | 適配新介面（移除 timestamp 參數） |
| `README.md` | 更新 Complete Workflow 和 CLI 說明 |

## 測試結果

- **398 passed** (non-slow), 0 failures
- 新增 22 tests（RunManager 13 + data validation 9）

## 完成標準

- [x] `uv run main.py --full` 一個指令完成所有步驟
- [x] `--fresh` flag 強制重新下載 + 重新訓練
- [x] 資料抽樣驗證：修改過的快取資料會被偵測並重新下載
- [x] 每次執行產出在 `results/runs/{datetime}/` 獨立目錄
- [x] `results/latest` symlink 指向最新 run
- [x] 全部既有 tests 通過 + 新增 tests 通過
- [x] README 更新反映新 CLI 用法
