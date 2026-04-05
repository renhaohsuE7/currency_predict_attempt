# Plan #0070: Multi-Model Comparison + Stock Ticker Support

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: prediction, visualization, reporting, config, CLI

## 目標

讓專案支援**多個模型同時預測**同一組 symbol（貨幣對或股票 ticker），並產出比較圖表與排名報告。同時擴展資料收集器以原生支援股票 ticker（AAPL, TSLA 等）。

## 設計決策

| 問題 | 決策 |
| --- | --- |
| 多 model 比較在哪一層？ | 新增 `prediction/comparer.py` — `ModelComparer` class |
| CLI 設計 | `--compare`, `--models`, `--symbols` flags |
| Stock ticker 支援 | `YahooFinanceCollector` 已原生支援，移除 currency-only 限制 |
| 比較圖表 | `CurrencyVisualizer.plot_model_comparison()` — 預測 overlay + metrics bar chart |
| 比較報告 | `ResultFormatter.format_comparison_results()` + `generate_comparison_report()` |

## 實作完成

### Phase 1: Stock Ticker 支援

- `data/collectors.py` — `supported_pairs` → `default_symbols`（保留向後相容 alias）、新增 `is_currency_pair()` 靜態方法
- `prediction/predictor.py` — 新增 `_clean_symbol()` helper，替換所有硬寫 `symbol.replace('=X', '')`
- `config/settings.py` — `symbols` description 更新為支援股票 ticker
- 測試：`test_is_currency_pair`, `test_default_symbols_backward_compat`, `test_clean_symbol` (5 tests)

### Phase 2: ModelComparer 核心

- `prediction/comparer.py` — `ModelComparer` class + `resolve_model_name()` + `MODEL_SHORTNAMES`
- `prediction/__init__.py` — 匯出 `ModelComparer`
- 測試：`test_model_comparer.py` (19 tests)

### Phase 3: 比較圖表

- `visualization/visualizer.py` — `plot_model_comparison()` 方法
- 測試：`TestPlotModelComparison` (4 tests)

### Phase 4: 評估報告

- `reporting/formatter.py` — `format_comparison_results()` + `generate_comparison_report()`
- 測試：`TestComparisonReport` (4 tests)

### Phase 5: Config + CLI 整合

- `config/settings.py` — 新增 `model_names: Optional[List[str]]`
- `config/manager.py` — 新增 `get_model_names()`
- `main.py` — `--compare`, `--models`, `--symbols` flags + `_run_compare_mode()`
- `config.json` — 新增 `"model_names": null`

### Phase 6: 文件更新

- `README.md` — 新增多模型比較 CLI 範例、更新 test count 和模組描述

## 新增檔案

| 檔案 | 說明 |
| --- | --- |
| `src/currency_predictor/prediction/comparer.py` | `ModelComparer` + `resolve_model_name()` |
| `tests/test_model_comparer.py` | 19 tests |
| `docs/plans/2026-04-03-0070-multi-model-comparison.md` | 本文件 |

## 修改檔案

| 檔案 | 變更 |
| --- | --- |
| `data/collectors.py` | `default_symbols` + `is_currency_pair()` |
| `prediction/predictor.py` | `_clean_symbol()` + 金融預測執行器 docstring |
| `prediction/__init__.py` | 匯出 `ModelComparer` |
| `visualization/visualizer.py` | `plot_model_comparison()` |
| `reporting/formatter.py` | `format_comparison_results()` + `generate_comparison_report()` |
| `config/settings.py` | `model_names` field + symbols description |
| `config/manager.py` | `get_model_names()` |
| `main.py` | `--compare`, `--models`, `--symbols` + compare mode |
| `config.json` | `model_names` field |
| `README.md` | 多模型比較、test count 更新 |
| `tests/test_data_collectors.py` | `test_is_currency_pair`, `test_default_symbols_backward_compat` |
| `tests/test_currency_predictor.py` | `TestCleanSymbol` (3 tests) |
| `tests/test_visualizer.py` | `TestPlotModelComparison` (4 tests) |
| `tests/test_result_formatter.py` | `TestComparisonReport` (4 tests) |

## 測試結果

- **270 passed** (non-slow), 0 new failures
- 新增 ~35 tests（comparer 19 + visualizer 4 + formatter 4 + collectors 2 + predictor 3 + config 相關）

## 完成標準

- [x] `uv run main.py` 行為不變（向後相容）
- [x] `uv run main.py --compare --models sklearn` 可執行
- [x] `uv run main.py --symbols AAPL` 能正確處理 stock ticker
- [x] `plot_model_comparison` 畫出多 model overlay 圖 + metrics bar chart
- [x] `format_comparison_results` 產出排名報告
- [x] 全部非 slow 測試通過 (270 passed)
