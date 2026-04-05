# Plan #0240: Backtesting Framework + Data Period 優化

- **Date**: 2026-04-04
- **Status**: completed
- **Module**: backtesting, config, prediction, visualization, reporting

## 目標

建立 walk-forward backtesting 框架，系統性驗證模型在不同市場階段的預測能力。同時修正資料期間設定不一致問題。

## 影響範圍

### 新增

| 檔案 | 說明 |
|------|------|
| `src/currency_predictor/backtesting/__init__.py` | 模組匯出 |
| `src/currency_predictor/backtesting/splitter.py` | WalkForwardSplitter |
| `src/currency_predictor/backtesting/metrics.py` | FinancialMetrics |
| `src/currency_predictor/backtesting/runner.py` | BacktestRunner |
| `src/currency_predictor/backtesting/result.py` | FoldResult, BacktestResult |
| `tests/test_backtest_splitter.py` | splitter 單元測試 |
| `tests/test_backtest_metrics.py` | metrics 單元測試 |
| `tests/test_backtest_runner.py` | runner 測試 (mock model) |

### 修改

| 檔案 | 變更 |
|------|------|
| `config/settings.py` | 新增 BacktestConfig |
| `config.json` | data period 3y + backtest 區塊 |
| `prediction/predictor.py` | split ratio 可配置 |
| `visualization/visualizer.py` | equity curve, drawdown charts |
| `reporting/formatter.py` | backtest report |
| `main.py` | --backtest CLI flag |

## 實作步驟

1. 建立 plan doc
2. Phase 1: Core (splitter + metrics + result + tests)
3. Phase 2: Runner + tests
4. Phase 3: Config + period fixes
5. Phase 4: CLI --backtest
6. Phase 5: Visualization
7. Phase 6: Reporting
8. Docker 全套測試

## 完成標準

- Walk-forward rolling/expanding 正確無 data leakage
- Financial metrics 計算正確
- --backtest CLI 可用
- 既有 472 tests 不受影響
