# Plan #0140: Interactive Visualization (Plotly + Dark Mode)

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: visualization/
- **Priority**: P2

## 目標

新增 Plotly 互動式圖表（含技術指標子圖），支援 light/dark 主題切換和 HTML 匯出。

## 現狀

`CurrencyVisualizer` 已有 10 種 matplotlib 靜態圖表：

- price_history, candlestick, volume, price_and_volume
- returns, moving_averages, comparison
- prediction_results, model_comparison, create_dashboard

**缺少**：互動式圖表、dark mode、HTML 匯出。

## 設計決策

| 問題 | 決策 |
| --- | --- |
| 新增 class 或擴展現有 | 新增 `InteractiveVisualizer` class（不改動現有 matplotlib 程式碼） |
| Plotly 版本 | plotly>=5.0.0（新增 optional dependency） |
| 互動式圖表種類 | candlestick + volume + RSI + MACD + Bollinger Bands（多子圖）、prediction overlay |
| 主題系統 | `ThemeManager` — light/dark preset + 自訂 |
| 匯出格式 | HTML（primary），使用 CDN plotlyjs |
| CLI flags | 無 — 專案無 CLI entrypoint，theme 直接透過 constructor/set_theme() |
| Dash dashboard | 不在此 plan 範圍 |

## 實作摘要

### 新增檔案

| 檔案 | 說明 |
| --- | --- |
| `visualization/themes.py` | `ThemeManager` class — light/dark presets，提供 `layout_defaults()`、`line_color(i)`、up/down color 等 |
| `visualization/interactive.py` | `InteractiveVisualizer` class — `plot_technical_analysis()`、`plot_prediction_comparison()`、`_save_html()` |
| `tests/test_interactive_visualizer.py` | 19 tests covering ThemeManager + InteractiveVisualizer |

### 修改檔案

| 檔案 | 改動 |
| --- | --- |
| `pyproject.toml` | `[project.optional-dependencies] interactive = ["plotly>=5.0.0"]` |
| `visualization/__init__.py` | 條件匯出 `InteractiveVisualizer`（plotly 不存在時 fallback 為 None）+ 匯出 `ThemeManager` |

### InteractiveVisualizer 功能

- **`plot_technical_analysis(df, symbol, save_path)`**：多子圖互動式圖表
  - Row 1: Candlestick + Bollinger Bands (SMA + 2σ)
  - Row 2: Volume bar chart（如有 Volume 欄位）
  - Row 3: RSI (adaptive window) + MACD/Signal（legend-only toggle）
  - 自動調整子圖數量（有/無 Volume）

- **`plot_prediction_comparison(actual, model_predictions, symbol, metrics, save_path)`**：多模型預測比較
  - Actual 線 + 各模型預測線（dash + markers）
  - 若提供 metrics 則在 legend 顯示 RMSE

### ThemeManager

- Light: plotly_white, #26a69a/#ef5350 up/down
- Dark: plotly_dark, #00e676/#ff5252 up/down
- 8-color line palette per theme, auto-cycling via `line_color(index)`
- `available_themes()` static method

## 測試結果

- **19 new tests**: all passed
- **311 passed total** (excluding pre-existing HF sandbox permission issue)
- 既有 matplotlib 圖表測試不受影響

## 完成標準

- [x] `InteractiveVisualizer.plot_technical_analysis()` 產出含 RSI/MACD/BB 的互動式圖表
- [x] `InteractiveVisualizer.plot_prediction_comparison()` 產出多模型互動比較圖
- [x] `ThemeManager("dark")` 切換暗色主題
- [x] HTML 檔匯出可在瀏覽器開啟
- [x] 既有 matplotlib 圖表不受影響
