# MASE/MDA Visualization Integration

- **Date**: 2026-04-05 20:57
- **Status**: completed
- **Module**: visualization/, reporting/, main.py

## 目標

將已實作的 MASE/MDA metrics 整合進所有視覺化輸出（Matplotlib、Plotly、Console/Markdown），並為 naive baseline 加上特殊樣式。

## 背景

#0250 (Naive Baseline + MASE) 和 #0280 (MDA) 已完成，metrics 在 predictor/comparer 中計算，但視覺化層尚未展示：

1. Matplotlib bar chart 缺 MASE
2. Plotly 只有 RMSE 標註，缺 MASE/MDA 及 metrics 子圖
3. Console 輸出缺 MDA
4. Naive 模型在圖表中無特殊 baseline 樣式

## 影響範圍

| 檔案 | 修改 |
| --- | --- |
| `src/currency_predictor/visualization/visualizer.py` | bar chart +MASE; naive baseline 樣式 |
| `src/currency_predictor/visualization/interactive.py` | legend +MASE/MDA; metrics 子圖 |
| `src/currency_predictor/reporting/formatter.py` | console/Markdown +MDA |
| `main.py` | 比較模式產生 interactive HTML |
| `tests/test_visualizer.py` | 更新 metrics dict |
| `tests/test_interactive_visualizer.py` | 更新 metrics dict |
| `tests/test_result_formatter.py` | 驗證 MDA 欄位 |

## 實作步驟

1. Matplotlib bar chart 加入 MASE metric key
2. Naive 在 plot_model_comparison / plot_forecast 使用灰色虛線樣式
3. Formatter console/Markdown 加入 MDA
4. Plotly legend 加 MASE/MDA，有 metrics 時加 bar 子圖
5. main.py 比較模式也產生 interactive HTML
6. 更新相關測試

## 風險評估

- 低風險：僅修改顯示層，不影響模型訓練/預測邏輯
- Plotly 為 optional dependency，需 graceful fallback

## 完成標準

- 全部測試通過
- Matplotlib 比較圖顯示 5 個 metrics (含 MASE)
- Naive 在圖表中以灰色虛線呈現
- Console/Markdown 報告顯示 MASE + MDA
- Plotly HTML 包含 metrics 子圖
