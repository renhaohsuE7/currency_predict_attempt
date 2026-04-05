# Plan #0150: PDF Report Generation

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: reporting/
- **Priority**: P2

## 目標

將預測報告（文字 + 圖表）匯出為 PDF 文件，方便分享和存檔。

## 現狀

- `ResultFormatter` 產出 Markdown 文字報告（`generate_report()`, `generate_comparison_report()`）
- `CurrencyVisualizer` 產出 PNG 圖表
- **無** PDF 匯出功能

## 設計決策

| 問題 | 決策 |
| --- | --- |
| PDF library | fpdf2（輕量、純 Python、無系統依賴） |
| 報告內容 | 標題頁 + pipeline status + 預測摘要表 + 多模型比較表 + 圖表嵌入 |
| 中文支援 | 未加入 — fpdf2 支援 TTF 載入但此版本使用 Helvetica（未來可擴展） |
| API 設計 | `ReportGenerator.generate_pdf(results, charts, output_path)` |
| CLI flags | 無 — 專案無 CLI entrypoint |

## 實作摘要

### 新增檔案

| 檔案 | 說明 |
| --- | --- |
| `reporting/pdf_generator.py` | `ReportGenerator` class — `generate_pdf()` + internal helpers |
| `tests/test_pdf_generator.py` | 13 tests covering pipeline/comparison/chart/edge cases |

### 修改檔案

| 檔案 | 改動 |
| --- | --- |
| `pyproject.toml` | `[project.optional-dependencies] pdf = ["fpdf2>=2.7.0"]` |
| `reporting/__init__.py` | 條件匯出 `ReportGenerator`（fpdf2 不存在時 fallback 為 None） |

### ReportGenerator API

```python
gen = ReportGenerator(output_dir="results/reports")
path = gen.generate_pdf(
    results=pipeline_results,      # or comparison_results
    charts=["chart1.png", "chart2.png"],
    output_path="report.pdf",
)
```

PDF 內容：

- **Title page**: 報告標題 + 生成時間 + overall status
- **Pipeline stages**: data_collection / model_training / prediction / results_saved 狀態表
- **Predictions table**: Symbol / Last Price / Predicted / Change (%)
- **Comparison tables**: per-symbol model metrics (RMSE/MAE/Time) + best model + overall ranking
- **Charts**: 嵌入 PNG/JPG 圖檔，自動跨頁

## 測試結果

- **13 new tests**: all passed
- **324 passed total** (excluding pre-existing HF sandbox permission issue)

## 完成標準

- [x] `ReportGenerator.generate_pdf()` 產出可開啟的 PDF
- [x] PDF 包含預測摘要、metrics 表格、圖表
- [x] Pipeline results 和 comparison results 都能處理
- [x] 既有測試不受影響
