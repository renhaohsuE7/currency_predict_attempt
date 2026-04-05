# Project Status & Roadmap

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: 全專案

## 專案最終目標

**使用 PatchTST 時間序列模型預測匯率及股票走勢，支援多模型比較，並產出視覺化報告與 PDF 文件。**

核心能力：

1. **資料收集**：從 Yahoo Finance 取得貨幣對（USDTWD=X）及股票 ticker（AAPL）的 OHLCV 資料
2. **多模型預測**：PatchTST sklearn / HuggingFace / Lightning 三種實作，透過 ModelFactory 統一建立
3. **模型比較**：ModelComparer 同時訓練多模型，產出 metrics 排名與比較圖表
4. **視覺化報告**：靜態圖表（matplotlib）、互動式圖表（plotly）、PDF 匯出
5. **CLI 操作**：`uv run main.py --compare --models sklearn,huggingface --symbols USDTWD=X,AAPL -v`

## 本文件目標

盤點專案目前的實作進度，整合散落在各 docs 中的待辦事項，建立統一的後續開發路線圖。

## 已完成

### 基礎建設

- [x] 專案重組：標準化目錄結構、測試框架、文檔分類 (`PROJECT_REORGANIZATION.md`)
- [x] `main.py` 重構：152 → 53 行，抽出 ConfigManager、ResultFormatter (`main_refactoring_report.md`)
- [x] Pydantic Settings 遷移：17 config tests passing (`pydantic_settings_migration_report.md`)
- [x] 測試基礎：12 test files、55+ test cases (`module_testing_report.md`)
- [x] Examples 整理：5 個獨立範例檔 (`example_usage_refactoring.md`)
- [x] Notebook 整理：移至 `notebooks/`，刪除重複 (`notebook_organization.md`)
- [x] Docker 容器化：multi-stage build (prod/dev/jupyter)、CUDA GPU、HF cache volume (`2026-04-03-0060`)
- [x] `.env.example`：記錄可用環境變數

### 功能模組

- [x] 資料收集：YahooFinanceCollector（OHLCV，支援貨幣對 + 股票 ticker）
- [x] 資料處理：DataProcessor（清洗、特徵工程、scaling）
- [x] 設定管理：ConfigManager + Pydantic Settings + 環境變數支援
- [x] 結果格式化：ResultFormatter（JSON / Markdown / 多模型比較報告）
- [x] 視覺化：CurrencyVisualizer，10 種圖表（含多模型比較圖）、中文字體支援
- [x] Pipeline 執行修復：4 個 critical issues (`basic_usage_fixes_summary.md`)
- [x] 多模型比較：`ModelComparer` — 多模型 train/predict/evaluate + 排名（`2026-04-03-0070`）
- [x] CLI 擴展：`--compare`, `--models`, `--symbols` 支援多模型比較和股票 ticker

### 模型

- [x] PatchTST sklearn 版（`models/patchtst/sklearn/`）
- [x] PatchTST HuggingFace 版骨架（`models/patchtst/huggingface/`）
- [x] HuggingFace Pretrained + Fine-Tune（`from_scratch` / `full` / `linear_probe`）
- [x] ModelFactory 動態建立模型

## 未完成 / 待辦

### P0 — 高優先

| 項目 | 來源文件 | 說明 |
| --- | --- | --- |
| ~~HuggingFace PatchTST 整合~~ | `2026-04-03-0040` | ✓ predict/uncertainty bug 修復 + 19 integration tests |
| ~~HuggingFace Pretrained + Fine-Tune~~ | `2026-04-03-0050` | ✓ 3 種 fine-tune 模式 + 6 slow integration tests |
| ~~Docker Containerization~~ | `2026-04-03-0060` | ✓ multi-stage build, CUDA GPU, HF cache volume |
| ~~PyTorch Lightning 版 PatchTST~~ | [`2026-04-03-0090`](2026-04-03-0090-patchtst-lightning.md) | ✓ 8 bug 修復 + 24 integration tests，274 passed |
| ~~測試覆蓋率 >80%~~ | `2026-04-03-0010` | ✓ 57% → **85%**，270 passed |
| ~~Multi-Model Comparison + Stock Ticker~~ | `2026-04-03-0070` | ✓ ModelComparer + 比較圖表 + CLI --compare/--models/--symbols |

### P1 — 中優先

| 項目 | 來源文件 | 說明 |
| --- | --- | --- |
| ~~Unified TrainingConfig~~ | [`2026-04-03-0100`](2026-04-03-0100-basetrainer-abstraction.md) | ✓ 統一 training_config 參數 + sklearn training history + CurrencyPredictor 簡化 |
| ~~Adaptive feature engineering~~ | [`2026-04-03-0110`](2026-04-03-0110-adaptive-feature-engineering.md) | ✓ 自適應 windows/lags + 12 新測試，292 passed |
| ~~mypy type checking~~ | [`2026-04-03-0120`](2026-04-03-0120-mypy-type-checking.md) | ✓ 111→0 errors, pydantic-mypy plugin, 14 files fixed |

### P2 — 低優先 / 未來

| 項目 | 來源文件 | 說明 |
| --- | --- | --- |
| ~~CI/CD 配置~~ | [`2026-04-03-0130`](2026-04-03-0130-cicd-pipeline.md) | ✓ GitHub Actions: lint → typecheck → test → coverage |
| ~~互動式圖表 (Plotly + Dark Mode)~~ | [`2026-04-03-0140`](2026-04-03-0140-interactive-visualization.md) | ✓ InteractiveVisualizer + ThemeManager + HTML export, 19 tests |
| ~~PDF 報告生成~~ | [`2026-04-03-0150`](2026-04-03-0150-pdf-report-generation.md) | ✓ ReportGenerator + fpdf2 匯出預測報告 + 圖表, 13 tests |
| ~~config.example.json + .env.example~~ | `pydantic_settings_migration_report.md` | ✓ `.env.example` 已在 #0060 建立 |
| ~~Notebook CI (nbstripout + papermill)~~ | [`2026-04-03-0160`](2026-04-03-0160-notebook-ci.md) | ✓ nbstripout git filter + CI verify + 1.5 MB repo size 減少 |

## 測試現況

### Phase 1: Import 修正（已完成 2026-04-03）

8 個 test files 使用了錯誤的 `from src.currency_predictor...` import（正確為 `from currency_predictor...`），已全部修正。

### Phase 2: 27 個 Failing Tests 修復（已完成 2026-04-03）

修復分類：

1. **API 不符合（method 改名/移除）** — `test_prediction_pipeline.py` (4), `test_data_storage.py` (5), `test_currency_predictor.py` (4)
2. **輸出格式變更（emoji→text）** — `test_result_formatter.py` (6→2，其中 4 個在 Phase 1 已修)
3. **模型參數/行為變更** — `test_models_patchtst.py` (3), `test_training_issues.py` (2)
4. **DataProcessor 邏輯變更** — `test_data_processor.py` (2)
5. **Config 預設值變更** — `test_config_manager.py` (1)
6. **Legacy test 不適用** — `test_minimal.py` (1)

### 最新測試結果（2026-04-03 全部 plan 完成後）

- **324 passed, 0 failed**（non-slow, non-e2e，含所有新增測試）
- **12 passed**（`@pytest.mark.slow`：7 offline E2E + 5 pretrained integration）
- **4 passed**（`@pytest.mark.e2e`：online E2E smoke test，需網路）
- 1 pre-existing failure in HuggingFace test（PermissionError，非新引入）
- 測試環境：Docker dev container, RTX 3090, CUDA 12.8

### 覆蓋率：≥80%（目標 >80% ✓）

### 下一步（各有獨立 plan，有執行順序）

1. ~~修復 27 個 failing tests~~ ✓ 已完成
2. ~~清理 legacy 模組~~ ✓ 已完成 → [`2026-04-03-0020-cleanup-legacy-modules.md`](2026-04-03-0020-cleanup-legacy-modules.md)
3. ~~補充覆蓋率 67%→85%~~ ✓ 已完成 → [`2026-04-03-0010-boost-test-coverage.md`](2026-04-03-0010-boost-test-coverage.md)
4. ~~選擇 P0 項目~~ ✓ 已完成 → [`2026-04-03-0030-patchtst-p0-selection.md`](2026-04-03-0030-patchtst-p0-selection.md) → HuggingFace
5. ~~HuggingFace PatchTST 整合~~ ✓ 已完成 → [`2026-04-03-0040-patchtst-huggingface-integration.md`](2026-04-03-0040-patchtst-huggingface-integration.md)
6. ~~HuggingFace Pretrained + Fine-Tune~~ ✓ 已完成 → [`2026-04-03-0050-patchtst-pretrained-finetune.md`](2026-04-03-0050-patchtst-pretrained-finetune.md)
7. ~~Docker Containerization (uv + CUDA GPU)~~ ✓ 已完成 → [`2026-04-03-0060-docker-containerization.md`](2026-04-03-0060-docker-containerization.md)
8. ~~Multi-Model Comparison + Stock Ticker~~ ✓ 已完成 → [`2026-04-03-0070-multi-model-comparison.md`](2026-04-03-0070-multi-model-comparison.md)
9. ~~Testing & Validation Strategy~~ ✓ 已完成 → [`2026-04-03-0080-testing-validation-strategy.md`](2026-04-03-0080-testing-validation-strategy.md)
10. ~~PyTorch Lightning PatchTST~~ ✓ 已完成 → [`2026-04-03-0090-patchtst-lightning.md`](2026-04-03-0090-patchtst-lightning.md)
11. ~~Unified TrainingConfig + Training History~~ ✓ 已完成 → [`2026-04-03-0100-basetrainer-abstraction.md`](2026-04-03-0100-basetrainer-abstraction.md)
12. ~~Adaptive Feature Engineering~~ ✓ 已完成 → [`2026-04-03-0110-adaptive-feature-engineering.md`](2026-04-03-0110-adaptive-feature-engineering.md)
13. ~~mypy Type Checking~~ ✓ 已完成 → [`2026-04-03-0120-mypy-type-checking.md`](2026-04-03-0120-mypy-type-checking.md)
14. ~~CI/CD Pipeline~~ ✓ 已完成 → [`2026-04-03-0130-cicd-pipeline.md`](2026-04-03-0130-cicd-pipeline.md)
15. ~~Interactive Visualization~~ ✓ 已完成 → [`2026-04-03-0140-interactive-visualization.md`](2026-04-03-0140-interactive-visualization.md)
16. ~~PDF Report Generation~~ ✓ 已完成 → [`2026-04-03-0150-pdf-report-generation.md`](2026-04-03-0150-pdf-report-generation.md)
17. ~~Notebook CI~~ ✓ 已完成 → [`2026-04-03-0160-notebook-ci.md`](2026-04-03-0160-notebook-ci.md)

### 建議執行順序

```text
P0: #0090 Lightning ✓
         ↓
P1: #0100 TrainingConfig ✓ ──→ #0110 Adaptive Features ✓
         ↓
P1: #0120 mypy ✓
         ↓
P2: #0130 CI/CD ✓ ──→ #0160 Notebook CI ✓
P2: #0140 Plotly ✓
P2: #0150 PDF ✓
```

## 風險評估

- ~~HuggingFace / Lightning 版可能需要大量 GPU 資源測試~~ → Docker + RTX 3090 已驗證
- ~~BaseTrainer 重構會影響現有所有 model 的 interface~~ → 改為統一 TrainingConfig 參數，風險低
- ~~覆蓋率不足可能隱藏既有 bug~~ → 270 tests, ≥80% coverage

## 完成標準

本文件為 roadmap 總覽，各項目實作時需建立獨立 plan。本文件在所有 P0 項目完成後標記為 completed。
