# Plan #0080: Testing & Validation Strategy — E2E Tests

- **Date**: 2026-04-03
- **Status**: completed
- **Module**: tests/, docs/

## 目標

建立完整的端到端（E2E）測試策略，填補目前測試架構中「無真實資料流驗證」的空缺。目前 270+ tests 全部使用 mock 外部依賴，無法捕捉跨模組整合問題。

## 現有測試架構

| 層級 | 範圍 | 測試數 | 外部依賴 |
| --- | --- | --- | --- |
| Unit | 單一 class/function | ~230 | 全部 mock |
| Integration | 跨模組（如 PredictionPipeline） | ~40 | mock yfinance + 簡化資料 |
| Slow Integration | HuggingFace pretrained 模型 | 6 | 需網路 + GPU |
| **E2E（缺失）** | **collect → train → predict → visualize → report** | **0** | **真實資料流** |

## 設計決策

| 問題 | 決策 |
| --- | --- |
| Fixture 資料 | 合成 deterministic CSV（300 行，`np.random.seed(42)`），避免市場資料授權問題 |
| E2E 分層 | Layer 1: Offline（fixture CSV, `@pytest.mark.slow`）<br>Layer 2: Online（yfinance, `@pytest.mark.e2e`） |
| E2E 模型 | `patchtst_sklearn`（最快、無需 GPU） |
| Model 參數 | `seq_len=50, pred_len=5, patch_len=10, stride=5, n_estimators=10, max_depth=3` — 控制在 10s 內 |
| conftest.py | 新建全域 shared fixtures + `matplotlib.use('Agg')` headless rendering |
| pytest 預設行為 | `addopts = "-m 'not slow and not e2e'"` — `uv run pytest` 只跑快速測試 |

## 三層測試策略

### Layer 0: Unit / Integration（既有，不改）

- 270+ tests，80%+ coverage
- 所有外部 I/O 用 mock
- 執行：`uv run pytest`（預設）

### Layer 1: Offline E2E（新增，`@pytest.mark.slow`）

**目的**：不需網路，用已提交的 fixture CSV 驗證完整 pipeline。

**覆蓋的 pipeline 階段**：

```
fixture CSV → DataProcessor.clean_data()
            → DataProcessor.create_technical_indicators()
            → DataProcessor.create_lagged_features()
            → PatchTSTSklearn.fit()
            → PatchTSTSklearn.predict()
            → PatchTSTSklearn.evaluate()
            → CurrencyVisualizer.plot_prediction_results()
            → ResultFormatter.generate_report()
```

**測試內容**（7 tests）：

| 測試 | 驗證重點 |
| --- | --- |
| `test_fixture_loads_and_has_expected_shape` | CSV 載入 ≥250 rows, OHLCV 欄位存在 |
| `test_data_processor_pipeline_preserves_rows` | clean → indicators → lag 保留 90%+ rows |
| `test_sklearn_fit_predict_round_trip` | 完整 load → process → fit → predict 產出 valid ndarray |
| `test_predictions_are_reasonable_range` | 預測值在 last close ±50% 以內 |
| `test_evaluate_returns_valid_metrics` | RMSE/MAE/MSE ≥ 0 |
| `test_prediction_chart_creates_file` | plot_prediction_results 產出 PNG (>0 bytes) |
| `test_report_generation_produces_content` | ResultFormatter 產出含 symbol 的 Markdown |

**執行**：`uv run pytest -m slow -v`

### Layer 2: Online E2E（新增，`@pytest.mark.e2e`）

**目的**：實際呼叫 yfinance，驗證從資料收集到最終報告的完整真實流程。

**覆蓋的 pipeline 階段**：

```
YahooFinanceCollector.get_currency_data()
→ DataStorage.save_raw_data() / load_raw_data()
→ DataProcessor (full pipeline)
→ PatchTSTSklearn.fit() / predict()
→ CurrencyVisualizer.plot_prediction_results()
→ ResultFormatter.generate_report()
```

**測試內容**（4 tests）：

| 測試 | 驗證重點 |
| --- | --- |
| `test_yfinance_returns_ohlcv` | 真實 yfinance 回傳 ≥30 rows OHLCV |
| `test_data_storage_round_trip` | save + load via DataStorage 一致 |
| `test_train_predict_on_real_data` | sklearn model 在真實資料上 fit + predict |
| `test_full_chain_chart_and_report` | predict → chart PNG → report Markdown 全部成功 |

**網路保護**：module-level `_network_available()` check，無網路時自動 skip。

**執行**：`uv run pytest -m e2e -v`

### Layer 3: Use Case & E2E Tests（新增，mixed markers）

**目的**：用真實 fixture data 驗證完整業務流程，以及 CLI entry point 和 error recovery。

**Plan docs**：`docs/plans/testing/` 目錄下 5 個獨立計畫文件。

**測試內容**（28 tests across 5 modules）：

| 測試檔案 | Marker | Tests | 說明 |
| --- | --- | --- | --- |
| `test_use_case_save_load_predict.py` | `@slow` | 5 | Train → Save → Load → Re-predict cycle |
| `test_use_case_currency_predictor.py` | `@slow` | 6 | CurrencyPredictor prepare/train/predict/save/load |
| `test_use_case_model_comparison.py` | `@slow` | 5 | ModelComparer with 2 real sklearn models |
| `test_e2e_cli.py` | 無 | 7 | CLI argparse + mode routing (mock-based) |
| `test_e2e_error_recovery.py` | 無 | 5 | Partial failure / graceful degradation |

**Shared fixtures**（`conftest.py` 新增）：

- `processed_fixture_data` — 載入 fixture CSV + DataProcessor pipeline → 可直接 split X/y

## 新增檔案

| 檔案 | 說明 |
| --- | --- |
| `tests/conftest.py` | 全域 fixtures：`sample_ohlcv_df`, `e2e_fixture_path`, `e2e_output_dir`, `fast_sklearn_params`, `processed_fixture_data` |
| `tests/fixtures/USDTWD_1y_sample.csv` | 合成 300 行 OHLCV（deterministic, seed=42） |
| `tests/test_e2e_offline.py` | Layer 1: 7 offline E2E tests |
| `tests/test_e2e_online.py` | Layer 2: 4 online E2E tests |
| `tests/test_use_case_save_load_predict.py` | Layer 3: 5 save/load/predict use case tests |
| `tests/test_use_case_currency_predictor.py` | Layer 3: 6 CurrencyPredictor use case tests |
| `tests/test_use_case_model_comparison.py` | Layer 3: 5 model comparison use case tests |
| `tests/test_e2e_cli.py` | Layer 3: 7 CLI e2e tests |
| `tests/test_e2e_error_recovery.py` | Layer 3: 5 error recovery tests |
| `docs/plans/testing/*.md` | Layer 3 的 5 個 plan docs |
| `docs/plans/2026-04-03-0080-testing-validation-strategy.md` | 本文件 |

## 修改檔案

| 檔案 | 變更 |
| --- | --- |
| `pyproject.toml` | 新增 `e2e` marker + `addopts` 排除 slow/e2e |
| `docs/plans/2026-04-03-0000-*` | Roadmap 新增 #0080 |

## 執行指令

```bash
# 快速測試（預設，排除 slow/e2e，339 tests）
uv run pytest

# Offline E2E + Use Case（@slow marker，30 tests）
uv run pytest -m slow -v

# Online E2E（需要網路，4 tests）
uv run pytest -m e2e -v

# 全部測試（排除 HuggingFace，373 tests）
uv run pytest --override-ini="addopts=" --ignore=tests/test_patchtst_huggingface.py

# 全部測試含 HuggingFace（407 tests，需 GPU + 網路）
uv run pytest --override-ini="addopts="

# 只跑 use case tests
uv run pytest tests/test_use_case_*.py -v

# 只跑 e2e tests（CLI + error recovery）
uv run pytest tests/test_e2e_cli.py tests/test_e2e_error_recovery.py -v
```

## 完成標準

- [x] `tests/conftest.py` 建立，既有 270 tests 行為不變
- [x] `tests/fixtures/USDTWD_1y_sample.csv` 存在（300 行合成資料）
- [x] `uv run pytest` 仍只跑快速測試（339 tests passing）
- [x] `uv run pytest -m slow` 通過（30 tests；含 use case tests）
- [x] `uv run pytest -m e2e` 通過（4 passed，有網路時全部通過）
- [x] Layer 3 五個 test modules 全部 passing（28 tests）
- [x] `docs/plans/testing/` 下 5 個 plan docs 全部 completed
- [x] 本文件完成
