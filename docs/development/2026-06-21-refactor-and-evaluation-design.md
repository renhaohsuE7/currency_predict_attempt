# 重構 + PatchTST 效果評估 設計規格

> 日期:2026-06-21 · 狀態:設計核准,待寫 implementation plan。
> 範圍:本 repo(`currency_predict_attempt`,FX 匯率預測)的清理重構 + 三個 PatchTST 實作的預測效果實測。
> 執行於 submodule 分支 `feat/refactor-and-eval`;此 repo 是父專案 `yfinance-project-demo-412` 的 git submodule(`external/currency_predict_attempt`)。
> 既有依據:`docs/development/architecture_analysis_report.md`(作者自列的重構 roadmap §4)、`docs/development/basic_usage_problems_analysis.md`(已知資料 drop 問題)。

## 目標(Goal)

1. **實測**三個 PatchTST 實作(sklearn 版、HuggingFace transformers 版、PyTorch Lightning 版)在 FX 資料上的**預測效果**,並對照 **naive 隨機漫步基準**,誠實回答「效果如何 / 能不能端到端跑」。
2. 依評估結果做**高價值重構**:清掉重複模型實作(留會動、相對好的)、統一 `fit()` 介面、拆 `CurrencyPredictor` god-class、修過時 README。
3. **檢查資料輸入流程**(collector → 檔案儲存 → processor),記錄其機制與弱點,並對照父專案的 PostgreSQL 較佳做法;拆 `DataManager` 時把儲存介面留乾淨,使日後可換 DB adapter。

## 非目標(Non-goals)

- 不做作者 roadmap 後段的 **DI 容器、ScalerStrategy / FeatureStrategy 策略群** —— 對此小專案是過度設計。
- **不**現在就建 DB 版儲存(只把介面留乾淨,DB adapter 列為後續)。
- **不**為了重構去修「與重構無關、本來就紅」的測試(記錄但不擴張範圍)。
- 不改模型的數學/演算法本身(行為保持);評估只量測現況。
- 不 push 到此 repo 的 GitHub remote(本地 commit + 父 repo gitlink 更新即可)。

## 全域限制(Global Constraints)

- 在 submodule 分支 `feat/refactor-and-eval` 內進行;**行為保持**重構,每階段以本 repo 的 `pytest` 套件當安全網(綠 → 動 → 仍綠)。
- 環境:`uv sync`(裝全依賴,含 torch 2.8 / transformers / pytorch-lightning);Python ≥ 3.11。
- 評估報告與本 spec/plan 放本 repo `docs/development/`;程式改動 commit 進本 repo。最後在父 repo 更新 submodule gitlink(`git add external/currency_predict_attempt`)。
- Conventional Commits。

## Phase 0 — 環境 + 綠色基準

- `uv sync`(全依賴);若 uv 不可用則以對等 pip/venv 安裝 `pyproject.toml` 依賴。
- 跑 `uv run pytest`(或 `pytest`)→ 記錄**基準**(通過/失敗清單)。
- 分流:
  - 全綠 → 進 Phase 1。
  - 有紅 → 區分「快速可修綠」與「pre-existing 無關失敗」;後者記在評估報告的「基準狀態」一節,排除於「必須維持綠」集合,**不**擴張重構範圍去修。

## Phase 1 — 標出活/死碼 + 檢查資料輸入流程

- 追 `ModelFactory` / `create_patchtst_model`(`src/currency_predictor/models/factory.py`)與 `tests/` 實際 import 哪些模型模組,判定 **canonical**(很可能是新的 `models/patchtst/{sklearn,huggingface,lightning}/`)vs **被取代**(`models.py`、`models/patchtst.py`、`models/patchtst_transformer.py`)。
- **資料輸入流程檢查**:讀 `data/collectors.py`(`YahooFinanceCollector`)、`data/storage.py`(`DataStorage`,檔案 CSV/JSON)、`data_processor.py`。記錄:格式(raw=CSV 檔、results=JSON)、命名/時間戳機制、重複/陳舊資料風險、與重構無關但影響「效果」的點(例如 `create_technical_indicators` + lag 造成 dropna 砍量)。
- 與父專案對照:父專案用 PostgreSQL(`yfinance_1d` 等表 + migration + upsert),較此處檔案散落機制可靠/可查詢;在報告記下「為何 DB 較佳」與「DataManager 介面化後可接 DB adapter」。

## Phase 1.5 — 實測 PatchTST 效果(三模型 vs naive)

- 對 **USDTWD**(`data/raw/` 已有;必要時補抓)為主,視時間加跑 EURUSD/GBPUSD:
  - 每個可跑的模型:訓練(或載入既有 `.joblib`/`.ckpt`)→ held-out 測試窗預測 → 量 **RMSE / MAE / R² / 方向準確率**。
  - **naive 隨機漫步基準**(預測值 = 前一日值)同窗同指標。**模型必須贏過 naive 才算有效**(匯率近 random walk)。
  - 記錄每個模型是否能**端到端跑**;若 dropna bug 使可訓資料 ≈ 0,如實記為「無法有效訓練」。
- 產出 `docs/development/patchtst_evaluation.md`:三模型 vs naive 指標表 + 可跑性 + 結論(哪個值得留)。
- **回饋重構**:Phase 2 清重複時,留評估中會動、相對好的實作;壞掉/沒料的標記並決定修或棄。

## Phase 2 — 清重複(刪死碼)

- 刪被 Phase 1/1.5 判定為被取代/不可用的重複模型檔;殘留 import 改指 canonical package。
- 每步後 `pytest` 維持綠(允許清單外的 pre-existing 紅)。

## Phase 3 — 統一 fit() 介面

- `BaseModel.fit(self, X, y, validation_data=None, **kwargs)`;兩模型 fit 簽名一致,使 `CurrencyPredictor.train_model` 內的 `validation_split` 特判(`predictor.py:215-243`)能簡化。測試綠。

## Phase 4 — 拆 god-class(保留公開 API)

- 依 architecture_analysis_report §4.2,把 `CurrencyPredictor`(6 職責)抽成:
  - `DataManager`(收集 + 儲存;**儲存走介面**,留 DB adapter 空間)
  - `ModelTrainer`(訓練 + 評估)
  - `PredictionEngine`(預測)
- `CurrencyPredictor` 留成 **thin Facade** 委派上述協作者,**公開方法簽名不變**(`collect_and_store_data`/`train_model`/`predict`/`save_model`/`load_model`/`get_model_info`),使現有 `tests/`、`examples/`、`main.py` 照常通過。測試綠。

## Phase 5 — README + 資料流註記

- 重寫 `README.md` 對齊真實結構(現嚴重過時:描述 `data_collector/models/utils`,實為 `config/prediction/reporting/visualization/models/patchtst/...`)。
- 在 README 或 docs 註記資料輸入流程現況 + 「DB 化」為建議後續(對照父專案 PostgreSQL)。

## 驗證(Testing)

- 本 repo 既有 `tests/`(16 檔:test_currency_predictor / test_models_* / test_data_* / test_prediction_* / test_config_manager / test_utils …)為安全網。
- 每階段:重構前後跑全套件,維持「基準綠集合」全綠;新增/調整測試僅在介面變更需要時(例如 Facade 委派、fit **kwargs)。
- 評估(Phase 1.5)以可重跑的腳本產生報告數字。

## 未來考量(本次不決定,使用者明示「到時再評估」)

- **視覺化吸收 + 圖表技術重評**:本 repo 的 `visualization/visualizer.py`(matplotlib 儀表板)+ **Iansui 中文字型** + 其顯示輸出,是可吸收進父專案的素材。但**到時**要重新評估圖表做法:沿用 matplotlib / 改用 **plotly** / 或用父專案報價頁的 **lightweight-charts**(視「要嵌進 web app」還是「產靜態圖」而定)。
- 本次重構**不動** `visualization/`(僅在 Phase 1 記錄其耦合問題,例如硬編 matplotlib 字型設定難以無 GUI 測試);視覺化吸收/改寫另開週期。
- 與 Phase 1.5 評估相關:若評估產出預測圖,先用現有 matplotlib+Iansui 產圖即可,不在本次切換圖表技術。

## 風險與決策紀錄

- **torch 全裝體積/時間**(GB 級);Phase 0 會耗時。
- **基準可能有紅**:本專案糙;以「基準綠集合」框定,不讓無關紅卡住。
- **評估可能顯示模型不如 naive / 無法訓練**:這本身就是誠實答案,並指導「留哪份、修或棄」。
- **公開 API 保持不變**:god-class 改 Facade 委派,降低破壞既有測試/examples 風險。
- **資料儲存介面化**:為日後接父專案 DB 預留,但本次不建 DB adapter(YAGNI)。
- **不 push**:成果留本地 + 父 repo gitlink;push 到 FX GitHub 由使用者另行決定。
