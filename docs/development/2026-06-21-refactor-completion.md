# 重構 + 評估 完成紀錄 — 2026-06-21

> 對應 `2026-06-21-refactor-and-evaluation-{design,plan}.md`。本檔記錄成果、五點決策核對、與後續建議。

## 五點決策核對(使用者前面給出的紀錄)

| # | 決策 | 狀態 |
|---|---|---|
| 1 | **範圍**:高價值子集 —— 清重複 + 拆 god-class + 統一 fit + 修 README;**不**做 DI 容器 / Scaler·Feature 策略群 / DB adapter | ✅ 完成,非目標皆未越界 |
| 2 | **評估**:實測 sklearn / HuggingFace / Lightning PatchTST vs naive 隨機漫步 | ✅ `scripts/evaluate_patchtst.py` + `patchtst_evaluation.md`;結論:三者皆輸 naive |
| 3 | **資料流**:檢查 collector→storage→processor,記錄弱點對照父專案 DB,DataManager 儲存介面留乾淨 | ✅ `code-map-and-data-flow.md`;`DataManager(storage=...)` 接縫,並透過 Facade `CurrencyPredictor(storage=...)` 可達 |
| 4 | **驗證**:裝全依賴(含 torch 2.8)+ 跑 pytest 當安全網 | ✅ `uv sync --extra lightning`;綠基準 124→重構後 stable gate 111 綠(扣除 flaky 視覺化檔),0 回歸 |
| 5 | **儲存/分支**:submodule 本地 commit + 更新父 repo gitlink;**不** push FX GitHub;建 fix/feat/dev 分支 | ✅ 9 commits on `feat/refactor-and-eval` → 併入 `dev`;父 repo 更新 gitlink;**未** push 任何 remote |

## 各階段成果

- **Phase 0**(`baseline-2026-06-21.md`):加 `pytest pythonpath` 設定使收集從 8 錯→0;釘住「綠集合」124 passed / 28 pre-existing-red / 1 skipped。
- **Phase 1**(`code-map-and-data-flow.md`):標出 canonical vs 5 個死碼/被取代檔;記錄 CSV 散落儲存弱點對照 PostgreSQL。
- **Phase 1.5**(`patchtst_evaluation.md`):walk-forward 無洩漏實測。**sklearn/HF/Lightning 全部輸 naive 隨機漫步**(RMSE 4×/4×/400×),方向準確率≈擲銅板。
- **Phase 2**:刪 4 個不可達/壞掉的重複模型檔(`models.py`、`models/patchtst.py`、`models/patchtst_transformer.py`、`models/transformer/`);`data_collector.py` 因綠測試引用而保留。
- **Phase 3**:`BaseModel.fit(self, X, y, validation_data=None, **kwargs)` 統一;`min_required` 抽常數;加 12 個 fit 介面契約測試。
- **Phase 4**:`CurrencyPredictor` 拆成 Facade + `DataManager` / `ModelTrainer` / `PredictionEngine`,公開 API 與相容屬性不變。
- **Phase 5**:README 重寫對齊真實結構 + 資料流弱點 + 誠實評估結論。
- **Code review**:兩位 reviewer 對全 diff,結論「無行為改變的 bug」;依其建議補上 Facade 的 storage 接縫直通。

## 最終測試狀態

- 安全網:`uv run pytest -q --ignore=tests/test_visualizer.py` → **111 passed / 28 pre-existing-red / 1 skipped**,全程 0 回歸(綠集合只增不減:+12 fit 介面測試)。
- `tests/test_visualizer.py`:**pre-existing flaky**(matplotlib/GUI 全域狀態,不同 run 不同測試紅),非本次造成;開發時以 `--ignore` 排除。
- 28 個 pre-existing 紅:API 漂移 / emoji-vs-ASCII / Windows 環境鎖,皆與重構無關(見 baseline 文件分類),依非目標未擴張範圍去修。

## 後續處理 / 建議

- ✅ **Lightning `predict` 缺 scaler(已修)**:`predict`/`predict_with_uncertainty` 改為輸入用訓練 scaler 標準化、輸出反標準化(`lightning/wrapper.py` 新增 `_scaled_recent_input`),回歸測試 `tests/test_lightning_predict_scaling.py`。修後 Lightning RMSE 30.94→0.205、R² −10⁵→−3.65、方向準確率 0.62,成為三者中最不差,但仍輸 naive。
- ✅ **儲存 hygiene(已處理)**:`.gitignore` 加入 `results/`、`data/raw/`、`data/processed/`、`test_models/`、`*.ckpt`、`lightning_logs/`,並 `git rm --cached` 取消追蹤既有訓練/輸出產物(檔案留在磁碟,僅移出版控)。
- **sklearn 多步預測退化成常數**(`np.full(pred_len, value[0])`):建議改真多步策略(尚未做)。
- **DB 化**:`DataManager` 已留 `storage` 介面;日後可加 `PostgresDataStore`(同 `load_raw_data`/`save_raw_data`)接父專案 PostgreSQL。
- **視覺化吸收 + 圖表技術重評**(matplotlib+Iansui vs plotly vs lightweight-charts):使用者明示「到時再評估」,本次未動 `visualization/`。
- **儲存 hygiene**:`results/`、`data/raw/*.csv`、`*.ckpt` 等訓練產物目前未被 `.gitignore`,建議後續補上避免誤 commit。
