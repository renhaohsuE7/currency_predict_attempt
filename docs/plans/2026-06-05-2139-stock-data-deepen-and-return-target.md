# Plan: 股票預測資料強化 — A 加深歷史+啟用 CAPM、D 改用報酬率目標

- **Date**: 2026-06-05 21:39
- **Status**: completed
- **Decisions**: 歷史 10y;保留 config.json 外匯預設 + 強化股票設定(tw2330 10y+CAPM、新增美股 stock_config.json);評估在報酬空間;A+D 一起做
- **驗證**: 全測 641 passed, 0 failed, 49 deselected;mypy clean;A+D 檔 black/flake8 clean
- **Module**: config、data_processor、prediction/predictor、prediction/comparer、prediction/pipeline、models（評估空間）
- **Goal 對應**: 專案最大目標「用 PatchTST 預測股票」之資料面強化
- **Related**: code review finding #4（CAPM zero-variance inf）、#5（交易日 vs 日曆日）

## 背景

目前(`docs/issues/2026-06-05-code-review-findings.md` 分析):
- 預設標的是外匯 `USDTWD=X`,期間 2–3 年(~500–750 交易日)。
- CAPM/大盤特徵存在但**預設關閉**(`capm.enabled=false`)。
- 預測目標是**價格水準 Close**(非平穩,naive 極難打敗)。

本計畫做兩件(A、D),先求快速見效與可學習性提升;多股 panel(B)留待後續。

---

## Part A — 加深單股歷史 + 啟用 CAPM 大盤特徵

### 目標
讓股票設定使用更長歷史(5–10y)並納入大盤關聯特徵(Beta/Alpha/相對報酬)。

### 影響範圍
| 檔案 | 變更 |
| --- | --- |
| `tw2330_config.json`(及新增 `stock_config.json` 範例) | `period` 2y → **10y**(data_collection/model_training/prediction/backtest.data_period);`capm.enabled` → **true**;market_index 依市場(`^TWII`/`^GSPC`) |
| `src/currency_predictor/config/settings.py` | (選擇性)`CAPMConfig` 加 `auto_enable_for_stock` 或文件說明;period 預設不動 |
| `src/currency_predictor/data_processor.py` | **修 CAPM zero-variance 防護(code review #4)**:`Rolling_Beta=Cov/Var`、`Sharpe` 加零分母 guard + 將 inf 轉 NaN 再 bfill/ffill。啟用 CAPM 後這條 bug 會變 live,必須一起修 |
| `src/currency_predictor/prediction/predictor.py` | 確認 `_get_market_index_data()` 抓取期間與訓練期間一致(目前抓固定 2y → 需隨 period 拉長,否則 CAPM 特徵在長歷史前段為 NaN) |

### 實作步驟
1. `data_processor.create_capm_features`:`var==0`/`rolling_std==0` 時該列 Beta/Sharpe 設 0(或 NaN);算完統一 `replace([inf,-inf], nan)` 後再 bfill/ffill。(= 修 #4)
2. `predictor._get_market_index_data()`:市場指數抓取 period 對齊訓練 period(參數化,而非寫死)。
3. 更新 `tw2330_config.json` → 10y + capm.enabled=true(`^TWII`);新增 `stock_config.json`(美股範例,`AAPL` + `^GSPC`)。
4. 端對端驗證:`docker compose run --rm prod uv run main.py --config tw2330_config.json`(或對應 CLI)跑通,確認特徵矩陣無 inf/NaN、模型可訓練。

### 風險
- 長歷史 + CAPM rolling_window=252 → 前 252 列特徵不穩;切分時注意 train 量。
- yfinance 10y 日線資料量穩定;市場指數對齊日期(交易日)需 reindex/ffill。

---

## Part D — 預測目標:價格 → 對數報酬率(log-return)

### 目標
把目標從非平穩的價格水準改為**對數報酬率** `r_t = ln(P_t) - ln(P_{t-1})`,提升可學習性;對使用者輸出時再還原為價格。

### 設計決策（重點）
1. **設定開關**:`model_training.target_transform: "price" | "log_return"`,**預設 `"price"`**(完全向後相容);股票設定改 `"log_return"`。
2. **訓練空間**:`prepare_training_data` 在 `log_return` 模式下令 `y = ln(Close).diff()`(丟首列 NaN),模型在報酬空間訓練/預測。
3. **評估空間**:**在報酬空間評估**(`BaseModel.evaluate_rolling/single_shot` 不變,y_true 即為報酬)。
   - 理由:naive 報酬預測=0(等價於價格持平 random walk),MASE 在報酬空間仍有意義且與價格空間概念一致;可保持 evaluate 層 model-agnostic 不動。
   - 額外(選擇性):在報表附「價格空間 reconstructed RMSE」供解讀。
4. **預測還原**:`predict()` 取模型輸出的 log-return 序列 `r̂`,以最後已知收盤 `P_last` 重建未來價格 `P̂_i = P_last * exp(cumsum(r̂)_i)`。使用者拿到的仍是價格 + 預測日期。

### 影響範圍
| 檔案 | 變更 |
| --- | --- |
| `src/currency_predictor/config/settings.py` | `ModelTrainingConfig` 新增 `target_transform: Literal["price","log_return"]="price"` |
| `src/currency_predictor/data_processor.py` | 新增 `to_log_returns(close)` / `from_log_returns(last_price, returns)` 純函式 |
| `src/currency_predictor/prediction/predictor.py` | `prepare_training_data`:依 transform 構造 `y`;`predict()`:報酬→價格重建;`__init__` 接收 `target_transform` |
| `src/currency_predictor/prediction/comparer.py`、`pipeline.py` | 將 `target_transform` 由 config 透傳給 predictor |
| `tests/test_target_transform.py`（新檔） | log-return 構造、還原往返(`from(to(x))≈x`)、predict 重建價格正確性、price 模式回歸不變 |

### 實作步驟
1. `settings.py` 加欄位。
2. `data_processor` 加 `to_log_returns`/`from_log_returns`(含邊界:長度、首列、P_last>0)。
3. `predictor.prepare_training_data`:`log_return` 模式構造 y(報酬);保留 `self._last_close`/price index 供還原。
4. `predictor.predict`:`log_return` 模式下 `P̂ = P_last * exp(cumsum(r̂))`;`price` 模式維持原樣。
5. `comparer`/`pipeline`:透傳 `target_transform`。
6. 測試 + 端對端:股票設定(`log_return`)跑通,forecast JSON/圖為價格;`price` 模式所有既有測試不變。

### 風險
- **還原基準**:重建需正確的 `P_last`(最後一筆實際收盤),須與預測日期對齊(注意 #5 交易日問題,但本計畫不強制修 #5)。
- **評估一致性**:切換空間後 RMSE 數量級改變(報酬 vs 價格),報表需標明空間,避免跨設定誤比。
- **向後相容**:預設 `price` 確保現有外匯/既有測試零影響。

---

## 完成標準
- [ ] A:股票設定 10y + CAPM 啟用可端對端跑通,特徵矩陣無 inf/NaN(#4 修正)
- [ ] A:`_get_market_index_data` 期間對齊訓練 period
- [ ] D:`target_transform="log_return"` 下訓練/預測/還原為價格正確;`price` 模式完全向後相容
- [ ] D:往返與重建單元測試通過
- [ ] 全測通過(維持 634 passed 基準,新增測試)
- [ ] black/flake8/mypy clean
- [ ] 容器內 `docker compose run --rm test`(`.env` uid 已修)

## 不在本計畫範圍（後續）
- B 多股 panel 全域模型(樣本量關鍵)
- C 多變量脈絡(同儕股為輸入 channel)
- #5 交易日行事曆(可在 D 之後單獨處理,會強化 forecast 對齊)
