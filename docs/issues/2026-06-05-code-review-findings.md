# Issue: Code Review Findings — `a4dfebf` + 工作目錄 WIP

- **Date**: 2026-06-05
- **Status**: open
- **Scope**: `git diff origin/transfomer_attempt → 工作目錄`(已 commit 的 `a4dfebf` + 未 commit 變更)
- **Method**: /code-review (high effort, 7 finder angles + 逐項 verify)

> **前置 blocker**:`models/` 套件誤刪見 `2026-06-05-models-package-accidental-deletion.md`(P0)。
> 下列多數 bug 在 `a4dfebf` 已提交碼中即存在,修好 import 後才會浮現。

## 確認的 bug(CONFIRMED / PLAUSIBLE)

| # | 嚴重度 | 檔案:行 | 問題 |
| --- | --- | --- | --- |
| 1 | P0 | `prediction/predictor.py:16` 等 | `models/` 誤刪 → 套件無法 import(獨立 issue) |
| 2 | High | `prediction/comparer.py:578` | `save_predictions_csv` 把**未來預測日期**配上**過去 test set actual**(`y_test[-n:]`),時間軸錯位 |
| 3 | High | `visualization/visualizer.py:781` | `plot_model_comparison` 用 `actual.index[-n:]` 把未來預測畫在歷史測試日期上,圖表誤導 |
| 4 | High | `data_processor.py:391` | `create_capm_features` 的 `Rolling_Beta=Cov/Var`、Sharpe 無零分母防護;`bfill/ffill` 清不掉 `inf`,污染股票特徵矩陣 |
| 5 | High | `prediction/predictor.py:451` | `predict()` 用 `timedelta(days=i+1)`(日曆日)產生預測日期 → 落在週末/假日,forecast 無法驗證、圖表扭曲 |
| 6 | High | `prediction/pipeline.py:377` | forecast JSON 檔名清洗規則與 `predictor._clean_symbol` 不一致(`EURUSD=X`→`EURUSDX` vs `EURUSD`),導致 `ForecastVerifier` 找不到檔 |
| 7 | Med | `backtesting/runner.py:144` | backtest 硬編 `risk_free_rate=0.04`,live 用 `_get_risk_free_rate()`(抓 ^IRX)→ 同模型 CAPM 特徵不一致 |
| 8 | Med | `backtesting/metrics.py:245` | 年化報酬 `(1+r)**(252/n_days)` 對短測試視窗 overflow 成 `inf`;`max_drawdown`/`compute_returns` 無零分母防護 |
| 9 | Med | `prediction/predictor.py:362` | `_evaluate_model` 改為依賴 `evaluate_rolling`/`evaluate_single_shot`,缺方法時 `AttributeError` 被 except 吞 → 空 metrics → 模型靜默被排除於 ranking(與 #1 相關:這兩方法目前無實作) |
| 10 | Med | `prediction/predictor.py:267` | `prepare_training_data` 由固定 80/20 改為 `len - min(test_days, len//2)`,失去「train ≥ seq_len+pred_len」保證,短歷史標的訓練退化且不報錯 |
| 11 | Low | `data/collectors.py:155` | `get_currency_data_range` 的 except 引用 try 內才綁定的 `start_str/end_str` → `UnboundLocalError` 掩蓋原始錯誤 |

### 設計層面備註(#2/#3)

`docs/plans/2026-04-15-forecast-verification.md` 的 Context 已自承 #2:`predictions_*.csv` 的 "Actual" 是 test set
`y_test[-n:]` 而非真實未來資料。其對策是**另存** `forecast_*.json` 給 `ForecastVerifier` 驗證,而非修 CSV。
因此:CSV / 比較圖仍會誤導(現況未修);真正驗證走 forecast JSON。這使 **#6(檔名不一致)升級為阻斷 verification 機制的關鍵 bug**。
此外該 plan「完成標準」勾選了「非交易日正確處理」,但 #5 顯示實作未做到 → 驗收與程式碼矛盾。

## 已排除(REFUTED)

- `runner.py` fold-metric 聚合 KeyError — splitter 保證每折 `test_size` 固定,所有折走同一分支,key set 一致。
- `save_metrics_csv` 的 `mse` 欄永遠空 — `evaluate_rolling` 的 aggregate 實際含 `mse`(`tests/test_evaluate_rolling.py` 佐證)。
- `train_model` 移除 validation-split 防護 — 該防護原本在 model 的 `fit()`(已刪 models/ 內),非 `predictor.py`;歸入 #1。

## 篇幅外的 cleanup(非 correctness,建議一併處理)

- **效率**:多模型比較路徑中,同一 symbol 的資料蒐集/驗證 + 完整前處理(read_csv + 技術指標 + lagged +
  **CAPM 下載 ^GSPC/^IRX**)重跑 `M+1` 次,且每個 model 的 `predict()` 又再前處理一次。應依 symbol 前處理一次後共用切分。
- **重複實作**:`compute_returns` vs `utils/helpers.calculate_returns`;drawdown 同時存在 `metrics` 與 `visualizer`;
  `win_rate` vs `prediction.metrics.mda`;forecast-JSON 序列化在 comparer 與 pipeline 逐字重複;
  `initial_capital=10000` 在 runner/metrics 散落 5 處硬編。

## 建議處理順序

1. #1(恢復 + 重實作 models/,見獨立 issue)
2. #5 + #6(讓 forecast verification 機制真的能運作)
3. #2 / #3(時間軸錯位)
4. #4 / #8(數值防護:inf/NaN)
5. #7 / #10 / #11
