# PatchTST 效果評估:三實作 vs naive 隨機漫步 — 2026-06-21

> Phase 1.5 of `2026-06-21-refactor-and-evaluation-plan.md`。
> 回答使用者:「驗證他 torch patch tst、huggingface patch tst 效果如何」。
> 可重跑腳本:`scripts/evaluate_patchtst.py`(`uv run python scripts/evaluate_patchtst.py`)。

## 方法

- **資料**:`USDTWD=X` 1y 日線(經專案自己的 `DataStorage` 載入),造因果特徵後 **250 點**,訓練 200 / 測試 50。
- **設定(三模型一致)**:`seq_len=60, pred_len=5, patch_len=8, stride=4`;HF/Lightning epochs=30。
- **評估法**:**固定模型 walk-forward 單步預測**(無洩漏)。訓練段 fit 一次(scaler 只看訓練段),
  再對每個測試點餵入到該點為止的歷史,取模型多步輸出的**第 1 步**當次日預測。
- **對照**:naive 隨機漫步(預測 `close_t = close_{t-1}`)。匯率近隨機漫步,**模型須贏過 naive RMSE 才算有效**。
- 刻意**不**走 god-class 的 `prepare_training_data`(它的 X/y framing 有疑慮),自建乾淨的次日預測設定。
- 方向準確率:`sign(pred − prev) == sign(actual − prev)`;naive 永遠預測「不變」故 dir_acc=0(基準特性,非缺陷)。

## 結果

| model | n | RMSE | MAE | R² | dir_acc | 端到端可跑 |
|---|---|---|---|---|---|---|
| **naive_random_walk** | 50 | **0.07614** | **0.06163** | 0.359 | 0.000 | — |
| patchtst_sklearn | 50 | 0.36973 | 0.35500 | −14.12 | 0.480 | ✔ fit+predict |
| patchtst_huggingface | 50 | 0.30142 | 0.26522 | −9.05 | 0.500 | ✔ fit+predict |
| patchtst_lightning | 50 | 0.20496 | 0.16267 | −3.65 | 0.620 | ✔ fit+predict(scaler 已修,見下) |

(USDTWD≈31.x;naive RMSE 0.076 ≈ 0.24%。)

> **更新(scaler 修復後)**:Lightning `predict` 原本缺 scaler,RMSE 30.94 / R² −10⁵(預測≈0)。
> 修復(輸入用訓練 scaler 標準化、輸出反標準化)後 → **RMSE 0.205、R² −3.65、方向準確率 0.62**,
> 一躍成為三者中**最不差**且方向準確率最高者,但**仍輸 naive**(0.205 > 0.076)。結論不變。

## 結論

**三個 PatchTST 實作沒有一個贏過 naive 隨機漫步,全部明顯更差。** 方向準確率 0.48–0.52,等同擲銅板,無預測優勢。
這對 FX 是預期內的(匯率近隨機漫步,要贏 naive 本就極難),但這些實作的弱點/bug 讓結果遠比「略輸」更糟:

- **sklearn(RMSE 5× naive)**:`predict` 把多步輸出寫成 `np.full(pred_len, value[0])` —— **退化成一條水平線**(常數);
  且 GBR 目標是「未來 pred_len 日 Close 的平均」,單步取第 1 步等於拿區間均值當次日點估,系統性偏離。可跑但無效。
- **huggingface(RMSE 4× naive,本組最不差)**:真 Transformer,但 `_prepare_data` **只用 Close 單變量**(忽略所有技術指標),
  且資料量小(訓練序列 ~135)、epochs 少。R²=−9 仍遠輸 naive。是「相對最不壞」的一個。
- **lightning(修前 RMSE 30 ≈ 預測 0;修後 RMSE 0.205,本組最不差)**:`predict` 路徑原本 **完全沒套用 scaler、也沒 inverse_transform**
  (`_prepare_input_tensor` 直接把原始值丟進模型)。模型在標準化空間(~N(0,1))訓練,推論卻吃原始 ~31 的值、輸出也不還原,
  → 預測值≈0、RMSE≈標的水準(30.9)、R²≈−10⁵。**已修復**:`predict`/`predict_with_uncertainty` 改為輸入用訓練 scaler 標準化、
  輸出反標準化(見 `lightning/wrapper.py` 的 `_scaled_recent_input`),回歸測試 `tests/test_lightning_predict_scaling.py`。修後三者中最不差,但仍輸 naive。

## 對「資料輸入流程/洩漏」說法的修正(誠實補記)

`code-map-and-data-flow.md` 原把 god-class 的「`y=當日 Close`、無 `shift`」記為「近似洩漏致準確率虛高」。
細看模型內部後**修正**:sklearn/HF/Lightline 的 `fit` 都會**自行用視窗切出未來 `pred_len` 目標**(真的是預測未來),
故並非「拿當日預當日」那種顯式洩漏。真正的問題是:
(a) sklearn 退化常數預測;(b) HF 單變量+資料少;(c) Lightning predict 缺 scaling;
(d) 各家 `evaluate()` 的 `y_actual = y[-len(pred):]` 對齊鬆散、god-class 測試段常 < `seq_len` 致測試指標為空。
`bfill()` 回填指標起始 NaN 仍是輕度未來洩漏,但不改本結論。

## 回饋重構(Phase 2 用)

- **刪除標的不變**:Task 4 刪的是 5 個 *被取代/壞掉* 的重複檔(`models.py`、`models/patchtst.py`、
  `models/patchtst_transformer.py`、`models/transformer/`、`data_collector.py`);**三個 backend 都留**
  (都接在 `ModelFactory` 上、有測試覆蓋,且使用者要求「評估」而非移除)。
- **品質標記(不在本次修,列後續)**:Lightning `predict` 的 scaler 缺漏是明確 bug,建議後續補
  `scaler.transform` / `inverse_transform`(或乾脆移除 Lightning backend);sklearn 多步預測的常數退化建議改真多步策略。
  本次「行為保持」不動演算法。
- **務實定位**:若父專案要「能用的 FX 預測」,目前沒有任一實作達標;短期 baseline 直接用 naive 隨機漫步即可,
  PatchTST 系列需要更多資料 + 修 bug + 多變量才談得上效果。
