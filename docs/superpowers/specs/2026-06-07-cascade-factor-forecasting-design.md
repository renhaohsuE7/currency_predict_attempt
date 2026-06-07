# Design: Cascade 因子增強預測(波動率/方向 → 價格)

- **Date**: 2026-06-07
- **Status**: design (awaiting review)
- **Topic**: 多任務 cascade 預測 —— 先預測波動率與方向當因子,再用來預測股價/報酬
- **Inspiration**: 氣象「先預測中間大氣變數再預測降雨」的二階段法
  - [GNSS-PWV two-step precipitation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12563812/)
  - [Factor-augmented volatility forecasting](https://arxiv.org/html/2508.01880)
  - [Two-stage RF feature-selection + BiGRU for stock indices](https://pmc.ncbi.nlm.nih.gov/articles/PMC12064028/)

## 目標

同時產出三種預測,並讓因子(波動率、方向)輔助最終的價格/報酬預測:

1. **波動率 vol**:未來 `pred_len` 天 log-return 的 realized volatility(標準差),標量。
2. **方向 dir**:未來 `pred_len` 天累積 log-return 的方向(P(up) ∈ [0,1] / sign),標量。
3. **價格/報酬**:沿用現有 `target_transform=log_return`(預測報酬,`predict()` 還原價格)。

## 架構:Cascade(二階段,模型無關)

```
context window X[t-seq_len : t]
   ├─ Stage-1a  FactorRegressor   → vol̂   (未來 horizon realized vol)
   ├─ Stage-1b  FactorClassifier  → dir̂   (未來 horizon P(up))
   └─ Stage-2   主預測器(任一 backend)
                 input = 既有特徵 + [vol̂, dir̂]  → 預測報酬路徑 → 還原價格
```

- **Stage-1 因子模型**:預設用輕量 sklearn(`GradientBoostingRegressor` for vol、
  `GradientBoostingClassifier`/`HistGradientBoosting*` for dir),輸入為既有向量化 patch 特徵。
  因子是標量,sklearn 是自然選擇(快、可分類)。設計成可插拔的 `FactorModel` 介面。
- **Stage-2 主預測器**:透過 `ModelFactory` 建立,**backend 任選**:`patchtst_sklearn` /
  `patchtst_huggingface` / `patchtst_lightning`。因子以**額外特徵欄位(channel)**注入 X。

### 因子注入(各 backend)
- **sklearn**:把 `vol̂`、`dir̂` 加為 X 的兩個常數欄(整個 context 同值)→ patch 特徵自動納入。
- **HuggingFace / Lightning**:同樣加為 X 的兩個額外欄位 → 經 DataProcessor 成為額外輸入 channel
  (`num_input_channels += 2`)。**整合點**:確認兩個 wrapper 接受多欄 X 作 channel;若目前是
  Close-only,需在 wrapper 開放 exogenous channel(實作階段處理,列為風險)。

## 無洩漏(最關鍵正確性)

- **Stage-1 target**:用 `shift` 取「未來」realized vol / direction —— 只用 `t` 之前的 context 特徵,
  target 是 `t..t+pred_len` 的未來統計量。
- **Stage-2 訓練因子**:**必須用 cross-fitting(out-of-fold)** —— 把 train 切 K 折,每折的因子由
  「在其他折訓練的 Stage-1」預測得到,**不可用真實未來 vol/dir**(否則 target leakage,Stage-2 會
  以為因子完美而學壞)。預測(inference)時用「在全部 train 上訓練的 Stage-1」。
- 評估一律 rolling-origin、報酬空間(沿用既有)。

## 元件 / 檔案

| 檔案 | 內容 |
| --- | --- |
| `prediction/factors.py`(新) | 純函式:`realized_volatility(returns, horizon)`、`direction_label(returns, horizon)`(含 shift、無洩漏);`FactorModel`(vol regressor + dir classifier 封裝) |
| `prediction/cascade.py`(新) | `CascadePredictor`:orchestrate stage1(cross-fit)→ 注入因子 → stage2;`fit`/`predict` 回傳 {price, vol, dir} |
| `config/settings.py` | `CascadeConfig`(enabled、vol/dir 定義、cross-fit 折數 k);掛到 ModelTrainingConfig 或 AppSettings |
| `prediction/predictor.py` / `comparer` | 既有特徵抽取重用;factor 欄位注入點 |
| `models/patchtst/huggingface/*`、`lightning/*` | (視需要)開放額外 exogenous channel |
| `tests/test_factors.py` | 因子純函式 + 無洩漏 |
| `tests/test_cascade.py` | cross-fit 無洩漏、端到端、factor-lift 對照;**跨三 backend 參數化**(HF/Lightning 用 `pytest.importorskip`) |

## 評估與成功標準(誠實的 factor-lift)

1. **因子可預測性**:
   - vol:RMSE / QLIKE 明顯優於 naive(波動率有 clustering,理論上可預測)。
   - dir:accuracy / AUC,檢視是否 > 0.5。
2. **Factor lift(核心驗收)**:cascade 的價格/報酬 **MASE / MDA** 對照「**同 backend、無因子 baseline**」。
   誠實呈現因子到底有沒有幫助(預期:對方向/區間/波動有用;對價格點預測幫助可能有限 —— 用數據說話)。
3. **不退步**:cascade 不得比無因子 baseline 明顯更差(fail-loud:若崩壞要報錯,不可靜默)。

## 跨 backend 測試(明確需求)

- `test_cascade.py` 以 `@pytest.mark.parametrize("backend", ["patchtst_sklearn", "patchtst_huggingface", "patchtst_lightning"])` 跑端到端;
  HF/Lightning 在 optional dep 缺失時 `pytest.importorskip` 跳過(CI `--all-extras` 會裝,故會實跑)。
- 驗證三 backend 都能:Stage-1 因子注入 → Stage-2 fit/predict → 回傳 {price, vol, dir}。

## v1 範圍 / YAGNI

- **做**:單股 cascade(model-agnostic,三 backend 皆可)+ cross-fit 無洩漏 + factor-lift 評估 + 跨 backend 測試。
- **不做(後續)**:joint multi-head 共享表徵、panel 全域 cascade、因子的機率/分位數輸出、更多因子
  (成交量、ATR、Parkinson vol 等)。

## 風險

- **HF/Lightning channel 注入**:若 wrapper 目前 Close-only,需開放 exogenous channel(實作風險,
  v1 必須處理因為要求三 backend 可測)。
- **cross-fit 成本**:k 折 → Stage-1 訓練 k 次;k 取小(如 3–5)。
- **因子無洩漏**:最易錯處;以無洩漏單元測試把關。
- **factor lift 可能為零或負**:這是誠實的可能結果;設計上以「對照 baseline」明確呈現,而非假設一定有用。

## 完成標準

- [ ] 因子純函式 + 無洩漏測試通過
- [ ] cross-fit 產生 Stage-2 訓練因子(無洩漏)
- [ ] CascadePredictor 端到端:fit→predict 回 {price, vol, dir}
- [ ] **三 backend(sklearn/HF/Lightning)端到端測試通過**
- [ ] factor-lift 報告:cascade vs 無因子 baseline(MASE/MDA)+ 因子本身準確度
- [ ] 全測綠;black/flake8/mypy clean
