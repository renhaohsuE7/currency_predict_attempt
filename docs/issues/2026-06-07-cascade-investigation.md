# Investigation: Cascade Factor Forecasting — Root Causes & Observations

- **Date**: 2026-06-07
- **Status**: open (findings; some are real bugs to fix)
- **Symbol**: 2330.TW, 2y (483 rows, 2024-06-05 → 2026-06-04, close 773→2425), seq_len=64, pred_len=15, test_days=120, return space
- **Follows**: `docs/issues/2026-06-07-cascade-factor-lift-results.md`
- **Method**: ran a controlled script (`_investigate.py`, removed after) inside the container — measured base rates, cross-fit factor quality, raw model predictions, and per-model rolling metrics. Numbers below are observed, not assumed.

## Observations (實測)

```
DIRECTION: up_base_rate=0.686  majority_baseline=0.686  crossfit_acc=0.699  skill_over_majority=+0.013
VOLATILITY: model_rmse=0.00911  naive_persistence_rmse=0.01039  mean_vol=0.01972  model_better=True
HF predicted log-returns (first 5): [1369.49, 1369.52, 1368.70, 1369.35, 1368.51]
  actual log-return scale: std=0.0215  abs_mean=0.0160   HF pred abs_max=1372.36
LIGHTNING vs NAIVE (return space, rolling):
  patchtst_lightning  rmse=0.02627  mase=0.858  mda=0.509
  naive               rmse=2032.49   mase=84849  mda=0.032
```

## Finding 1 — 方向「準確度 0.70」是假象(其實是趨勢 base rate)

- 2024-2026 台積電是**上升趨勢**:68.6% 的 15 天窗口是「漲」。
- 因此「永遠猜漲」就有 **0.686** 準確度;cross-fit 模型 0.699 只比 majority-class **+1.3%**。
- **結論:方向預測幾乎沒有真實技巧** —— 先前 `dir_accuracy≈0.70` 被誤讀為「有訊號」,實際上是趨勢造成的 base rate。**必須對照 majority baseline 才有意義。**

## Finding 2 — 波動率「確實可預測」(唯一真訊號)

- 模型 vol RMSE 0.00911 **< naive-persistence(用過去 H 天 realized vol)0.01039**。
- 波動率叢聚(volatility clustering)是真的;這是 cascade 因子中**唯一有真實預測力**的。
- → 因子的價值在**輸出波動率本身**(可預測),而非方向、也非改善價格。

## Finding 3 — HF multi-channel 在報酬空間「預測錯尺度」(真 bug)

- HF 被訓練在 log-return 目標(~0.02),卻**預測出 ~1369 的值**(價格尺度)。
- **根因**:cascade 的 Stage-2 X = 「除 Close 外所有欄位」(feature_cols 不含 Close);HF/Lightning multi-channel 用 `target_channel_idx = index("Close")`,但 Close 不在 X 中 → fallback 到 **channel 0**(某個價格尺度的特徵欄,如 SMA),於是 HF 預測那個欄位的尺度(~1369),不是 return 目標。
- → 先前 factor-lift 的「HF RMSE 1827 / 發散」**不是模型發散,是預測錯 channel/尺度**。HF 在報酬空間 + multi-channel 的目標對齊是壞的。
- **Lightning 卻給出 return 尺度(rmse 0.0263)** —— 兩個 wrapper 對「target channel 不在 X」的處理不一致(Lightning 可能 fallback 行為不同)。這個不一致本身要修。

## Finding 4 — NaiveModel 在報酬空間是無效 baseline(真 bug)

- naive RMSE **2032**、MASE **84849**:`NaiveModel.predict()` 永遠回「最後一個 Close」(價格 ~2300),但目標是 return(~0.02)→ 拿價格比報酬,完全錯位。
- → 在 `target_transform=log_return` 設定下,**naive baseline 失效**;先前 panel/cascade 在報酬空間報的「naive MASE≈5」等數字**不可信**(同樣的尺度錯位)。
- 影響:報酬空間的「贏過 naive」比較目前都不可靠 —— 因為各模型預測的「空間」不一致(sklearn=報酬、HF=錯尺度、naive=價格)。

## Finding 5 — 因子當 channel 對 channel-independent 模型無效(已於程式碼確認)

- HF `channel_attention=False`(`huggingface/model.py:317`);Lightning `modules.py:108` `reshape(b*c, n, d)` + `:337` 逐 channel encode → **channel-independent**(各 channel 只用自己歷史)。
- 因此把因子當「額外 channel」**根本不影響 Close 的預測** → 解釋 HF/Lightning「with==without 完全相同」。只有 sklearn(flatten 全 channel patch 特徵)會混入,但那裡因子有害。

## 哪些先前結論需要修正

| 先前說法 | 修正 |
| --- | --- |
| dir_accuracy 0.70 = 因子有方向訊號 | ❌ 幾乎=趨勢 base rate(+1.3% over majority);無真實方向技巧 |
| HF 在真實資料「發散」(RMSE 1827) | ❌ 不是發散,是 multi-channel 在報酬空間**預測錯 channel/尺度** |
| 報酬空間「naive MASE≈1.2 / 5」可信 | ⚠️ naive 在報酬空間預測價格 → baseline 失效,跨模型報酬空間比較不可靠 |
| 波動率可預測 | ✅ 成立(模型贏 naive-persistence) |
| channel-independent → 因子當 channel 沒用 | ✅ 程式碼確認 |

## 建議修復(後續 issue)

1. **報酬空間的 target-channel 對齊(HF/Lightning)**:multi-channel 時必須確保「被預測/取出的 channel」就是訓練目標;若 X 不含目標欄,應把目標當獨立 channel 或明確指定 `target_channel_idx`,否則 fail-loud 報錯(現在靜默取 channel 0)。
2. **NaiveModel 尊重 target_transform**:`log_return` 模式下 naive 應預測「0 報酬」(persistence in return space),而非回傳 price;或在報酬空間評估時用報酬版 naive。否則 naive baseline 無意義。
3. **方向評估必須對照 majority baseline**(報 skill_over_majority,而非裸 accuracy)。
4. cascade 因子若要對 HF/Lightning 有效,需改注入法(channel_attention / exogenous head),而非額外 channel。

## 總結(誠實)

- 真正可預測的只有**波動率**;方向是趨勢假象、價格點預測接近不可預測。
- 這次調查的最大收穫不是「模型好壞」,而是揭露**評估層的尺度/channel/baseline 對齊 bug**——這些 bug 讓先前報酬空間的跨模型比較(HF 發散、naive MASE)失真。修好這些,才能對「因子/模型有沒有用」下可靠結論。
