# Issue: level-anchoring / mean-reversion failure on trending series — 2026-06-22

> 為什麼 PatchTST 在**股票**上也輸 naive,而且輸得比 FX 更慘。實測診斷 + 根因。

## 症狀
1-step walk-forward(預測明天收盤;`naive` = 明天=今天),實測 **2330.TW**(近一年大漲):

```
2330   訓練均值 1481  →  測試均值 2251         (近一年 ~+50% 的漲勢)
NAIVE  RMSE 47.2     corr(naive, 實際) = 0.91
MODEL  RMSE 426.7    corr(model, 實際) = 0.06    (patchtst_lightning)
       預測均值 1840  vs  實際均值 2251  →  系統性偏低 -412 (~-18%)
       預測範圍 [1782, 1886](幾乎平的)  vs  實際範圍 [2020, 2510]
```

模型 RMSE 是 naive 的 **~9 倍**,且 corr 僅 0.06(**完全沒在跟價格走**),預測幾乎是一條 ~1840 的水平線。

## 機制(根因)
- 模型是**單變量(只用 Close)+ 全域 `StandardScaler`,且 scaler 只在訓練段擬合**。
- 它學到「往訓練分布中心回歸」→ 輸出卡在**訓練期價位 ~1840 附近、近乎常數**。
- 但測試期股價已漲到 2020–2510,**遠離訓練中心** → 模型等於在漲勢中**一路預測「會跌回 1840」**。
- 一句話根因:**用「絕對價位水準」+ 訓練段 scaler 去預測非平穩(趨勢)序列** —— 模型被鎖在「訓練期的價位空間」,測試價格一離開那個區間就崩。

## 為什麼「股票比 FX 慘」
- FX(USDTWD)接近區間震盪:訓練均值 ≈ 測試均值 → level-anchoring 偏差小(輸 naive ~2–5×)。
- 強趨勢股票:訓練均值 ≪ 現價 → 偏差爆掉(輸 ~9×)。**越有趨勢的標的,這個錯越大。**

## 對兩個常見直覺的澄清
1. **不是「輸給平盤 naive 因為漲勢」**:相反,**naive(明天=今天)在漲勢裡追得很好**(corr 0.91),因為日線「明天≈今天」極強(每日波動 ~2% ≪ 慢漲)。
2. **模型不是在追漲**:它**反向 mean-revert 往下預測**,所以同時**輸 naive、又方向錯**。

## 影響範圍
- 確認於 `patchtst_lightning`;`patchtst_huggingface`(同為單變量 + 全域 scaler)機制相同;`patchtst_sklearn`(已改真多步)仍是絕對價位 + 全域 scaler,同病。
- eval(1-step)量到此問題;個股 chart 的多步 forecast 也呈現同向偏差(2330 forecast 第 1 步 2305.9 < 現價 2510)。

## 修正方向(兩條,各成一份 plan)
- **Plan A — 預測「報酬/變化」而非「絕對價位」**(`2026-06-22-fix-plan-a-predict-returns.md`):target 改 log-return / diff,模型預測變化量,再從最後已知價累積還原 → level-agnostic、平穩。
- **Plan B — 逐 window 實例正規化(RevIN / instance norm)**(`2026-06-22-fix-plan-b-instance-norm.md`):每個輸入視窗用**自己**的最後值/均值標準化、預測、再用同視窗統計還原 → 解除「訓練期價位」錨定;這也是 PatchTST 原論文處理 distribution shift 的標準做法。

## 重現
`uv run python` 跑 `scripts/evaluate_patchtst.py` 的 `build_features` + 對 `2330.TW`(DataStorage)做 1-step walk-forward,印出 naive/model 的 RMSE、corr、預測均值/範圍即可重現上表。

## 文獻佐證(2026-06-22 web research)
查了 PatchTST 股市預測(英文 + 繁中各一輪),三份來源都佐證本診斷與兩條修正方向(逐源筆記在 `docs/references/external_sites/`):

- **RevIN(`2026-06-22-revin-instance-normalization.md`)** — RevIN = 逐 instance normalize → 預測 → denormalize,在輸入層移除非平穩的 mean/variance、輸出層還原;專治 distribution shift。這正是 **Plan B**;我們 wrapper 疊一層**全域 train-fit `StandardScaler`** 恰好把它廢掉、把預測錨在訓練期價位。
- **MQL5 PatchTST 價格預測(`2026-06-22-mql5-patchtst-price-action.md`)** — 實作預測**絕對 OHLC 價位**、也用 RevIN,但**完全沒跟 naive baseline 比**("predictive superiority remains undemonstrated"),且預測會 ~20–30% 反向 —— 與我們 corr 0.06 一致。佐證:就算裝了 RevIN、預測「價位」仍未必贏 naive,我們誠實做 naive 對照是對的紀律。
- **quant67 量化時序陷阱(繁中)(`2026-06-22-quant67-dl-timeseries-pitfalls.md`)** — 直接點名:**單資產報酬率預測信噪比極低、DL 難贏簡單 baseline**;建議**「避免預測絕對價格,改用報酬率差分 / 截面排名 / 分位數」** → 正好背書 **Plan A(預測報酬)**;且「日頻選股是 GBDT 的主場、序列模型稀釋截面信號」→ 佐證 **Two-School Screener 用規則 / 非 ML forecast**、ML 預測僅供展示。

**結論:** 這是 modern TSF 的教科書級已知問題。Plan A(預測報酬,quant67 明確建議)修「偏差/非平穩」;Plan B(RevIN/instance-norm,PatchTST 原生)修「distribution shift」;兩者正交可疊加。但即使修好,單資產 price/return 預測仍可能僅略勝或打平 naive —— 故 ML forecast 維持 **experimental / display-only**、選股策略走規則派,方向正確。台股專屬的 PatchTST 研究目前查無(屬空白)。
