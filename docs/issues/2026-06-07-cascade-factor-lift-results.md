# Cascade Factor-Lift — Real-Data Results (2330.TW)

- **Date**: 2026-06-07
- **Status**: completed (honest evaluation)
- **Setup**: 2330.TW, 2y, seq_len=64, pred_len=15, test_days=120, crossfit k=3,
  `target_transform=log_return`, rolling-origin eval (42 origins), return space.
- **Plan**: `docs/plans/2026-06-07-1131-cascade-factor-forecasting.md`

## Results

### Stage-1 factor quality（因子本身)
| metric | 值 | 解讀 |
| --- | --- | --- |
| `vol_rmse` | 0.0091 | realized-vol 預測誤差小 → **波動率確實可預測**(clustering) |
| `dir_accuracy` | **0.699** | 方向(cross-fit OOF)約 70% → 明顯 > 0.5,**因子本身有訊號** |

### Factor lift（cascade vs 無因子 baseline,RMSE / MASE,報酬空間)
| backend | with factors | without factors | 結論 |
| --- | --- | --- | --- |
| patchtst_sklearn | RMSE 0.0445 / MASE 1.50 | RMSE 0.0398 / MASE 1.33 | 因子讓它**變差** |
| patchtst_huggingface | RMSE 1827 / MASE 76476 | RMSE 1827 / MASE 76476 | with==without **完全相同**;且 HF 數值爆掉(發散) |
| patchtst_lightning | RMSE 0.0263 / MASE 0.858 | RMSE 0.0263 / MASE 0.858 | with==without **完全相同**(MASE<1 但與因子無關) |

## 誠實結論

1. **因子本身可預測**(vol_rmse 低、dir_acc 0.70),但**注入後對價格/報酬預測沒有幫助**:
   sklearn 反而更差;HF/Lightning 完全無變化。
2. **關鍵架構發現:PatchTST 是 channel-independent** —— 每個 channel 只用自己的歷史預測(共享權重
   但不跨 channel 混合)。因此把因子當**額外 channel** 注入,**根本不會影響 Close channel 的預測**
   → 這解釋了 HF/Lightning「with==without 完全相同」。只有 sklearn(把所有 channel 的 patch 統計
   flatten 成單一向量)才真的混入因子,但在那裡因子**有害**。
3. **HF backend 在真實 2y 資料上發散**(RMSE 1827 於報酬空間,顯然爆掉)—— 與 cascade 無關,是 HF
   wrapper 既有的不穩定問題(另案)。
4. 對應先前的誠實分析:**日線價格/報酬點預測接近不可預測**,因子無法拯救;Lightning MASE 0.858<1
   是少數打贏 naive 的個案,但與因子無關。

## 對設計的修正(後續方向)

cascade 的「**因子當 channel**」注入法**不適用於 channel-independent 模型**(HF/Lightning)。若要讓因子
真的影響預測,需改注入方式,例如:
- 用 **channel_attention=True / cross-channel mixing**(讓 Close channel 能 attend 因子 channel);
- 或把因子當 **exogenous covariate** 接到預測 head(而非當作另一條待預測序列);
- 或回到「因子當下游 sklearn 特徵向量」(已驗證:會混入,但此資料上無益)。

**但更根本的結論不變**:在日線股價這個問題上,換注入法或換模型**不太可能把點預測拉到顯著贏 naive**;
因子的價值更可能在**輸出波動率/方向本身**(vol 可預測、dir~0.70),而非改善價格點預測。

## v1 交付狀態
- cascade 機制(Stage-1 因子 + cross-fit 無洩漏 + Stage-2 三 backend)**可運作、有測試**(全測 685 passed)。
- factor-lift 評估**誠實呈現**:目前因子對價格預測無正面 lift —— 這正是引入對照 baseline 的目的。
