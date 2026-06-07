# Two-Step ML for Precipitation (GNSS-PWV)

- **URL**: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12563812/
- **Fetched**: 2026-06-07
- **Topic**: 氣象「先預測中間變數再預測降雨」的二階段法(cascade 因子設計的對照)

## 重點
- 二階段框架:**Stage-1 Random Forest** 先用 PWV(可降水量)、地面氣象參數與輔助大氣變數
  估計「當前降水」;**Stage-2 LSTM** 再用時間依賴預測下一小時降水。
- 核心概念:**先預測/估計中間物理因子,再餵給下游模型預測最終目標** —— 與本專案 cascade
  (先測 vol/方向因子 → 再測股價)同構。
- 其他降雨研究也大量用「多個中間大氣變數」當 predictor(溫度、濕度、渦度、雲量、位勢高度等)。

## 對本專案的啟示
- 驗證 cascade 設計方向:中間因子(波動率、方向)當下游價格預測的輸入是成熟做法。
- 氣象用「物理可預測」的中間量;金融對應「波動率」(clustering 可預測),這正是我們選 vol 當因子的理由。
