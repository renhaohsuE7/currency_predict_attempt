# Factor-Augmented Volatility & Two-Stage Stock Forecasting

- **URLs**:
  - https://arxiv.org/html/2508.01880 (Time-Varying Factor-Augmented Models for Volatility Forecasting)
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC12064028/ (Two-stage RF feature-selection + BiGRU for stock indices)
- **Fetched**: 2026-06-07
- **Topic**: 金融「波動率/方向當因子、二階段預測」的文獻(cascade 設計的金融對照)

## 重點
- **Factor-augmented volatility**:從 realized volatilities 直接抽取**時變潛在因子**,因子與載荷隨市場
  regime 即時調整 → 用來預測波動率。佐證「波動率可由因子結構預測」。
- **Two-stage(RF → BiGRU)**:Stage-1 用 random forest 子集特徵選擇,從 high/low/close/volume 等
  挑出最佳特徵子集;Stage-2 BiGRU+attention 預測股價。= 二階段「先選/造因子 → 再預測價格」。
- 關鍵論點:**「波動率通常可被預測 → 意味報酬的『方向(sign)』也可被預測」** —— 直接支持把
  vol + direction 當下游因子。

## 對本專案的啟示
- 金融端佐證 cascade:vol/direction 是合理且有文獻支持的中間因子。
- 但仍須誠實驗證 **factor lift**(vs 無因子 baseline)—— 文獻說「可預測」不代表在我們的日線設定一定
  贏 naive;以對照實驗為準。
