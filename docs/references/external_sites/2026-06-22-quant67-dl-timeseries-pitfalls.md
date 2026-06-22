# quant67 — 量化交易的時序深度學習:實踐與陷阱 (TCN/Transformer/PatchTST)

- **URL:** https://quant67.com/post/quant/13-deep-learning-time-series/13-deep-learning-time-series.html
- **Accessed:** 2026-06-22
- **Why read:** Traditional-Chinese practitioner view on DL time-series for stocks — does it confirm "predict price → loses to naive", and what's the recommended fix?

## Key takeaways (directly validates our issue + plans)
- **"單資產收益率預測信噪比極低,深度學習難以超越簡單基線"** — single-asset return prediction has very low SNR; DL struggles to beat simple baselines. = our naive-beats-model finding.
- **"避免預測絕對價格,改為:報酬率差分、截面排名、分位數預測"** — **avoid predicting absolute price; use returns/diffs, cross-sectional rank, quantiles.** → directly endorses **Fix Plan A (predict returns)**, and the **cross-sectional rank** = our Two-School Screener.
- **"日頻選股是 GBDT 的主場,序列模型反而稀釋截面信號"** — daily stock-selection is GBDT's domain; sequence models dilute the cross-sectional signal. → validates the Two-School Screener being **rule-based / not ML-forecast** (ML display-only).
- **"殘差學習(學 baseline 之外的部分)優於直接替換"** — residual learning (model the part beyond a baseline) beats replacing it — i.e. predict the residual vs naive, not the level.
- Train-inference normalization must be **identical** (normalization口徑一致) — our global-scaler bug is exactly a train/inference normalization mismatch in spirit.
- Other pitfalls: future leakage on 停牌/涨跌停 days (mask them), high seed-sensitivity (multi-seed ensemble), winsorize extremes, DL eng cost 3–5× GBDT.
- **Pragmatic conclusion:** DL is a niche tool (HF microstructure / multivariate), not a better algorithm; most daily stock-selection is GBDT's ceiling. → reinforces keeping ML forecast **display-only** and the screener rule-based.
