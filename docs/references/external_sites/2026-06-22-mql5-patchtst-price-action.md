# MQL5 — Using PatchTST for predicting the next 24 hours of price action

- **URL:** https://www.mql5.com/en/articles/15198
- **Accessed:** 2026-06-22
- **Why read:** a practical (non-academic) PatchTST-on-price write-up — does it predict levels or returns, does it use RevIN, and does it honestly compare to a naive baseline?

## Key takeaways
- Predicts **absolute OHLC price levels** (not returns) over a 24h horizon — same approach as our model.
- Uses **RevIN** explicitly: `x=(x-mean)/std`, denorm `x*std+mean`, plus a learnable `affine_bias` for skew/kurtosis — to fight distribution shift ("our trained EA no longer seems to predict the market").
- **Critically: no comparison against a naive baseline** (random walk / persistence). Training loss drops, but "actual predictive superiority remains undemonstrated." → mirrors our finding: people rarely check vs naive; when you do (as we did), the edge doesn't hold.
- Honest practitioner caveats: high false-positive rate, predictions **"invert" ~20–30%** of the time (direction flips) — consistent with our corr(model,actual)=0.06; expect ~0.5–1.5% weekly gain vs 1–2.5% risk; needs technical-pattern confirmation + risk management.
- **Implication for us:** even a RevIN-equipped, level-predicting PatchTST is not demonstrably better than naive on price; our honest naive comparison is the right discipline, and "experimental, display-only" framing is justified.
